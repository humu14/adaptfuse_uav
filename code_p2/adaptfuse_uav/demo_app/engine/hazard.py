"""
Zero-shot hazard recognition from one image-encoder forward (ViT-B/16, CLIP space):

* scene: the pooled embedding against prompt-ensemble text embeddings -> probabilities
  over HAZARD_EVENTS (normal, fire, smoke, explosion, collapsed building, flood,
  landslide, traffic accident), plus a per-event logit bias tuned on the grouped val split
* zones: dense patch embeddings from the last block taken ClearCLIP-style (q-q attention,
  no residual, no MLP) against the same text embeddings -> per-patch event map ->
  connected regions -> boxes for collapsed building / flood / landslide / accident

The text side is precomputed by scripts/export_hazard.py (weights/hazard_text.pt), so
only the vision tower ships and runs. The frame keeps its aspect ratio: short side
224 px, long side up to 448 px, with interpolated position embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .labels import HAZARD_EVENTS, HAZARD_NAMES, ZONE_EVENTS

PATCH = 16
SHORT = 224
LONG_MAX = 448
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
COARSE = np.array([e["coarse"] for e in HAZARD_EVENTS])


def input_size(w: int, h: int) -> tuple[int, int]:
    """(width, height), both multiples of 16, short side 224, aspect kept up to 2:1."""
    def snap(x):
        return int(min(LONG_MAX, max(SHORT, round(x / PATCH) * PATCH)))
    return (snap(SHORT * w / h), SHORT) if w >= h else (SHORT, snap(SHORT * h / w))


def to_coarse(p_event: np.ndarray, n: int = 4) -> np.ndarray:
    """Sum event probabilities into the 4 DISASTER_CLASSES."""
    p_event = np.asarray(p_event, dtype=np.float32)
    out = np.zeros(p_event.shape[:-1] + (n,), dtype=np.float32)
    for i, c in enumerate(COARSE):
        out[..., c] += p_event[..., i]
    return out


def softmax_bias(logits: np.ndarray, bias) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) + (0.0 if bias is None else np.asarray(bias, dtype=np.float64))
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def fuse(p_adaptfuse: np.ndarray, p_event: np.ndarray, w: float, t_af: float = 1.0, t_zs: float = 1.0) -> np.ndarray:
    """Scene probabilities as a weighted product of the two temperature-scaled experts:
    softmax(w / t_af * log p_AdapFuse + (1 - w) / t_zs * log coarse(p_event))."""
    la = np.log(np.clip(np.asarray(p_adaptfuse, dtype=np.float64), 1e-8, 1.0))
    lz = np.log(np.clip(to_coarse(p_event).astype(np.float64), 1e-8, 1.0))
    return softmax_bias(w / t_af * la + (1 - w) / t_zs * lz, None)


def load_hazard_calibration(path: Path) -> dict:
    default = {"bias": {}, "fusion_weight": {}, "temperature": {}, "zone": {}}
    path = Path(path)
    if not path.exists():
        print(f"[hazard] WARNING: {path.name} not found - zero bias, default fusion weights")
        return default
    return {**default, **json.loads(path.read_text(encoding="utf-8"))}


class HazardModel:
    def __init__(self, vision_dir: Path, text_path: Path, device, calibration: dict | None = None):
        from transformers import CLIPVisionModelWithProjection
        self.device = torch.device(device)
        # fp16 only pays off with tensor cores (compute capability >= 7); older GPUs run fp32 faster
        self.dtype = (torch.float16 if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 7
                      else torch.float32)
        self.model = CLIPVisionModelWithProjection.from_pretrained(str(vision_dir)).to(self.device, self.dtype).eval()
        txt = torch.load(text_path, map_location="cpu", weights_only=False)
        if list(txt["names"]) != HAZARD_NAMES:
            raise ValueError(f"{Path(text_path).name} was built for {txt['names']}, engine expects {HAZARD_NAMES}; "
                             "re-run scripts/export_hazard.py")
        self.text = txt["text"].float().to(self.device)            # (E, D), L2-normalised
        self.scale = float(txt["logit_scale"])
        self.cal = calibration or {"bias": {}, "fusion_weight": {}, "temperature": {}, "zone": {}}

    def bias(self, condition: str):
        return self.cal["bias"].get(condition)

    def fusion_weight(self, condition: str) -> float:
        """Weight of the AdapFuse expert in the fused scene (rest: zero-shot)."""
        return float(self.cal["fusion_weight"].get(condition, 0.35))

    def fuse(self, p_adaptfuse: np.ndarray, p_event: np.ndarray, condition: str) -> np.ndarray:
        t = self.cal.get("temperature", {}).get(condition, {})
        return fuse(p_adaptfuse, p_event, self.fusion_weight(condition), t.get("adaptfuse", 1.0), t.get("zero_shot", 1.0))

    def _tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        h, w = frame_bgr.shape[:2]
        size = input_size(w, h)
        img = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    @torch.no_grad()
    def logits(self, frame_bgr: np.ndarray, dense: bool = True):
        """Return (scene_logits (E,), dense_logits (gh, gw, E) or None) before bias / softmax."""
        px = self._tensor(frame_bgr).to(self.device, self.dtype)
        gh, gw = px.shape[2] // PATCH, px.shape[3] // PATCH
        vm = self.model.vision_model
        out = vm(pixel_values=px, output_hidden_states=True, interpolate_pos_encoding=True)
        cls = self.model.visual_projection(out.pooler_output).float()
        cls = cls / cls.norm(dim=-1, keepdim=True)
        scene = (self.scale * cls @ self.text.T)[0].cpu().numpy()
        if not dense:
            return scene, None
        last = vm.encoder.layers[-1]
        x = last.layer_norm1(out.hidden_states[-2])
        at = last.self_attn
        B, N, D = x.shape
        q = at.q_proj(x).view(B, N, at.num_heads, at.head_dim).transpose(1, 2)
        v = at.v_proj(x).view(B, N, at.num_heads, at.head_dim).transpose(1, 2)
        a = torch.softmax((q @ q.transpose(-1, -2)) * at.scale, dim=-1)
        o = at.out_proj((a @ v).transpose(1, 2).reshape(B, N, D))
        d = self.model.visual_projection(vm.post_layernorm(o[:, 1:])).float()
        d = d / d.norm(dim=-1, keepdim=True)
        dl = (self.scale * d @ self.text.T)[0].reshape(gh, gw, -1).cpu().numpy()
        return scene, dl

    def predict(self, frame_bgr: np.ndarray, condition: str = "rgb", dense: bool = True):
        """(event probabilities (E,), dense event probabilities (gh, gw, E) or None)."""
        s, d = self.logits(frame_bgr, dense)
        b = self.bias(condition)
        return softmax_bias(s, b), (softmax_bias(d, b) if d is not None else None)


def extract_zones(dense: np.ndarray, scene: np.ndarray, gate: float = 0.25, thr: float = 0.5,
                  min_frac: float = 0.02) -> tuple[list, dict]:
    """Regions of the dense map for zone events that the scene itself supports.

    dense: (gh, gw, E) per-patch event probabilities, scene: (E,) event probabilities.
    Returns (zones, grid): zones are boxes in normalised xyxy, grid is the per-cell event
    index (-1 = none) and probability for drawing a heat layer.
    """
    gh, gw, _ = dense.shape
    cell_cls = np.full((gh, gw), -1, dtype=np.int16)
    cell_p = np.zeros((gh, gw), dtype=np.float32)
    zones = []
    for name, display in ZONE_EVENTS.items():
        ci = HAZARD_NAMES.index(name)
        if scene[ci] < gate:
            continue
        m = cv2.GaussianBlur(dense[..., ci].astype(np.float32), (3, 3), 0)
        mask = (m > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < max(2, min_frac * gh * gw):
                continue
            comp = cc == k
            conf = float(m[comp].mean()) * float(np.sqrt(scene[ci]))
            zones.append({"cls": display, "event": name, "conf": conf, "source": "zone",
                          "box": [x / gw, y / gh, (x + w) / gw, (y + h) / gh]})
            take = comp & (m > cell_p)
            cell_cls[take], cell_p[take] = ci, m[take]
    zones.sort(key=lambda z: -z["conf"])
    grid = {"w": int(gw), "h": int(gh), "cls": cell_cls.flatten().tolist(),
            "p": [round(float(x), 2) for x in cell_p.flatten()]}
    return zones[:6], grid
