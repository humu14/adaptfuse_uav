"""
Scene classifier used by the demo: an ensemble of AdapFuse-V1 members (grouped split,
seeds 41 and 42), optional horizontal-flip TTA, and a per-condition logit bias on the
disaster head tuned on the grouped val split (weights/calibration.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HEADS = ("disaster", "victim", "nuisance")


@torch.no_grad()
def member_outputs(model, rgb, th, au, has) -> dict:
    """One AdapFuseV1 forward on a batch; `has` is (B, 3) = [rgb, thermal, audio] flags."""
    d, v, rel = model(rgb, th, au, has[:, 0], has[:, 1], has[:, 2])
    extra = model.get_full_outputs()
    out = {
        "disaster": torch.softmax(d.float(), -1),
        "victim": torch.softmax(v.float(), -1),
        "nuisance": torch.softmax(extra["nuisance"].float(), -1),
        "reliability": rel.float(),
        "uncertainty": extra["uncertainty"].float(),
    }
    return {k: t.detach().cpu() for k, t in out.items()}


def apply_bias(p: np.ndarray, bias) -> np.ndarray:
    """softmax(log p + b) along the last axis; bias=None leaves p unchanged."""
    if bias is None:
        return p
    z = np.log(np.clip(p, 1e-8, 1.0)) + np.asarray(bias, dtype=np.float64)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def load_calibration(path: Path) -> dict:
    default = {"stage": "ensemble+tta", "members": None, "tta": True, "bias": {}}
    path = Path(path)
    if not path.exists():
        print(f"[classifier] WARNING: {path.name} not found - using both members, flip TTA, zero bias")
        return default
    return {**default, **json.loads(path.read_text(encoding="utf-8"))}


class SceneClassifier:
    def __init__(self, members: list, device, tta: bool = True, bias: dict | None = None):
        if not members:
            raise ValueError("SceneClassifier needs at least one member model")
        self.members = members
        self.device = device
        self.tta = tta
        self.bias = bias or {}

    def predict(self, rgb, th, au, has, condition: str) -> dict:
        if self.tta:
            rgb = torch.cat([rgb, torch.flip(rgb, dims=[3])])
            th = torch.cat([th, torch.flip(th, dims=[3])])
            au = torch.cat([au, au])
        flags = torch.tensor([list(has)], dtype=torch.float32).repeat(rgb.shape[0], 1)
        dev = self.device
        outs = [member_outputs(m, rgb.to(dev), th.to(dev), au.to(dev), flags.to(dev))
                for m in self.members]
        res = {k: _nan_safe_mean(torch.stack([o[k] for o in outs]), probs=k in HEADS) for k in outs[0]}
        res["disaster"] = apply_bias(res["disaster"], self.bias.get(condition))
        return res


def _nan_safe_mean(x: torch.Tensor, probs: bool) -> np.ndarray:
    """Mean over (members, views), skipping rows that came back NaN; uniform / 0 if all did."""
    x = x.reshape(-1, x.shape[-1]).numpy()
    finite = ~np.isnan(x).any(axis=1)
    if finite.any():
        return x[finite].mean(axis=0)
    k = x.shape[-1]
    return np.full(k, 1.0 / k if probs else 0.0, dtype=np.float32)
