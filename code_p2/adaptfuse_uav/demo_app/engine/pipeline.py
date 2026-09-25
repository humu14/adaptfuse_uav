"""
Per-frame analysis: one video frame + the audio window ending at its timestamp
-> boxes, scene / victim / nuisance labels, RUE reliability, audio labels.
Labels are EMA-smoothed (tau = 0.5 s), boxes must persist 2 of 3 analyzed frames, and an
agreement flag compares scene and detector (engine/temporal.py).

Preprocessing mirrors datasets/multimodal_dataset.py (eval split):
  RGB     : BGR->RGB, resize 224, ImageNet mean/std
  thermal : greyscale, resize 224, per-image z-score, 1 channel
  audio   : 2 s @ 16 kHz, 64-mel log-power (ref=max), z-score, 63 frames
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import librosa
import numpy as np
import torch

from .clahe import clahe_bgr
from .labels import AUDIOSET_RELEVANT, DISASTER_CLASSES, NUISANCE_CLASSES, VICTIM_CLASSES
from .media import AudioTrack
from .models import DETECTOR_IMGSZ, ModelBundle
from .temporal import TemporalState

IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AUDIO_WINDOW_S = 2.0
AUDIO_STEP_S = 0.5            # audio labels refresh every 0.5 s of video time
SILENCE_RMS = 1e-4


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()


def _top(probs: np.ndarray, names: list) -> dict:
    probs = np.asarray(probs, dtype=np.float32)
    i = int(probs.argmax())
    return {"label": names[i], "index": i, "conf": float(probs[i]),
            "probs": {n: float(p) for n, p in zip(names, probs)}}


def _rgb_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def _thermal_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    g = cv2.resize(g, (IMG, IMG), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return torch.from_numpy(g)[None, None]


def _logmel(y16: np.ndarray) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(y=y16, sr=16000, n_mels=64, hop_length=512, n_fft=1024)
    lm = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    lm = (lm - lm.mean()) / (lm.std() + 1e-6)
    T = lm.shape[-1]
    lm = lm[:, :63] if T >= 63 else np.pad(lm, ((0, 0), (0, 63 - T)), constant_values=lm.min())
    return torch.from_numpy(lm)[None, None]


class Analyzer:
    """Stateful per-video analyzer (holds the audio track and the audio-label cache)."""

    def __init__(self, bundle: ModelBundle, audio: Optional[AudioTrack], modality: str = "rgb",
                 reset_gap: float = 2.0):
        self.m = bundle
        self.audio = audio
        self.modality = modality
        self._audio_key = None
        self._audio_state: dict = {}
        self._audio_tensor: torch.Tensor = torch.zeros(1, 1, 64, 63)
        self._has_audio = 0.0
        self.temporal = TemporalState(scene_names=DISASTER_CLASSES, reset_gap=reset_gap)

    def reset(self) -> None:
        """Forget smoothing / box history (new video position or new input condition)."""
        self.temporal.reset()

    def set_modality(self, modality: str) -> None:
        if modality not in ("rgb", "thermal"):
            raise ValueError(modality)
        if modality != self.modality:
            self.modality = modality
            self.reset()

    # ── audio ────────────────────────────────────────────────────────────────
    def _update_audio(self, t: float) -> None:
        key = int(t / AUDIO_STEP_S)
        if key == self._audio_key:
            return
        self._audio_key = key
        if self.audio is None:
            self._audio_state = {"status": "none", "note": "video has no audio track"}
            self._audio_tensor, self._has_audio = torch.zeros(1, 1, 64, 63), 0.0
            return

        y16 = self.audio.window(t, AUDIO_WINDOW_S, 16000)
        rms = float(np.sqrt(np.mean(y16 ** 2)))
        if rms < SILENCE_RMS:
            self._audio_state = {"status": "silent", "rms": rms}
            self._audio_tensor, self._has_audio = torch.zeros(1, 1, 64, 63), 0.0
            return

        dev = self.m.device
        au = _logmel(y16)
        y32 = torch.from_numpy(self.audio.window(t, AUDIO_WINDOW_S, 32000))[None]
        with self.m.lock, torch.no_grad():
            d, v, _ = self.m.audio_only(audio=au.to(dev))
            tags = self.m.tagger(y32.to(dev))[0].cpu().numpy()

        relevant = sorted(((float(tags[i]), i) for i in AUDIOSET_RELEVANT), reverse=True)[:5]
        top_any = int(tags.argmax())
        self._audio_state = {
            "status": "ok",
            "rms": rms,
            "window_s": AUDIO_WINDOW_S,
            "window_end": round(t, 2),
            "pipeline": {
                "disaster": _top(_softmax(d), DISASTER_CLASSES),
                "victim": _top(_softmax(v), VICTIM_CLASSES),
            },
            "tags": [{"name": self.m.audioset_names[i], "group": AUDIOSET_RELEVANT[i], "score": s}
                     for s, i in relevant],
            "top_any": {"name": self.m.audioset_names[top_any], "score": float(tags[top_any])},
        }
        self._audio_tensor, self._has_audio = au, 1.0

    # ── detection ────────────────────────────────────────────────────────────
    def _detect(self, frame_bgr, det_conf: float, person_conf: float) -> list:
        dev = self.m.device
        enhanced = clahe_bgr(frame_bgr)            # the D-Fire detector was trained on CLAHE images
        with self.m.lock:
            fr = self.m.fire_det.predict(enhanced, imgsz=DETECTOR_IMGSZ, conf=det_conf,
                                         verbose=False, device=dev)[0]
            pr = self.m.person_det.predict(frame_bgr, imgsz=640, conf=person_conf, classes=[0],
                                           verbose=False, device=dev)[0]
        boxes = []
        for r, source in ((fr, "dfire"), (pr, "coco")):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxyn.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                boxes.append({"cls": r.names[int(k)], "conf": float(c), "source": source,
                              "box": [float(x) for x in b]})
        return boxes

    # ── main entry ───────────────────────────────────────────────────────────
    def analyze(self, frame_bgr: np.ndarray, t: float, det_conf: float = 0.25,
                person_conf: float = 0.35) -> dict:
        t0 = time.perf_counter()
        self._update_audio(t)
        t_audio = time.perf_counter()

        boxes = self._detect(frame_bgr, det_conf, person_conf)
        t_det = time.perf_counter()

        if self.modality == "rgb":
            rgb, has_rgb = _rgb_tensor(frame_bgr), 1.0
            th, has_th = torch.zeros(1, 1, IMG, IMG), 0.0
        else:
            rgb, has_rgb = torch.zeros(1, 3, IMG, IMG), 0.0
            th, has_th = _thermal_tensor(frame_bgr), 1.0
        with self.m.lock:
            cls = self.m.classifier.predict(rgb, th, self._audio_tensor,
                                            (has_rgb, has_th, self._has_audio), self.modality)
        t_cls = time.perf_counter()

        raw = {"disaster": cls["disaster"], "victim": cls["victim"], "nuisance": cls["nuisance"]}
        smooth, confirmed, agreement = self.temporal.update(float(t), raw, boxes)
        rel, unc = cls["reliability"], cls["uncertainty"]
        return {
            "t": round(float(t), 3),
            "modality": self.modality,
            "has": {"rgb": bool(has_rgb), "thermal": bool(has_th), "audio": bool(self._has_audio)},
            "boxes": confirmed,
            "raw_boxes": boxes,
            "disaster": _top(smooth["disaster"], DISASTER_CLASSES),
            "victim": _top(smooth["victim"], VICTIM_CLASSES),
            "nuisance": _top(smooth["nuisance"], NUISANCE_CLASSES),
            "raw": {"disaster": _top(raw["disaster"], DISASTER_CLASSES),
                    "victim": _top(raw["victim"], VICTIM_CLASSES),
                    "nuisance": _top(raw["nuisance"], NUISANCE_CLASSES)},
            "agreement": agreement,
            "reliability": {"rgb": float(rel[0]), "thermal": float(rel[1]), "audio": float(rel[2])},
            "uncertainty": {"rgb": float(unc[0]), "thermal": float(unc[1]), "audio": float(unc[2])},
            "audio": self._audio_state,
            "timing_ms": {
                "audio": round((t_audio - t0) * 1e3, 1),
                "detect": round((t_det - t_audio) * 1e3, 1),
                "classify": round((t_cls - t_det) * 1e3, 1),
                "total": round((t_cls - t0) * 1e3, 1),
            },
        }
