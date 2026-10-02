"""
Per-frame analysis: one video frame + the audio window ending at its timestamp
-> boxes, hazard zones, scene / event / victim / nuisance labels, RUE reliability,
sound events.

Scene = weighted product of the AdapFuse ensemble and the zero-shot hazard model
(engine/hazard.py fuse(); weight and temperatures per input condition from
weights/hazard_calibration.json); the event label (collapsed building, flood, ...) is the
most likely hazard event inside the winning scene class.
The hazard model and the audio networks refresh every 0.5 s of video time.
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
from .hazard import extract_zones
from .labels import (COCO_KEEP, DISASTER_CLASSES, HAZARD_EVENTS, HAZARD_NAMES, NUISANCE_CLASSES,
                     SOUND_GROUPS, VICTIM_CLASSES)
from .media import AudioTrack
from .models import DETECTOR_IMGSZ, ModelBundle
from .temporal import TemporalState, ema_alpha

IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AUDIO_WINDOW_S = 2.0          # AdapFuse audio branch / AudioCNN input (as in training)
AUDIO_STEP_S = 0.5            # audio labels refresh every 0.5 s of video time
HAZARD_STEP_S = 0.5           # hazard model refreshes every 0.5 s of video time
SILENCE_RMS = 1e-4
SOUND_TAU = 1.0               # EMA time constant of sound-event scores (s)
CONTEXT_THR = 0.3             # threshold for sound groups without a tuned one


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


def event_label(scene: np.ndarray, event: np.ndarray) -> dict:
    """Most likely hazard event inside the winning scene class; conf is the scene probability."""
    c = int(np.argmax(scene))
    cand = [i for i, e in enumerate(HAZARD_EVENTS) if e["coarse"] == c]
    i = max(cand, key=lambda k: event[k])
    return {"label": HAZARD_NAMES[i], "index": i, "conf": float(scene[c]), "group": DISASTER_CLASSES[c],
            "probs": {n: float(p) for n, p in zip(HAZARD_NAMES, event)}}


def sound_events(scores: np.ndarray, thresholds: dict) -> list:
    """Sound groups, strongest first relative to their threshold (active = score >= threshold)."""
    out = []
    for (name, g), s in zip(SOUND_GROUPS.items(), scores):
        thr = float(thresholds.get(name, CONTEXT_THR))
        out.append({"name": name, "score": float(s), "thr": thr, "hazard": bool(g["hazard"]),
                    "active": bool(s >= thr)})
    out.sort(key=lambda e: -e["score"] / e["thr"])
    return out


def headline(events: list) -> dict:
    """The sound to report: strongest active hazard group, else strongest active context group."""
    for want_hazard in (True, False):
        act = [e for e in events if e["active"] and e["hazard"] == want_hazard]
        if act:
            return {"name": act[0]["name"], "score": act[0]["score"], "hazard": want_hazard}
    return {"name": "background noise", "score": 0.0, "hazard": False}


class Analyzer:
    """Stateful per-video analyzer (holds the audio track and the audio / hazard caches)."""

    def __init__(self, bundle: ModelBundle, audio: Optional[AudioTrack], modality: str = "rgb",
                 reset_gap: float = 2.0):
        self.m = bundle
        self.audio = audio
        self.modality = modality
        self.reset_gap = reset_gap
        self._audio_key = None
        self._audio_state: dict = {}
        self._audio_tensor: torch.Tensor = torch.zeros(1, 1, 64, 63)
        self._has_audio = 0.0
        self._sound = None            # EMA of sound-group scores
        self._sound_t = None
        self._hz_key = None
        self._hz_event = np.full(len(HAZARD_NAMES), 1.0 / len(HAZARD_NAMES), dtype=np.float32)
        self._zones: list = []
        self._zone_grid = None
        self.temporal = TemporalState(scene_names=DISASTER_CLASSES, reset_gap=reset_gap)

    def reset(self) -> None:
        """Forget smoothing / box history (new video position or new input condition)."""
        self.temporal.reset()
        self._sound, self._sound_t = None, None

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
        ytag = self.audio.window(t, self.m.tag_window, self.m.tagger.SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(y16 ** 2)))
        if max(rms, float(np.sqrt(np.mean(ytag ** 2)))) < SILENCE_RMS:
            self._audio_state = {"status": "silent", "rms": rms}
            self._audio_tensor, self._has_audio = torch.zeros(1, 1, 64, 63), 0.0
            self._sound, self._sound_t = None, None
            return

        dev = self.m.device
        au = _logmel(y16)
        with self.m.lock, torch.no_grad():
            d, v, _ = self.m.audio_only(audio=au.to(dev))
            tags = self.m.tagger(torch.from_numpy(ytag)[None].to(dev))[0].float().cpu().numpy()

        groups = np.array([tags[g["idx"]].max() for g in SOUND_GROUPS.values()], dtype=np.float32)
        if self._sound is None or t < self._sound_t or t - self._sound_t > self.reset_gap:
            self._sound = groups
        else:
            self._sound = self._sound + ema_alpha(t - self._sound_t, SOUND_TAU) * (groups - self._sound)
        self._sound_t = t
        events = sound_events(self._sound, self.m.sound_thr)
        top = np.argsort(-tags)[:5]
        self._audio_state = {
            "status": "ok",
            "rms": rms,
            "window_s": self.m.tag_window,
            "window_end": round(t, 2),
            "headline": headline(events),
            "events": events,
            "victim_cue": next(e for e in events if SOUND_GROUPS[e["name"]].get("victim")),
            "top_tags": [{"name": self.m.audioset_names[i], "score": float(tags[i])} for i in top],
            "pipeline": {
                "disaster": _top(_softmax(d), DISASTER_CLASSES),
                "victim": _top(_softmax(v), VICTIM_CLASSES),
            },
        }
        self._audio_tensor, self._has_audio = au, 1.0

    # ── hazard model ─────────────────────────────────────────────────────────
    def _update_hazard(self, frame_bgr: np.ndarray, t: float) -> None:
        key = (int((t + HAZARD_STEP_S / 2) / HAZARD_STEP_S), self.modality)   # offset from audio refresh
        if key == self._hz_key:
            return
        self._hz_key = key
        hz = self.m.hazard
        with self.m.lock:
            ev, dense = hz.predict(frame_bgr, self.modality, dense=self.modality == "rgb")
        self._hz_event = ev
        if dense is None:                               # thermal: no zone maps
            self._zones, self._zone_grid = [], None
        else:
            self._zones, self._zone_grid = extract_zones(dense, ev, **hz.cal.get("zone", {}))

    # ── detection ────────────────────────────────────────────────────────────
    def _detect(self, frame_bgr, det_conf: float, person_conf: float) -> list:
        dev = self.m.device
        enhanced = clahe_bgr(frame_bgr)            # the D-Fire detector was trained on CLAHE images
        with self.m.lock:
            fr = self.m.fire_det.predict(enhanced, imgsz=DETECTOR_IMGSZ, conf=det_conf,
                                         verbose=False, device=dev)[0]
            pr = self.m.person_det.predict(frame_bgr, imgsz=640, conf=person_conf, classes=list(COCO_KEEP),
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
        self._update_hazard(frame_bgr, t)
        t_cls = time.perf_counter()

        w = self.m.hazard.fusion_weight(self.modality)
        scene = self.m.hazard.fuse(cls["disaster"], self._hz_event, self.modality)
        raw = {"disaster": scene, "event": self._hz_event, "adaptfuse": cls["disaster"],
               "victim": cls["victim"], "nuisance": cls["nuisance"]}
        smooth, confirmed, agreement = self.temporal.update(float(t), raw, boxes)
        # zones only for hazards of the scene class actually reported (no flood tint on a calm lake)
        scene_cls = int(np.argmax(smooth["disaster"]))
        zones = [z for z in self._zones if HAZARD_EVENTS[HAZARD_NAMES.index(z["event"])]["coarse"] == scene_cls]
        rel, unc = cls["reliability"], cls["uncertainty"]
        return {
            "t": round(float(t), 3),
            "modality": self.modality,
            "has": {"rgb": bool(has_rgb), "thermal": bool(has_th), "audio": bool(self._has_audio)},
            "boxes": confirmed,
            "raw_boxes": boxes,
            "zones": zones,
            "zone_grid": self._zone_grid if zones else None,
            "disaster": _top(smooth["disaster"], DISASTER_CLASSES),
            "event": event_label(smooth["disaster"], smooth["event"]),
            "adaptfuse": _top(smooth["adaptfuse"], DISASTER_CLASSES),
            "victim": _top(smooth["victim"], VICTIM_CLASSES),
            "nuisance": _top(smooth["nuisance"], NUISANCE_CLASSES),
            "raw": {"disaster": _top(raw["disaster"], DISASTER_CLASSES),
                    "event": _top(raw["event"], HAZARD_NAMES),
                    "adaptfuse": _top(raw["adaptfuse"], DISASTER_CLASSES),
                    "victim": _top(raw["victim"], VICTIM_CLASSES),
                    "nuisance": _top(raw["nuisance"], NUISANCE_CLASSES)},
            "fusion_weight": w,
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
