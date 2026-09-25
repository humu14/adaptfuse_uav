"""Guess whether a single-stream video is visible RGB or thermal infrared."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

THERMAL_NAME_HINTS = ("thermal", "therm", "_ir", "-ir", "lwir", "flir", "infrared")


def _channel_spread(frame_bgr: np.ndarray) -> float:
    """Mean absolute difference between colour channels (0 for greyscale)."""
    f = frame_bgr.astype(np.int16)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    return float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)


def _palette_score(frame_bgr: np.ndarray) -> float:
    """
    Fraction of pixels lying on a typical false-colour thermal palette
    (ironbow / jet): saturated pixels whose hue is monotonic with brightness.
    Measured as |corr(hue, value)| over saturated pixels.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].ravel()
    mask = s > 120
    if mask.mean() < 0.5:
        return 0.0
    h = hsv[..., 0].ravel()[mask].astype(np.float32)
    v = hsv[..., 2].ravel()[mask].astype(np.float32)
    if h.std() < 1e-3 or v.std() < 1e-3:
        return 0.0
    return float(abs(np.corrcoef(h, v)[0, 1]))


def detect_modality(path: str | Path, samples: int = 8) -> dict:
    """Return {'modality': 'rgb'|'thermal', 'reason': str, 'spread': float}."""
    path = Path(path)
    if any(k in path.stem.lower() for k in THERMAL_NAME_HINTS):
        return {"modality": "thermal", "reason": "file name", "spread": None}

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(n - 1, 0), samples).astype(int) if n else range(samples)
    spreads, palettes = [], []
    for i in idxs:
        if n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (160, 120))
        spreads.append(_channel_spread(small))
        palettes.append(_palette_score(small))
    cap.release()
    if not spreads:
        return {"modality": "rgb", "reason": "unreadable, default", "spread": None}

    spread = float(np.median(spreads))
    palette = float(np.median(palettes))
    if spread < 4.0:
        return {"modality": "thermal", "reason": "greyscale frames", "spread": spread}
    if palette > 0.75:
        return {"modality": "thermal", "reason": "false-colour palette", "spread": spread}
    return {"modality": "rgb", "reason": "colour frames", "spread": spread}
