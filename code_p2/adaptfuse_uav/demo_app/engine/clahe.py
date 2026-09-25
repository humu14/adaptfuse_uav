"""CLAHE preprocessing identical to adaptfuse_od/data/prepare_dfire_clahe.py (detector input only)."""

from __future__ import annotations

import cv2
import numpy as np

CLIP_LIMIT = 2.0
GRID = 8


def clahe_bgr(img: np.ndarray, clip_limit: float = CLIP_LIMIT, grid: int = GRID) -> np.ndarray:
    """CLAHE on LAB luminance, chroma untouched. A new CLAHE object per call keeps it thread-safe."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
