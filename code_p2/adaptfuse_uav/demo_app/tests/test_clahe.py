from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.clahe import clahe_bgr

DFIRE = Path(__file__).resolve().parents[4] / "adaptfuse_od" / "data" / "datasets"


def _reference(img):
    # core of adaptfuse_od/data/prepare_dfire_clahe.py::enhance_image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((c.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def test_matches_reference_on_synthetic():
    img = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    assert np.array_equal(clahe_bgr(img), _reference(img))


def test_keeps_shape_and_dtype():
    img = np.full((37, 53, 3), 90, np.uint8)
    out = clahe_bgr(img)
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.skipif(not (DFIRE / "dfire_clahe" / "images" / "test").exists(),
                    reason="D-Fire datasets not present")
def test_matches_stored_dfire_clahe():
    raws = sorted((DFIRE / "dfire" / "images" / "test").glob("*.jpg"))[:20]
    diffs = []
    for p in raws:
        ref = cv2.imread(str(DFIRE / "dfire_clahe" / "images" / "test" / p.name))
        ours = clahe_bgr(cv2.imread(str(p)))
        diffs.append(np.abs(ours.astype(int) - ref.astype(int)).mean())
    assert len(diffs) == 20 and max(diffs) < 2.0, diffs
