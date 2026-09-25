import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_classifier import tune_bias  # noqa: E402

from engine.classifier import apply_bias  # noqa: E402


def _precision(p, y, c):
    pred = p.argmax(1)
    return ((pred == c) & (y == c)).sum() / max((pred == c).sum(), 1)


def test_bias_never_buys_f1_with_low_precision():
    # Mirrors the thermal failure: collapse rows are only slightly more "collapse-like" than
    # normal rows, so boosting class 2 raises macro-F1 while flooding normal frames with it.
    rng = np.random.default_rng(0)
    y = np.array([0] * 400 + [1] * 400 + [2] * 100)
    p = np.full((len(y), 4), 0.01)
    p[:400, 0] = 0.6; p[:400, 2] = rng.uniform(0.10, 0.50, 400)      # normal
    p[400:800, 1] = 0.9; p[400:800, 2] = 0.02                        # fire
    p[800:, 0] = 0.6; p[800:, 2] = rng.uniform(0.30, 0.55, 100)      # collapse
    p /= p.sum(1, keepdims=True)
    unconstrained_gain = apply_bias(p, [0, 0, 0.5, 0]).argmax(1)
    assert (unconstrained_gain == 2).sum() > 0                        # boosting does flood class 2
    b = tune_bias(p, y)
    q = apply_bias(p, b)
    for c in range(4):
        if (q.argmax(1) == c).any():
            assert _precision(q, y, c) >= 0.5, (c, b)


def test_bias_still_helps_when_precision_allows():
    # class 3 is under-predicted but separable: a positive bias should be found.
    y = np.array([0] * 300 + [3] * 100)
    p = np.zeros((400, 4)) + 0.01
    p[:300, 0] = 0.7; p[:300, 3] = 0.2
    p[300:, 0] = 0.5; p[300:, 3] = 0.4
    p /= p.sum(1, keepdims=True)
    assert tune_bias(p, y)[3] > 0


def test_exempt_class_cannot_be_flooded():
    # One stray base prediction of class 2 (precision 0) must not exempt it from the floor.
    rng = np.random.default_rng(0)
    y = np.array([0] * 400 + [1] * 400 + [2] * 100)
    p = np.full((len(y), 4), 0.01)
    p[:400, 0] = 0.6; p[:400, 2] = rng.uniform(0.10, 0.50, 400)
    p[400:800, 1] = 0.9; p[400:800, 2] = 0.02
    p[800:, 0] = 0.6; p[800:, 2] = rng.uniform(0.30, 0.55, 100)
    p[0, 2] = 0.95                                                   # one normal row -> collapse
    p /= p.sum(1, keepdims=True)
    base = (p.argmax(1) == 2).sum()
    q = apply_bias(p, tune_bias(p, y))
    n2 = (q.argmax(1) == 2).sum()
    assert n2 <= base or _precision(q, y, 2) >= 0.5, (n2, _precision(q, y, 2))
