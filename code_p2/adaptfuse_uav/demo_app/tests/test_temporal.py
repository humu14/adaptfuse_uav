import math

import numpy as np

from engine.temporal import TemporalState, ema_alpha, iou

NORMAL = {"disaster": np.array([0.9, 0.05, 0.03, 0.02]), "victim": np.array([0.9, 0.1]),
          "nuisance": np.array([1.0, 0, 0, 0, 0])}
FIRE = {"disaster": np.array([0.05, 0.9, 0.03, 0.02]), "victim": np.array([0.9, 0.1]),
        "nuisance": np.array([1.0, 0, 0, 0, 0])}


def box(cls="smoke", conf=0.7, xyxy=(0.1, 0.1, 0.5, 0.5)):
    return {"cls": cls, "conf": conf, "source": "dfire", "box": list(xyxy)}


def test_iou_basic():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]) == 0.0
    assert abs(iou([0, 0, 1, 1], [0, 0, 0.5, 1]) - 0.5) < 1e-9


def test_first_update_equals_raw_then_ema():
    s = TemporalState()
    p, _, _ = s.update(0.0, NORMAL, [])
    np.testing.assert_allclose(p["disaster"], NORMAL["disaster"])
    p, _, _ = s.update(0.5, FIRE, [])
    a = 1 - math.exp(-1)
    np.testing.assert_allclose(p["disaster"], NORMAL["disaster"] + a * (FIRE["disaster"] - NORMAL["disaster"]), atol=1e-6)


def test_large_dt_no_reset():
    s = TemporalState()
    s.update(0.0, NORMAL, [])
    p, _, _ = s.update(1.0, FIRE, [])          # stride of 10 frames at 10 fps
    expected = NORMAL["disaster"] + ema_alpha(1.0, 0.5) * (FIRE["disaster"] - NORMAL["disaster"])
    np.testing.assert_allclose(p["disaster"], expected, atol=1e-6)


def test_backward_jump_resets():
    s = TemporalState()
    s.update(5.0, NORMAL, [box()])
    s.update(5.1, NORMAL, [box()])
    p, confirmed, _ = s.update(4.6, FIRE, [box()])     # user seeks back 0.5 s
    np.testing.assert_allclose(p["disaster"], FIRE["disaster"])
    assert confirmed == []                             # history cleared: 1 of 1 frames


def test_forward_gap_resets():
    s = TemporalState()
    s.update(0.0, NORMAL, [])
    p, _, _ = s.update(2.5, FIRE, [])
    np.testing.assert_allclose(p["disaster"], FIRE["disaster"])


def test_box_needs_two_of_three():
    s = TemporalState()
    assert s.update(0.0, NORMAL, [box(conf=0.6)])[1] == []
    c = s.update(0.1, NORMAL, [box(conf=0.8)])[1]
    assert len(c) == 1 and c[0]["hits"] == 2 and abs(c[0]["conf"] - 0.7) < 1e-9


def test_box_class_and_iou_must_match():
    s = TemporalState()
    s.update(0.0, NORMAL, [box(cls="fire")])
    assert s.update(0.1, NORMAL, [box(cls="smoke")])[1] == []          # class differs
    s2 = TemporalState()
    s2.update(0.0, NORMAL, [box(xyxy=(0.0, 0.0, 0.2, 0.2))])
    assert s2.update(0.1, NORMAL, [box(xyxy=(0.6, 0.6, 0.9, 0.9))])[1] == []   # IoU 0


def test_box_expires_after_window():
    s = TemporalState()
    s.update(0.0, NORMAL, [box()])
    s.update(0.1, NORMAL, [])
    s.update(0.2, NORMAL, [])
    assert s.update(0.3, NORMAL, [box()])[1] == []      # the t=0 hit has left the 3-frame window


def test_detector_only_flag():
    s = TemporalState(scene_names=["normal", "fire / smoke", "collapse / flood", "other disaster"])
    s.update(0.0, NORMAL, [box(conf=0.7)])
    _, _, ag = s.update(0.1, NORMAL, [box(conf=0.7)])
    assert ag["status"] == "detector_only" and "smoke 0.70" in ag["reason"] and "normal" in ag["reason"]
    assert ag["since"] == 0.1


def test_weak_box_does_not_flag():
    s = TemporalState()
    s.update(0.0, NORMAL, [box(conf=0.3)])
    assert s.update(0.1, NORMAL, [box(conf=0.3)])[2]["status"] == "agree"


def test_classifier_only_after_two_seconds():
    s = TemporalState()
    statuses = {t: s.update(t, FIRE, [])[2]["status"] for t in (0.0, 0.5, 1.0, 1.5, 2.0)}
    assert statuses[1.5] == "agree" and statuses[2.0] == "classifier_only"


def test_box_restarts_classifier_only_clock():
    s = TemporalState()
    for t in (0.0, 0.5, 1.0):
        s.update(t, FIRE, [])
    s.update(1.5, FIRE, [box(cls="fire", conf=0.4)])
    s.update(1.6, FIRE, [box(cls="fire", conf=0.4)])   # confirmed fire box at 1.6
    assert s.update(3.0, FIRE, [])[2]["status"] == "agree"            # 1.4 s since box
    assert s.update(3.6, FIRE, [])[2]["status"] == "classifier_only"  # 2.0 s since box
