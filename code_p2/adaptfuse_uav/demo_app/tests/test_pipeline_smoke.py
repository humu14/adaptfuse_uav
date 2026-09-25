import numpy as np
import pytest

from engine import Analyzer, load_models
from engine.media import AudioTrack


@pytest.fixture(scope="module")
def bundle():
    return load_models("cpu")


def _frame(seed=0):
    return (np.random.default_rng(seed).random((360, 640, 3)) * 255).astype(np.uint8)


def _tone():
    t16, t32 = np.arange(0, 5, 1 / 16000), np.arange(0, 5, 1 / 32000)
    return AudioTrack((0.1 * np.sin(2 * np.pi * 440 * t16)).astype(np.float32),
                      (0.1 * np.sin(2 * np.pi * 440 * t32)).astype(np.float32))


def test_result_keys_and_probabilities(bundle):
    r = Analyzer(bundle, _tone(), "rgb").analyze(_frame(), 1.0)
    for k in ("t", "modality", "has", "boxes", "raw_boxes", "disaster", "victim", "nuisance",
              "raw", "reliability", "uncertainty", "agreement", "audio", "timing_ms"):
        assert k in r, k
    for head in ("disaster", "victim", "nuisance"):
        assert abs(sum(r[head]["probs"].values()) - 1) < 1e-4
        assert abs(sum(r["raw"][head]["probs"].values()) - 1) < 1e-4
    assert r["agreement"]["status"] in ("agree", "detector_only", "classifier_only")


def test_no_audio_video(bundle):
    r = Analyzer(bundle, None, "thermal").analyze(_frame(), 0.0)
    assert r["has"] == {"rgb": False, "thermal": True, "audio": False}
    assert r["reliability"]["audio"] == 0.0 and r["reliability"]["rgb"] == 0.0


def test_set_modality_resets_temporal(bundle):
    a = Analyzer(bundle, None, "rgb")
    a.analyze(_frame(1), 0.0)
    assert a.temporal.last_t == 0.0
    a.set_modality("rgb")                   # unchanged: keep state
    assert a.temporal.last_t == 0.0
    a.set_modality("thermal")
    assert a.temporal.last_t is None


def test_reset_gap_is_configurable(bundle):
    a = Analyzer(bundle, None, "rgb", reset_gap=5.0)
    a.analyze(_frame(2), 0.0)
    a.analyze(_frame(2), 2.5)                 # 2.5 s step: must not reset with a 5 s gap
    assert a.temporal.last_t == 2.5 and len(a.temporal.history) == 2


def test_gradio_reset_gap_scales_with_step():
    import gradio_app
    assert gradio_app.temporal_reset_gap(1, 30.0) == 2.0            # normal playback: spec value
    assert gradio_app.temporal_reset_gap(10, 4.0) > 2.5             # 2.5 s step must not reset
    assert gradio_app.temporal_reset_gap(60, 29.97) > 60 / 29.97
