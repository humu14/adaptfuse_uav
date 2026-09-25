import json

import numpy as np
import torch
import torch.nn as nn

from engine.classifier import SceneClassifier, apply_bias, load_calibration


class FakeMember(nn.Module):
    """Stands in for AdapFuseV1: fixed disaster probabilities, records what it was fed."""

    def __init__(self, probs):
        super().__init__()
        self.logits = torch.log(torch.tensor(probs))
        self.batch, self.flags, self._o = None, None, {}

    def forward(self, rgb, th, au, has_rgb, has_th, has_au):
        B = rgb.shape[0]
        self.batch, self.flags = B, torch.stack([has_rgb, has_th, has_au], -1)
        self._o = {"nuisance": torch.zeros(B, 5), "uncertainty": torch.zeros(B, 3)}
        return self.logits.repeat(B, 1), torch.zeros(B, 2), self.flags.clone()

    def get_full_outputs(self):
        return self._o


def _inputs():
    return torch.zeros(1, 3, 224, 224), torch.zeros(1, 1, 224, 224), torch.zeros(1, 1, 64, 63)


def test_ensemble_averages_probabilities():
    a, b = FakeMember([0.7, 0.1, 0.1, 0.1]), FakeMember([0.1, 0.7, 0.1, 0.1])
    out = SceneClassifier([a, b], "cpu", tta=False).predict(*_inputs(), (1.0, 0.0, 1.0), "rgb")
    np.testing.assert_allclose(out["disaster"], [0.4, 0.4, 0.1, 0.1], atol=1e-5)
    assert abs(out["victim"].sum() - 1) < 1e-5 and abs(out["nuisance"].sum() - 1) < 1e-5


def test_tta_doubles_batch():
    m = FakeMember([0.25] * 4)
    SceneClassifier([m], "cpu", tta=True).predict(*_inputs(), (1.0, 0.0, 1.0), "rgb")
    assert m.batch == 2
    SceneClassifier([m], "cpu", tta=False).predict(*_inputs(), (1.0, 0.0, 1.0), "rgb")
    assert m.batch == 1


def test_bias_selected_by_condition():
    clf = SceneClassifier([FakeMember([0.4, 0.3, 0.2, 0.1])], "cpu", tta=False,
                          bias={"rgb": [0, 0, 0, 0], "thermal": [0, 0, 5, 0]})
    assert int(clf.predict(*_inputs(), (1.0, 0.0, 1.0), "rgb")["disaster"].argmax()) == 0
    assert int(clf.predict(*_inputs(), (0.0, 1.0, 1.0), "thermal")["disaster"].argmax()) == 2


def test_flags_pass_through_no_audio():
    m = FakeMember([0.25] * 4)
    out = SceneClassifier([m], "cpu", tta=True).predict(*_inputs(), (1.0, 0.0, 0.0), "rgb")
    assert m.flags.tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert out["reliability"][2] == 0.0


def test_apply_bias_identity_and_normalised():
    p = np.array([0.5, 0.3, 0.15, 0.05], np.float32)
    np.testing.assert_allclose(apply_bias(p, None), p)
    np.testing.assert_allclose(apply_bias(p, [0, 0, 0, 0]), p, atol=1e-6)
    q = apply_bias(p, [0, 0, 1.5, 0])
    assert abs(q.sum() - 1) < 1e-6 and q[2] > p[2]


def test_load_calibration_missing_file(tmp_path):
    cal = load_calibration(tmp_path / "calibration.json")
    assert cal["tta"] is True and cal["bias"] == {} and cal["members"] is None


def test_load_calibration_reads_file(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"stage": "ensemble", "tta": False, "members": ["a.pth"],
                             "bias": {"rgb": [0, 0.3, 0.8, 0.1]}}))
    cal = load_calibration(p)
    assert cal["tta"] is False and cal["members"] == ["a.pth"] and cal["bias"]["rgb"][2] == 0.8


class NanMember(FakeMember):
    """Returns NaN logits for every row (seen once on the GTX 960, non-deterministically)."""

    def forward(self, rgb, th, au, has_rgb, has_th, has_au):
        d, v, rel = super().forward(rgb, th, au, has_rgb, has_th, has_au)
        return torch.full_like(d, float("nan")), v, rel


def test_nan_member_is_ignored():
    good, bad = FakeMember([0.7, 0.1, 0.1, 0.1]), NanMember([0.25] * 4)
    out = SceneClassifier([good, bad], "cpu", tta=True).predict(*_inputs(), (1.0, 0.0, 1.0), "rgb")
    np.testing.assert_allclose(out["disaster"], [0.7, 0.1, 0.1, 0.1], atol=1e-5)


def test_all_nan_falls_back_to_uniform():
    out = SceneClassifier([NanMember([0.25] * 4)], "cpu", tta=False).predict(*_inputs(), (1.0, 0.0, 1.0), "rgb")
    assert not np.isnan(out["disaster"]).any()
    np.testing.assert_allclose(out["disaster"], [0.25] * 4, atol=1e-6)
