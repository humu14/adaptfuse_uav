# Demo App Accuracy Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AdapFuse-UAV demo app correct and more accurate. Replace the swapped-label detector with the user's best CLAHE YOLO26s. Replace the row-level classifier with a measured ensemble of the grouped seeds 41 + 42, with flip TTA and class-bias calibration. Add temporal smoothing and a "models disagree" warning.

**Architecture:** All changes sit behind the existing `engine/` boundary, which the live FastAPI app and the Gradio app share. The new pieces are three focused modules:
* `engine/clahe.py`: detector preprocessing;
* `engine/classifier.py`: `SceneClassifier`, an ensemble with TTA and bias;
* `engine/temporal.py`: `TemporalState`, which handles smoothing, box persistence and agreement.

`engine/pipeline.py:Analyzer` wires them together. Two offline scripts do the rest:
* `scripts/check_detector.py` verifies the detector;
* `scripts/eval_classifier.py` measures each classifier stage on the grouped split and writes `weights/calibration.json`, which the engine reads.

**Tech Stack:** Python 3.12, PyTorch 2.2.2, ultralytics 8.4 (YOLO26), OpenCV, librosa, scikit-learn, FastAPI/uvicorn, Gradio 5, pytest 7.

**Spec:** `docs/superpowers/specs/2026-09-25-demo-accuracy-design.md`

## Global Constraints

- Working directory for all commands: `code_p2/adaptfuse_uav/demo_app/` unless a step says otherwise. Repo root is `Co-Sup/`.
- Detector: `adaptfuse_od/outputs/accuracy_study/accuracy_yolo26s_clahe_640/weights/best.pt`, bundled as `weights/dfire_yolo26s_clahe.pt`, inference `imgsz=640`, default conf 0.25.
- Detector class names must equal `{0: "smoke", 1: "fire"}`. Person detector `names[0] == "person"`. Labels always come from `model.names`.
- CLAHE: BGR→LAB, CLAHE on L with `clipLimit=2.0`, `tileGridSize=(8, 8)`, merge, LAB→BGR. Only the detector input is enhanced.
- Classifier members: `outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth` and `seed_42/...`, bundled as fp16 `weights/adaptfuse_v1_grouped_s41.pth` / `_s42.pth`.
- Bias tuned on `data/metadata_grouped/seed_42/val.csv` only. Coordinate search over `[-2, 2]`, step 0.1, per condition `rgb` / `thermal`. The test split is used only for reporting and the adoption rule.
- Smoothing: time EMA with τ = 0.5 s. Box confirmed if the same class with IoU ≥ 0.3 appears in ≥ 2 of the last 3 analyzed frames. Reset when `t` goes backwards or jumps forward by more than 2 s.
- Disagreement: `detector_only` = a confirmed fire/smoke box with conf ≥ 0.5 while the smoothed scene ≠ `fire / smoke`. `classifier_only` = scene `fire / smoke` for ≥ 2 s with no confirmed fire/smoke box. Nothing is overridden.
- Live app ≥ 8 analyses/s on the GTX 960. If the target is missed, drop TTA first and record it in the report.
- fp16 checkpoints: identical argmax on ≥ 99.9 % of grouped-test rows.
- Detector check: test mAP@50 = 0.771 ± 0.005. CLAHE parity: mean abs difference < 2 grey levels on 20 D-Fire test images.
- No retraining. Person detector (COCO YOLO26n) and PANNs tags are unchanged.
- Provenance wording: pipeline-trained models are described as "trained in the AdapFuse-UAV pipeline"; the COCO and AudioSet models as "pretrained, not fine-tuned".
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Gradio "analyze every N-th frame" > 1.** The time step between analyses is large (e.g. 1 s). Smoothing must still converge without resetting, and persistence must still work. *Test in Task 5 (`test_large_dt_no_reset`).*
- **User switches modality RGB↔thermal mid-video.** Smoothed probabilities and box history from the old condition must not bleed into the new one. *Test in Task 6 (`test_set_modality_resets_temporal`).*
- **Video with no audio track** (`has_audio = 0`). The ensemble must pass the zero flag to every member and view, and audio reliability must be 0. *Tests in Task 2 (`test_flags_pass_through_no_audio`) and Task 6 (`test_no_audio_video`).*
- **Fresh clone with a missing weight or a missing `calibration.json`.** One missing seed should warn and run with the other; a missing calibration should warn and use zero bias. Neither should crash. *Tests in Task 2 (`test_load_calibration_missing_file`) and Task 3 (`test_resolve_members_one_missing`, `test_resolve_members_none_raises`).*
- **Small backward seek in the live player** (e.g. back 0.5 s). Old boxes and probabilities must be dropped, not confirmed against a different moment. *Test in Task 5 (`test_backward_jump_resets`).*

---

### Task 1: Detector swap — CLAHE preprocessing, name guard, YOLO26s bundle

**Files:**
- Create: `engine/clahe.py`
- Create: `tests/conftest.py`, `tests/test_clahe.py`, `tests/test_models_guard.py`
- Create: `scripts/check_detector.py`
- Modify: `engine/models.py` (constants, `check_detector_names`, detector loading)
- Modify: `engine/pipeline.py` (`Analyzer._detect`)
- Modify: `engine/labels.py` (remove `DET_FIRE_CLASSES`)
- Modify: `scripts/export_weights.py` (`COPIES`)
- Delete: `weights/dfire_yolo26n.pt`

**Interfaces:**
- Produces: `engine.clahe.clahe_bgr(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray` (same shape, uint8 BGR)
- Produces: `engine.models.check_detector_names(names: dict, expected: dict, path, exact: bool) -> None` (raises `ValueError`)
- Produces: `engine.models.DETECTOR_FILE = "dfire_yolo26s_clahe.pt"`, `DETECTOR_NAMES = {0: "smoke", 1: "fire"}`
- Produces: boxes from `Analyzer._detect` carry `cls` taken from `result.names`

- [ ] **Step 1: Write test scaffolding and failing tests**

`tests/conftest.py`:
```python
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda OpenMP clash
DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[4]          # Co-Sup/
sys.path.insert(0, str(DEMO_DIR))
```

`tests/test_clahe.py`:
```python
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
```

`tests/test_models_guard.py`:
```python
import pytest

from engine.models import DETECTOR_NAMES, check_detector_names


def test_rejects_swapped_names():
    with pytest.raises(ValueError, match="swapped"):
        check_detector_names({0: "fire", 1: "smoke"}, DETECTOR_NAMES, "old.pt", exact=True)


def test_accepts_correct_names():
    check_detector_names({0: "smoke", 1: "fire"}, DETECTOR_NAMES, "best.pt", exact=True)


def test_exact_rejects_extra_classes():
    with pytest.raises(ValueError):
        check_detector_names({0: "smoke", 1: "fire", 2: "x"}, DETECTOR_NAMES, "b.pt", exact=True)


def test_person_subset_ok():
    coco = {i: f"c{i}" for i in range(80)}
    coco[0] = "person"
    check_detector_names(coco, {0: "person"}, "coco.pt", exact=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clahe.py tests/test_models_guard.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.clahe'` and `ImportError: cannot import name 'DETECTOR_NAMES'`.

- [ ] **Step 3: Implement `engine/clahe.py`**

```python
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
```

- [ ] **Step 4: Add constants and guard to `engine/models.py`**

Below `WEIGHTS_DIR = DEMO_DIR / "weights"` add:
```python
DETECTOR_FILE = "dfire_yolo26s_clahe.pt"          # YOLO26s fine-tuned on CLAHE D-Fire (mAP@50 0.771)
DETECTOR_NAMES = {0: "smoke", 1: "fire"}           # official D-Fire order
PERSON_FILE = "yolo26n_coco.pt"
DETECTOR_IMGSZ = 640


def check_detector_names(names: dict, expected: dict, path, exact: bool) -> None:
    """Refuse detectors whose label map differs from the dataset's (e.g. swapped fire/smoke)."""
    got = {int(k): str(v) for k, v in dict(names).items()}
    bad = any(got.get(k) != v for k, v in expected.items()) or (exact and len(got) != len(expected))
    if bad:
        raise ValueError(
            f"{Path(path).name}: class names {got} do not match expected {expected}. "
            "Refusing to load a detector whose labels may be swapped."
        )
```

In `load_models`, replace the two YOLO lines:
```python
    fire_det = YOLO(str(_require(WEIGHTS_DIR / "dfire_yolo26n.pt")))
    person_det = YOLO(str(_require(WEIGHTS_DIR / "yolo26n_coco.pt")))
```
with:
```python
    fire_path = _require(WEIGHTS_DIR / DETECTOR_FILE)
    fire_det = YOLO(str(fire_path))
    check_detector_names(fire_det.names, DETECTOR_NAMES, fire_path, exact=True)
    person_path = _require(WEIGHTS_DIR / PERSON_FILE)
    person_det = YOLO(str(person_path))
    check_detector_names(person_det.names, {0: "person"}, person_path, exact=False)
```
Update the module docstring table row for fire and smoke boxes to read: `YOLO26s fine-tuned on CLAHE D-Fire | weights/dfire_yolo26s_clahe.pt`.

- [ ] **Step 5: Use CLAHE and model names in `engine/pipeline.py`**

Add `from .clahe import clahe_bgr` and `from .models import DETECTOR_IMGSZ, ModelBundle` (the latter replaces `from .models import ModelBundle`). In the `from .labels import (...)` block, remove `DET_FIRE_CLASSES` and `PERSON_CLASS`. Replace `Analyzer._detect` with:
```python
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
```
In `engine/labels.py`, delete the lines `DET_FIRE_CLASSES = ["fire", "smoke"] ...` and `PERSON_CLASS = "person" ...`. Then run `grep -rn "DET_FIRE_CLASSES\|PERSON_CLASS" --include=*.py .` and expect no matches; remove any remaining import.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_clahe.py tests/test_models_guard.py -q`
Expected: 7 passed (on this machine the D-Fire parity test runs; elsewhere it skips).

- [ ] **Step 7: Bundle the detector and write `scripts/check_detector.py`**

In `scripts/export_weights.py`, replace the `COPIES` dict entry `"dfire_yolo26n.pt": ...` with:
```python
    "dfire_yolo26s_clahe.pt": REPO / "adaptfuse_od/outputs/accuracy_study/accuracy_yolo26s_clahe_640/weights/best.pt",
```
Copy it and remove the old file:
```bash
python -c "import shutil; shutil.copy2('../../../adaptfuse_od/outputs/accuracy_study/accuracy_yolo26s_clahe_640/weights/best.pt', 'weights/dfire_yolo26s_clahe.pt')"
git rm -q weights/dfire_yolo26n.pt
```

`scripts/check_detector.py`:
```python
"""
Verify the bundled D-Fire detector: class names, CLAHE parity with the training data,
and test-split mAP@50 against the recorded 0.771.

Run from code_p2/adaptfuse_uav/demo_app:
    python scripts/check_detector.py            # full check (needs adaptfuse_od datasets)
    python scripts/check_detector.py --skip-val # names + CLAHE only
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from engine.clahe import clahe_bgr  # noqa: E402
from engine.models import DETECTOR_FILE, DETECTOR_IMGSZ, DETECTOR_NAMES, WEIGHTS_DIR, check_detector_names  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DS = REPO / "adaptfuse_od" / "data" / "datasets"
DATA_YAML = REPO / "adaptfuse_od/outputs/accuracy_study/accuracy_yolo26s_clahe_640/dataset.resolved.yaml"
RECORDED_MAP50 = 0.771


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args()

    path = WEIGHTS_DIR / DETECTOR_FILE
    model = YOLO(str(path))
    check_detector_names(model.names, DETECTOR_NAMES, path, exact=True)
    print(f"[ok] names {model.names}")

    raws = sorted((DS / "dfire" / "images" / "test").glob("*.jpg"))[:20]
    diffs = [np.abs(clahe_bgr(cv2.imread(str(p))).astype(int)
                    - cv2.imread(str(DS / "dfire_clahe" / "images" / "test" / p.name)).astype(int)).mean()
             for p in raws]
    print(f"[{'ok' if diffs and max(diffs) < 2.0 else 'FAIL'}] CLAHE parity on {len(diffs)} images: "
          f"max mean-abs-diff {max(diffs):.3f}")
    if not diffs or max(diffs) >= 2.0:
        return 1

    if args.skip_val:
        return 0
    device = 0 if torch.cuda.is_available() else "cpu"
    m = model.val(data=str(DATA_YAML), split="test", imgsz=DETECTOR_IMGSZ, batch=8, device=device,
                  plots=False, verbose=False, project=tempfile.mkdtemp(), name="val")
    map50 = float(m.box.map50)
    ok = abs(map50 - RECORDED_MAP50) <= 0.005
    print(f"[{'ok' if ok else 'FAIL'}] test mAP@50 {map50:.4f} (recorded {RECORDED_MAP50})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run the detector check**

Run: `python scripts/check_detector.py`
Expected: `[ok] names {0: 'smoke', 1: 'fire'}`, `[ok] CLAHE parity ... max mean-abs-diff <2`, `[ok] test mAP@50 0.77xx`. It takes about 3–6 minutes on the GTX 960. If mAP is off by more than 0.005, stop and report. Do not continue.

- [ ] **Step 9: Commit**

```bash
git add engine/clahe.py engine/models.py engine/pipeline.py engine/labels.py scripts/export_weights.py scripts/check_detector.py tests/ weights/dfire_yolo26s_clahe.pt
git commit -m "Use CLAHE YOLO26s detector with class-name guard

The bundled YOLO26n had fire/smoke names swapped, so smoke was drawn as fire.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `SceneClassifier` — ensemble, flip TTA, class bias

**Files:**
- Create: `engine/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Produces: `member_outputs(model, rgb, th, au, has) -> dict[str, torch.Tensor]`. Inputs are batched tensors on the model's device; `has` is `(B, 3)` float `[rgb, thermal, audio]`. It returns CPU float32 `(B, K)` tensors for keys `disaster` (softmax), `victim` (softmax), `nuisance` (softmax), `reliability`, `uncertainty`.
- Produces: `apply_bias(p: np.ndarray, bias) -> np.ndarray`, computing `softmax(log p + b)` over the last axis. `bias=None` is the identity.
- Produces: `load_calibration(path: Path) -> dict` with keys `stage`, `members`, `tta`, `bias`. Defaults when the file is missing: `members=None`, `tta=True`, `bias={}`.
- Produces: `SceneClassifier(members: list[nn.Module], device, tta: bool = True, bias: dict | None = None)`. `.predict(rgb, th, au, has: tuple[float, float, float], condition: str) -> dict[str, np.ndarray]` takes single-frame CPU tensors `(1, …)` and returns 1-D numpy vectors per key. Bias applies to `disaster` only, using `bias.get(condition)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_classifier.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_classifier.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.classifier'`.

- [ ] **Step 3: Implement `engine/classifier.py`**

```python
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
        res = {k: torch.stack([o[k] for o in outs]).mean(dim=(0, 1)).numpy() for k in outs[0]}
        res["disaster"] = apply_bias(res["disaster"], self.bias.get(condition))
        return res
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_classifier.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/classifier.py tests/test_classifier.py
git commit -m "Add SceneClassifier: member ensemble, flip TTA, per-condition class bias

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Bundle the grouped fp16 members and load the ensemble in the engine

**Files:**
- Modify: `scripts/export_weights.py` (`CLASSIFIERS`, fp16 conversion)
- Modify: `engine/models.py` (`ModelBundle`, `resolve_members`, `load_models`)
- Modify: `engine/pipeline.py` (`_top`, classification block of `Analyzer.analyze`)
- Modify: `live/server.py` and `gradio_app.py` only where they reference `bundle.fusion` (check with grep)
- Delete: `weights/adaptfuse_v1.pth`
- Test: `tests/test_models_guard.py` (add `resolve_members` tests)

**Interfaces:**
- Consumes: `SceneClassifier`, `load_calibration` (Task 2).
- Produces: `engine.models.CLASSIFIER_FILES = ("adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth")`, `CALIBRATION_FILE = "calibration.json"`.
- Produces: `engine.models.resolve_members(files, weights_dir: Path) -> list[Path]`. It warns about each missing file and raises `FileNotFoundError` if none exist.
- Produces: `engine.models.load_member(path: Path, device) -> nn.Module`, an eval-mode fp32 AdapFuseV1.
- Produces: `ModelBundle.classifier: SceneClassifier`. The field `ModelBundle.fusion` is removed.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_models_guard.py`)

```python
from engine.models import resolve_members


def test_resolve_members_one_missing(tmp_path, capsys):
    (tmp_path / "a.pth").write_bytes(b"x")
    got = resolve_members(("a.pth", "b.pth"), tmp_path)
    assert got == [tmp_path / "a.pth"]
    assert "b.pth" in capsys.readouterr().out


def test_resolve_members_none_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        resolve_members(("a.pth", "b.pth"), tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models_guard.py -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_members'`.

- [ ] **Step 3: Export fp16 members**

In `scripts/export_weights.py`, replace `CLASSIFIERS` with:
```python
CLASSIFIERS = {
    "adaptfuse_v1_grouped_s41.pth": ROOT / "outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth",
    "adaptfuse_v1_grouped_s42.pth": ROOT / "outputs/grouped_repeated/seed_42/checkpoints/adaptfuse_v1_grouped_best.pth",
    "baseline_audio.pth": ROOT / "outputs/checkpoints/baseline_audio_best.pth",
}
FP16 = {"adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth"}   # halves repo size
```
In the loop, replace `"model_state": ck["model_state"],` with:
```python
            "model_state": ({k: (v.half() if v.is_floating_point() else v) for k, v in ck["model_state"].items()}
                            if name in FP16 else ck["model_state"]),
```
Run: `python scripts/export_weights.py` (paths resolve from the script location)
Expected: `[ok] adaptfuse_v1_grouped_s41.pth: 26.x MB` and `_s42.pth: 26.x MB`, plus the other files.
Then: `git rm -q weights/adaptfuse_v1.pth`

- [ ] **Step 4: Load the ensemble in `engine/models.py`**

Add imports near the top: `from .classifier import SceneClassifier, load_calibration`. Add constants below `DETECTOR_IMGSZ`:
```python
CLASSIFIER_FILES = ("adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth")
CALIBRATION_FILE = "calibration.json"
```
Add functions:
```python
def resolve_members(files, weights_dir: Path) -> list:
    found = []
    for name in files:
        p = Path(weights_dir) / name
        if p.exists():
            found.append(p)
        else:
            print(f"[models] WARNING: classifier member {name} missing - ensemble runs without it")
    if not found:
        raise FileNotFoundError(f"No classifier member found in {weights_dir} (expected {list(files)})")
    return found


def load_member(path: Path, device) -> nn.Module:
    from models.full_model import AdapFuseV1
    m = AdapFuseV1(num_disaster_classes=4, num_victim_classes=2, pretrained=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["model_state"].items()})
    return m.to(device).eval()
```
In `ModelBundle`, replace `fusion: nn.Module` with `classifier: object          # SceneClassifier`.
In `load_models`, delete the `fusion = AdapFuseV1(...)` and `_load_state(fusion, ...)` lines and the `from models.full_model import AdapFuseV1` import. Add after `dev = ...`:
```python
    cal = load_calibration(WEIGHTS_DIR / CALIBRATION_FILE)
    member_paths = resolve_members(cal["members"] or CLASSIFIER_FILES, WEIGHTS_DIR)
    classifier = SceneClassifier([load_member(p, dev) for p in member_paths], dev,
                                 tta=bool(cal["tta"]), bias=cal["bias"])
    print(f"[models] scene classifier: {[p.name for p in member_paths]}, tta={cal['tta']}, "
          f"bias={'yes' if cal['bias'] else 'no'} (stage {cal['stage']})")
```
Change `for m in (fusion, audio_only, tagger):` to `for m in (audio_only, tagger):`. Change the return to `ModelBundle(dev, classifier, audio_only, fire_det, person_det, tagger, names)`. Update the docstring table row for scene labels to `AdapFuseV1 ensemble (grouped seeds 41+42) | weights/adaptfuse_v1_grouped_s4{1,2}.pth`.

- [ ] **Step 5: Use the classifier in `engine/pipeline.py`**

Replace `_top` with a version that takes probabilities:
```python
def _top(probs: np.ndarray, names: list) -> dict:
    probs = np.asarray(probs, dtype=np.float32)
    i = int(probs.argmax())
    return {"label": names[i], "index": i, "conf": float(probs[i]),
            "probs": {n: float(p) for n, p in zip(names, probs)}}
```
(The audio-only calls `_top(_softmax(d), ...)` remain valid.) In `Analyzer.analyze`, replace everything from `flags = [torch.tensor(...` through the `return {...}` with:
```python
        with self.m.lock:
            cls = self.m.classifier.predict(rgb, th, self._audio_tensor,
                                            (has_rgb, has_th, self._has_audio), self.modality)
        t_cls = time.perf_counter()

        rel, unc = cls["reliability"], cls["uncertainty"]
        return {
            "t": round(float(t), 3),
            "modality": self.modality,
            "has": {"rgb": bool(has_rgb), "thermal": bool(has_th), "audio": bool(self._has_audio)},
            "boxes": boxes,
            "disaster": _top(cls["disaster"], DISASTER_CLASSES),
            "victim": _top(cls["victim"], VICTIM_CLASSES),
            "nuisance": _top(cls["nuisance"], NUISANCE_CLASSES),
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
```
Delete the now-unused `dev = self.m.device` line before the modality branch, if nothing else in `analyze` uses it.

- [ ] **Step 6: Check for stale references**

Run: `grep -rn "\.fusion\|adaptfuse_v1.pth\|dfire_yolo26n" --include=*.py --include=*.md --include=*.js . ; echo exit=$?`
Expected: no matches in `engine/`, `live/` or `gradio_app.py` (README matches are fixed in Task 8). Fix any code match.

- [ ] **Step 7: Run all tests, then a live smoke run**

Run: `python -m pytest tests -q`
Expected: 16 passed (Task 1: 7, Task 2: 7, Task 3: 2).
Run:
```bash
python -c "import sys;sys.path.insert(0,'.');from engine import load_models,Analyzer,extract_audio;import cv2;m=load_models();c=cv2.VideoCapture('samples/sample_rgb.mp4');c.set(1,70);ok,f=c.read();a=Analyzer(m,extract_audio('samples/sample_rgb.mp4'),'rgb');r=a.analyze(f,7.0);print(r['disaster']['label'],[(b['cls'],round(b['conf'],2)) for b in r['boxes']],r['timing_ms'])"
```
Expected: a `calibration.json not found` warning, the classifier line with both members and `tta=True`, then a box list containing `('smoke', 0.6x)` and **no** `fire` box for this frame.

- [ ] **Step 8: Commit**

```bash
git add engine/ scripts/export_weights.py tests/ weights/adaptfuse_v1_grouped_s41.pth weights/adaptfuse_v1_grouped_s42.pth
git commit -m "Switch scene classifier to grouped AdapFuse-V1 ensemble (seeds 41+42, fp16)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Classifier evaluation, bias tuning and `calibration.json`

**Files:**
- Create: `scripts/eval_classifier.py`
- Create (generated): `eval/classifier_eval.md`, `eval/classifier_eval.json`, `weights/calibration.json`
- Modify: repo-root `.gitignore` (ignore `code_p2/adaptfuse_uav/demo_app/eval/*.npz`)

**Interfaces:**
- Consumes: `engine.pipeline._rgb_tensor`, `_thermal_tensor`, `_logmel`; `engine.classifier.member_outputs`, `apply_bias`; `engine.models.load_member`, `CLASSIFIER_FILES`, `WEIGHTS_DIR`.
- Produces: `weights/calibration.json` = `{"stage": str, "members": [file names], "tta": bool, "bias": {"rgb": [4 floats], "thermal": [4 floats]} or {}, "tuned_on": str, "selected_by": str}`, which Task 3's `load_models` reads.

- [ ] **Step 1: Write `scripts/eval_classifier.py`**

```python
"""
Measure the demo's scene classifier on the grouped split exactly as the app feeds it
(one visual modality + audio when the row has it), stage by stage, and tune the
per-condition class bias on the val split.

Run from code_p2/adaptfuse_uav/demo_app:
    python scripts/eval_classifier.py                 # full run (~10-20 min on GTX 960)
    python scripts/eval_classifier.py --limit 300     # quick smoke run, writes nothing to weights/
    python scripts/eval_classifier.py --compare-fp32  # also check fp16 vs training checkpoints

Outputs: eval/classifier_eval.{md,json}; weights/calibration.json (full runs only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import engine  # noqa: E402,F401
import cv2  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402

from engine.classifier import apply_bias, member_outputs  # noqa: E402
from engine.labels import DISASTER_CLASSES  # noqa: E402
from engine.models import CLASSIFIER_FILES, PROJECT_DIR, WEIGHTS_DIR, load_member  # noqa: E402
from engine.pipeline import _logmel, _rgb_tensor, _thermal_tensor  # noqa: E402

SPLIT_DIR = PROJECT_DIR / "data" / "metadata_grouped" / "seed_42"
TRAIN_CKPTS = {
    "adaptfuse_v1_grouped_s41.pth": PROJECT_DIR / "outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth",
    "adaptfuse_v1_grouped_s42.pth": PROJECT_DIR / "outputs/grouped_repeated/seed_42/checkpoints/adaptfuse_v1_grouped_best.pth",
}
OUT = DEMO / "eval"
CONDITIONS = ("rgb", "thermal")
CHAIN = ("seed42", "ensemble", "ensemble+tta", "ensemble+tta+bias")   # adoption order
BATCH = 32


def load_rows(split: str, condition: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    df = df[df["has_rgb" if condition == "rgb" else "has_thermal"] == 1].reset_index(drop=True)
    if limit:
        # stratified head so every class is present in smoke runs
        df = df.groupby("disaster_label", group_keys=False).head(max(1, limit // 4)).reset_index(drop=True)
    return df


_AUDIO_CACHE: dict = {}


def audio_tensor(row) -> tuple:
    if not int(row["has_audio"]) or not isinstance(row["audio_path"], str):
        return torch.zeros(1, 1, 64, 63), 0.0
    p = row["audio_path"]
    if p not in _AUDIO_CACHE:
        y, _ = librosa.load(p, sr=16000, duration=2.0, mono=True)   # training: first 2 s, end-padded
        y = np.pad(y, (0, max(0, 32000 - len(y))))[:32000]
        _AUDIO_CACHE[p] = _logmel(y)
    return _AUDIO_CACHE[p], 1.0


def visual_tensors(row, condition: str):
    if condition == "rgb":
        return _rgb_tensor(cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)), torch.zeros(1, 1, 224, 224), 1.0, 0.0
    img = cv2.imread(row["thermal_path"], cv2.IMREAD_COLOR)
    return torch.zeros(1, 3, 224, 224), _thermal_tensor(img), 0.0, 1.0


def member_probs(members: dict, df: pd.DataFrame, condition: str, device) -> dict:
    """{member_name: {"orig": {head: (N,K)}, "flip": {...}}} as numpy."""
    acc = {n: {"orig": {}, "flip": {}} for n in members}
    for s in range(0, len(df), BATCH):
        rows = [df.iloc[i] for i in range(s, min(s + BATCH, len(df)))]
        vis = [visual_tensors(r, condition) for r in rows]
        aud = [audio_tensor(r) for r in rows]
        rgb = torch.cat([v[0] for v in vis]); th = torch.cat([v[1] for v in vis])
        au = torch.cat([a[0] for a in aud])
        has = torch.tensor([[v[2], v[3], a[1]] for v, a in zip(vis, aud)], dtype=torch.float32)
        for view, (r_, t_) in (("orig", (rgb, th)), ("flip", (torch.flip(rgb, [3]), torch.flip(th, [3])))):
            for name, m in members.items():
                o = member_outputs(m, r_.to(device), t_.to(device), au.to(device), has.to(device))
                for k in ("disaster", "victim"):
                    acc[name][view].setdefault(k, []).append(o[k].numpy())
        print(f"\r  {condition}: {min(s + BATCH, len(df))}/{len(df)}", end="", flush=True)
    print()
    return {n: {v: {k: np.concatenate(x) for k, x in d.items()} for v, d in views.items()}
            for n, views in acc.items()}


def stage_probs(P: dict, stage: str, bias=None) -> dict:
    s41, s42 = "adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth"
    if stage == "seed42":
        return P[s42]["orig"]
    if stage == "seed41":
        return P[s41]["orig"]
    views = [P[s41]["orig"], P[s42]["orig"]]
    if stage.startswith("ensemble+tta"):
        views += [P[s41]["flip"], P[s42]["flip"]]
    out = {k: np.mean([v[k] for v in views], axis=0) for k in ("disaster", "victim")}
    if stage == "ensemble+tta+bias":
        out["disaster"] = apply_bias(out["disaster"], bias)
    return out


def metrics(p: dict, df: pd.DataFrame) -> dict:
    yd, yv = df["disaster_label"].to_numpy(), df["victim_label"].to_numpy()
    pd_, pv = p["disaster"].argmax(1), p["victim"].argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(yd, pd_, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "disaster_macro_f1": float(f1_score(yd, pd_, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "victim_macro_f1": float(f1_score(yv, pv, labels=[0, 1], average="macro", zero_division=0)),
        "per_class": {DISASTER_CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                            "f1": float(f1[i]), "support": int(sup[i])} for i in range(4)},
        "n": int(len(df)),
    }


def tune_bias(p: np.ndarray, y: np.ndarray) -> list:
    """Coordinate search on classes 1..3 (class 0 fixed at 0: softmax is shift-invariant)."""
    b = np.zeros(4)
    grid = np.round(np.arange(-2.0, 2.0001, 0.1), 2)

    def score(bb):
        return f1_score(y, apply_bias(p, bb).argmax(1), labels=[0, 1, 2, 3], average="macro", zero_division=0)

    best = score(b)
    for _ in range(3):
        for c in (1, 2, 3):
            for g in grid:
                trial = b.copy(); trial[c] = g
                s = score(trial)
                if s > best + 1e-9:
                    best, b = s, trial
    return [float(x) for x in b]


def compare_fp32(members: dict, device, limit: int | None) -> dict:
    df = load_rows("test", "rgb", limit)
    fp32 = {n: load_member(TRAIN_CKPTS[n], device) for n in members}
    a = member_probs(members, df, "rgb", device)
    b = member_probs(fp32, df, "rgb", device)
    res = {}
    for n in members:
        pa, pb = a[n]["orig"]["disaster"], b[n]["orig"]["disaster"]
        res[n] = {"argmax_agreement": float((pa.argmax(1) == pb.argmax(1)).mean()),
                  "max_abs_prob_diff": float(np.abs(pa - pb).max()), "n": int(len(df))}
    return res


def write_report(results: dict, chosen: str, cal: dict, fp: dict | None, secs: float, limit) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "classifier_eval.json").write_text(json.dumps(
        {"results": results, "chosen_stage": chosen, "calibration": cal, "fp16_check": fp,
         "limit": limit, "seconds": round(secs, 1)}, indent=1))
    L = ["# Scene classifier evaluation (grouped split, seed_42)", "",
         "Input condition as in the demo app: **one** visual modality + audio where the row has it. "
         "Bias tuned on **val**; numbers below are **test**. "
         "The adoption rule compares stages on test macro-F1, so the chosen stage is selected on test.", ""]
    if limit:
        L += [f"> Smoke run with --limit {limit}; not a real measurement.", ""]
    for c in CONDITIONS:
        L += [f"## Condition: {c}  (n = {results[c]['seed42']['n']})", "",
              "| Stage | Disaster macro-F1 | F1 normal | F1 fire/smoke | F1 collapse/flood | F1 other | Victim macro-F1 |",
              "|---|---|---|---|---|---|---|"]
        for st in ("seed41",) + CHAIN:
            m = results[c][st]; pc = m["per_class"]
            mark = " **(in app)**" if st == chosen else ""
            L.append(f"| {st}{mark} | {m['disaster_macro_f1']:.4f} | " + " | ".join(
                f"{pc[k]['f1']:.4f}" for k in DISASTER_CLASSES) + f" | {m['victim_macro_f1']:.4f} |")
        L += ["", f"Per-class precision / recall for the app stage (`{chosen}`):", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for k, v in results[c][chosen]["per_class"].items():
            L.append(f"| {k} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
        L.append("")
    L += ["## Calibration written to `weights/calibration.json`", "", "```json", json.dumps(cal, indent=1), "```", ""]
    if fp:
        L += ["## fp16 vs fp32 check (test, rgb)", "", "| Member | Argmax agreement | Max abs Δp | n |", "|---|---|---|---|"]
        L += [f"| {n} | {v['argmax_agreement']:.5f} | {v['max_abs_prob_diff']:.5f} | {v['n']} |" for n, v in fp.items()]
        L.append("")
    L.append(f"Run time: {secs / 60:.1f} min.")
    (OUT / "classifier_eval.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare-fp32", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    members = {n: load_member(WEIGHTS_DIR / n, dev) for n in CLASSIFIER_FILES}

    results, bias = {}, {}
    for c in CONDITIONS:
        print(f"[val] {c}")
        dv = load_rows("val", c, args.limit)
        Pv = member_probs(members, dv, c, dev)
        bias[c] = tune_bias(stage_probs(Pv, "ensemble+tta")["disaster"], dv["disaster_label"].to_numpy())
        print(f"  bias {c}: {bias[c]}")
        print(f"[test] {c}")
        dt = load_rows("test", c, args.limit)
        Pt = member_probs(members, dt, c, dev)
        results[c] = {st: metrics(stage_probs(Pt, st, bias[c]), dt) for st in ("seed41",) + CHAIN}

    chosen = CHAIN[0]
    for st in CHAIN[1:]:
        if all(results[c][st]["disaster_macro_f1"] >= results[c][chosen]["disaster_macro_f1"] for c in CONDITIONS):
            chosen = st
    cal = {
        "stage": chosen,
        "members": ["adaptfuse_v1_grouped_s42.pth"] if chosen == "seed42" else list(CLASSIFIER_FILES),
        "tta": chosen.startswith("ensemble+tta"),
        "bias": bias if chosen == "ensemble+tta+bias" else {},
        "tuned_on": "data/metadata_grouped/seed_42/val.csv",
        "selected_by": "adoption rule on data/metadata_grouped/seed_42/test.csv disaster macro-F1",
    }
    fp = compare_fp32(members, dev, args.limit) if args.compare_fp32 else None
    write_report(results, chosen, cal, fp, time.time() - t0, args.limit)
    if not args.limit:
        (WEIGHTS_DIR / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
        print(f"[ok] wrote {WEIGHTS_DIR / 'calibration.json'}")
    print(f"[ok] chosen stage: {chosen}; report: {OUT / 'classifier_eval.md'}")
    if fp and any(v["argmax_agreement"] < 0.999 for v in fp.values()):
        print("[FAIL] fp16 argmax agreement < 99.9 %")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run on a small slice**

Run: `python scripts/eval_classifier.py --limit 200`
Expected: progress lines for val/test × rgb/thermal, `bias rgb: [0.0, …]`, and `[ok] chosen stage: …`. `eval/classifier_eval.md` exists with both condition tables and the smoke-run warning. `weights/calibration.json` is **not** created.

- [ ] **Step 3: Full run with the fp16 check**

Run: `python scripts/eval_classifier.py --compare-fp32`
Expected: runs for about 10–25 min, ends with `[ok] wrote …calibration.json` and exit code 0. fp16 argmax agreement ≥ 0.999 for both members.

- [ ] **Step 4: Read the report and sanity-check it**

Open `eval/classifier_eval.md`. Check that:
* `seed42` rgb/thermal rows exist;
* support for collapse/flood is 936 and for other is 438 in both conditions;
* the `(in app)` marker sits on the chosen stage;
* `weights/calibration.json` has matching `stage`, `members`, `tta` and `bias`.

If the ensemble is worse than seed42 in a condition, the rule keeps seed42. Record that plainly in the task summary.

- [ ] **Step 5: Ignore the cache files and commit**

Append to repo-root `.gitignore`:
```text
code_p2/adaptfuse_uav/demo_app/eval/*.npz
```
```bash
git add scripts/eval_classifier.py eval/classifier_eval.md eval/classifier_eval.json weights/calibration.json ../../../.gitignore
git commit -m "Evaluate classifier stages on grouped split and write calibration.json

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `TemporalState` — smoothing, box persistence, agreement

**Files:**
- Create: `engine/temporal.py`
- Test: `tests/test_temporal.py`

**Interfaces:**
- Produces: `iou(a: list[float], b: list[float]) -> float` for normalized xyxy boxes.
- Produces: `ema_alpha(dt: float, tau: float) -> float` = `1 - exp(-max(dt, 0)/tau)`.
- Produces: `TemporalState(tau=0.5, window=3, min_hits=2, iou_thr=0.3, reset_gap=2.0, disagree_conf=0.5, classifier_only_s=2.0, scene_names=None)`, with `.reset()` and `.update(t: float, raw: dict[str, np.ndarray], boxes: list[dict]) -> tuple[dict[str, np.ndarray], list[dict], dict]`. The return is `(smoothed_probs, confirmed_boxes, agreement)`. `agreement = {"status": "agree" | "detector_only" | "classifier_only", "since": float, "reason": str}`. Confirmed boxes are copies of the current boxes, with `conf` averaged over matches and `hits: int` added. The `raw` keys used are `disaster` (4), `victim` (2) and `nuisance` (5). Scene index 1 means `fire / smoke`.

- [ ] **Step 1: Write the failing tests**

`tests/test_temporal.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.temporal'`.

- [ ] **Step 3: Implement `engine/temporal.py`**

```python
"""
Temporal consistency for the demo: time-based EMA of class probabilities, box
persistence (k-of-n frames), and a scene-vs-detector agreement flag. Nothing here
overrides a model output; it only smooths and annotates.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

FIRE_SCENE = 1                      # index of "fire / smoke" in DISASTER_CLASSES
HAZARD_BOXES = ("fire", "smoke")


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ema_alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-max(dt, 0.0) / tau)


class TemporalState:
    def __init__(self, tau: float = 0.5, window: int = 3, min_hits: int = 2, iou_thr: float = 0.3,
                 reset_gap: float = 2.0, disagree_conf: float = 0.5, classifier_only_s: float = 2.0,
                 scene_names: list | None = None):
        self.tau, self.window, self.min_hits, self.iou_thr = tau, window, min_hits, iou_thr
        self.reset_gap, self.disagree_conf, self.classifier_only_s = reset_gap, disagree_conf, classifier_only_s
        self.scene_names = scene_names
        self.reset()

    def reset(self) -> None:
        self.last_t = None
        self.probs = None
        self.history = deque(maxlen=self.window)
        self.fire_scene_since = None
        self.last_hazard_box_t = None
        self.status, self.status_since = "agree", None

    def update(self, t: float, raw: dict, boxes: list):
        if self.last_t is not None and (t < self.last_t - 1e-6 or t - self.last_t > self.reset_gap):
            self.reset()
        cur = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        if self.probs is None:
            self.probs = {k: v.copy() for k, v in cur.items()}
        else:
            a = ema_alpha(t - self.last_t, self.tau)
            self.probs = {k: self.probs[k] + a * (cur[k] - self.probs[k]) for k in cur}
        self.last_t = t

        self.history.append(boxes)
        past_frames = list(self.history)[:-1]
        confirmed = []
        for b in boxes:
            matches = [b]
            for past in past_frames:
                same = [p for p in past if p["cls"] == b["cls"]]
                if not same:
                    continue
                best = max(same, key=lambda p: iou(p["box"], b["box"]))
                if iou(best["box"], b["box"]) >= self.iou_thr:
                    matches.append(best)
            if len(matches) >= self.min_hits:
                confirmed.append({**b, "conf": float(np.mean([m["conf"] for m in matches])),
                                  "hits": len(matches)})
        return {k: v.copy() for k, v in self.probs.items()}, confirmed, self._agreement(t, confirmed)

    def _agreement(self, t: float, confirmed: list) -> dict:
        scene = int(np.argmax(self.probs["disaster"]))
        hazard = [b for b in confirmed if b["cls"] in HAZARD_BOXES]
        if hazard:
            self.last_hazard_box_t = t
        if scene == FIRE_SCENE:
            if self.fire_scene_since is None:
                self.fire_scene_since = t
        else:
            self.fire_scene_since = None

        status, reason = "agree", ""
        strong = [b for b in hazard if b["conf"] >= self.disagree_conf]
        if strong and scene != FIRE_SCENE:
            top = max(strong, key=lambda b: b["conf"])
            reason = f'detector: {top["cls"]} {top["conf"]:.2f}'
            if self.scene_names:
                reason += f" · scene: {self.scene_names[scene]}"
            status = "detector_only"
        elif scene == FIRE_SCENE:
            ref = self.fire_scene_since
            if self.last_hazard_box_t is not None:
                ref = max(ref, self.last_hazard_box_t)
            if t - ref >= self.classifier_only_s - 1e-9:
                status = "classifier_only"
                reason = f"scene: fire / smoke for {t - ref:.1f} s · detector: no fire/smoke box"

        if status != self.status or self.status_since is None:
            self.status, self.status_since = status, t
        return {"status": status, "since": self.status_since, "reason": reason}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_temporal.py -q`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/temporal.py tests/test_temporal.py
git commit -m "Add TemporalState: EMA smoothing, 2-of-3 box persistence, agreement flag

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Wire temporal logic into `Analyzer`

**Files:**
- Modify: `engine/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: `TemporalState` (Task 5); `SceneClassifier` via `bundle.classifier` (Task 3).
- Produces: `Analyzer.reset() -> None`. `Analyzer.set_modality(m)` resets when the modality changes.
- Produces: the `Analyzer.analyze(...)` result dict gains these keys. `boxes` now means **confirmed** boxes. `raw_boxes` holds every detection this frame. `raw` = `{"disaster", "victim", "nuisance"}`, each a `_top` dict of the unsmoothed probabilities. `agreement` = the dict from `TemporalState`. `disaster` / `victim` / `nuisance` are now **smoothed**. All other keys are unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline_smoke.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_smoke.py -q`
Expected: FAIL with `KeyError`/`AssertionError: raw_boxes` and `AttributeError: 'Analyzer' object has no attribute 'temporal'`.

- [ ] **Step 3: Implement the changes in `engine/pipeline.py`**

Add the import `from .temporal import TemporalState`. In `Analyzer.__init__`, append:
```python
        self.temporal = TemporalState(scene_names=DISASTER_CLASSES)
```
Replace `set_modality` and add `reset`:
```python
    def reset(self) -> None:
        """Forget smoothing / box history (new video position or new input condition)."""
        self.temporal.reset()

    def set_modality(self, modality: str) -> None:
        if modality not in ("rgb", "thermal"):
            raise ValueError(modality)
        if modality != self.modality:
            self.modality = modality
            self.reset()
```
In `analyze`, directly after `t_cls = time.perf_counter()`, add:
```python
        raw = {"disaster": cls["disaster"], "victim": cls["victim"], "nuisance": cls["nuisance"]}
        smooth, confirmed, agreement = self.temporal.update(float(t), raw, boxes)
```
and in the returned dict replace the four entries `"boxes"`, `"disaster"`, `"victim"`, `"nuisance"` with:
```python
            "boxes": confirmed,
            "raw_boxes": boxes,
            "disaster": _top(smooth["disaster"], DISASTER_CLASSES),
            "victim": _top(smooth["victim"], VICTIM_CLASSES),
            "nuisance": _top(smooth["nuisance"], NUISANCE_CLASSES),
            "raw": {"disaster": _top(raw["disaster"], DISASTER_CLASSES),
                    "victim": _top(raw["victim"], VICTIM_CLASSES),
                    "nuisance": _top(raw["nuisance"], NUISANCE_CLASSES)},
            "agreement": agreement,
```
Update the module docstring's first paragraph to mention: "labels are EMA-smoothed (τ = 0.5 s), boxes must persist 2 of 3 analyzed frames, and an agreement flag compares scene and detector".

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests -q`
Expected: 31 passed (16 + 12 temporal + 3 pipeline). The pipeline tests take about 10–20 s on CPU.

- [ ] **Step 5: Commit**

```bash
git add engine/pipeline.py tests/test_pipeline_smoke.py
git commit -m "Smooth labels, confirm persistent boxes and flag disagreement in Analyzer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: UI — live badge and event log; Gradio panel, CSV and summary

**Files:**
- Modify: `live/static/index.html`, `live/static/app.js`, `live/static/style.css`
- Modify: `engine/render.py`
- Modify: `gradio_app.py`

**Interfaces:**
- Consumes: result keys `agreement`, `raw`, `raw_boxes` (Task 6).

- [ ] **Step 1: Live page — badge element and provenance text**

In `live/static/index.html`, inside `<div class="stage" id="stage">` after the `modBadge` div, add:
```html
        <div class="disagree" id="disagree" hidden></div>
```
Replace the `<p class="provenance">…</p>` body with:
```html
        Scene, victim and nuisance labels and RUE come from an ensemble of two <b>AdapFuse-V1</b> models
        (grouped split, seeds 41 + 42, flip TTA, class calibration); the audio-only label from the pipeline
        AudioCNN; fire/smoke boxes from <b>YOLO26s fine-tuned on CLAHE D-Fire</b>. These are all
        trained in the AdapFuse-UAV pipeline. <b>Person boxes</b> use COCO-pretrained YOLO26n and
        <b>sound tags</b> use AudioSet-pretrained PANNs Cnn6; neither was fine-tuned here. Labels are
        smoothed over ~1 s and boxes are shown once they persist for 2 of 3 analyzed frames.
```

- [ ] **Step 2: Live page — style**

Append to `live/static/style.css`:
```css
.disagree { position: absolute; left: 10px; bottom: 56px; max-width: calc(100% - 20px);
  padding: 6px 12px; border-radius: 8px; background: rgba(40, 28, 0, .85);
  border: 1px solid var(--warn); color: var(--warn); font-size: 13px; font-weight: 600; }
```

- [ ] **Step 3: Live page — script**

In `live/static/app.js`:
1. Change `let lastScene = null, lastVictim = null, lastTag = null;` to `let lastScene = null, lastVictim = null, lastTag = null, lastAgree = null;`
2. In `resetSession()`, change `lastScene = lastVictim = lastTag = null;` to `lastScene = lastVictim = lastTag = lastAgree = null; $("disagree").hidden = true;`
3. Add near the top, after `const SEND_WIDTH = 640;`:
```js
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
```
4. At the end of `updatePanel(r)`, before the closing brace, add:
```js
  const ag = r.agreement || { status: "agree", reason: "" };
  const badge = $("disagree");
  if (ag.status === "agree") badge.hidden = true;
  else { badge.hidden = false; badge.textContent = `⚠ models disagree — ${ag.reason}`; }
  if (ag.status !== lastAgree) {
    if (lastAgree !== null || ag.status !== "agree")
      logEvent(r.t, ag.status === "agree" ? "Scene and detector agree again"
        : `<b style="color:var(--warn)">Models disagree</b> — ${esc(ag.reason)}`);
    lastAgree = ag.status;
  }
```

- [ ] **Step 4: Burned-in panel — `engine/render.py`**

Add `WARN = (32, 176, 255)  # BGR amber` below `MUTED`. In `render_frame`, directly after the nuisance line block (the `if n:` block), add:
```python
    ag = res.get("agreement", {"status": "agree"})
    if ag["status"] != "agree":
        _text(panel, "MODELS DISAGREE", (x, y), 0.5, WARN, 2); y += 18
        for part in ag["reason"].split(" · ")[:2]:
            _text(panel, part[:36], (x, y), 0.42, WARN); y += 16
        y += 6
```

- [ ] **Step 5: Gradio — CSV columns, summary and model note**

In `gradio_app.py`:
1. In the CSV header list, after `"ms"`, add `"agreement", "raw_disaster", "n_raw_boxes"`. At the end of the row list, after `r["timing_ms"]["total"]`, add `r["agreement"]["status"], r["raw"]["disaster"]["label"], len(r["raw_boxes"])`.
2. In `_summary`, before `lines = [`, add:
```python
    dis = [r["agreement"]["status"] for r in rows]
    det_only, cls_only = dis.count("detector_only"), dis.count("classifier_only")
```
and after the `"**Most frequent sound tag:** "` entry, add:
```python
        f"  \n**Scene vs. detector disagreement:** {(det_only + cls_only) / max(n, 1):.0%} of analyzed frames "
        f"(detector-only {det_only / max(n, 1):.0%}, classifier-only {cls_only / max(n, 1):.0%})",
```
3. Replace `MODEL_NOTE` with:
```python
MODEL_NOTE = """
**Where each output comes from.** Scene, victim and nuisance labels and the RUE reliability
bars come from an ensemble of two **AdapFuse-V1** models (grouped split, seeds 41 + 42, flip TTA,
class calibration; see `demo_app/eval/classifier_eval.md`), trained in the AdapFuse-UAV pipeline.
The audio-only label comes from the pipeline's AudioCNN baseline. Fire and smoke boxes come from
**YOLO26s fine-tuned on CLAHE D-Fire** (test mAP@50 0.771). **Person boxes** come from the
COCO-pretrained YOLO26n, and **sound tags** from the AudioSet-pretrained PANNs Cnn6; neither was
fine-tuned in this work. Labels are smoothed over ~1 s; boxes are drawn once they persist for
2 of 3 analyzed frames; *models disagree* marks frames where scene and boxes contradict.
"""
```

- [ ] **Step 6: Run the tests and the headless Gradio path**

Run: `python -m pytest tests -q`, expecting all pass.
Run:
```bash
python -c "import sys;sys.path.insert(0,'.');import gradio_app as g;P=lambda *a,**k:None;v,p,s,f=g.analyze_video('samples/sample_rgb.mp4','auto',1,0.25,0.35,0,progress=P);print(s)"
```
Expected: a summary including `Scene vs. detector disagreement:` and `Max boxes in one frame: fire …, smoke ≥1`. The CSV (path in `f[0]`) has the three new columns.

- [ ] **Step 7: Commit**

```bash
git add live/static/ engine/render.py gradio_app.py
git commit -m "Show models-disagree badge, log agreement changes, export agreement in Gradio

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Speed and acceptance checks, docs

**Files:**
- Modify: `demo_app/README.md`, repo-root `README.md`
- Possibly modify: `weights/calibration.json` (`"tta": false`) and `eval/classifier_eval.md` (note), only if the speed target is missed

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Measure the engine analysis rate on the GTX 960**

Run:
```bash
python -c "
import sys,time;sys.path.insert(0,'.')
from engine import load_models,Analyzer,extract_audio
import cv2
m=load_models();a=Analyzer(m,extract_audio('samples/sample_rgb.mp4'),'rgb')
c=cv2.VideoCapture('samples/sample_rgb.mp4');fr=[]
while True:
    ok,f=c.read()
    if not ok: break
    fr.append(f)
for i in range(5): a.analyze(fr[i],i*0.1)
a.reset();t=time.perf_counter()
for i,f in enumerate(fr[:60]): r=a.analyze(f,i*0.1)
print(f'{60/(time.perf_counter()-t):.1f} analyses/s', r['timing_ms'])"
```
Expected: ≥ 9 analyses/s. This is the engine only; the live app adds about 10 % for JPEG and WebSocket overhead. If it's lower, set `"tta": false` in `weights/calibration.json`, append `> Flip TTA disabled in the app: engine speed X/s < target.` to `eval/classifier_eval.md`, and re-measure.

- [ ] **Step 2: Live acceptance in the browser**

Start `python live/server.py` and open <http://127.0.0.1:8000>. Click **sample_rgb** and check:
* around 7 s, the white plume is boxed as **smoke**, not fire;
* the *Analysis rate* readout shows ≥ 8 /s;
* the badge appears or disappears with a matching event-log entry;
* seeking back does not leave stale boxes.

Then click **sample_thermal** and check that the thermal badge and the detector note show. Record the measured rate.

- [ ] **Step 3: Update `demo_app/README.md`**

In the "What is shown" table, change these rows:
```markdown
| Disaster class (normal · fire/smoke · collapse/flood · other), victim yes/no, nuisance type | Ensemble of 2 × AdapFuse-V1 (grouped split, seeds 41 + 42) + flip TTA + class calibration — see [`eval/classifier_eval.md`](eval/classifier_eval.md) | `adaptfuse_v1_grouped_s41.pth`, `adaptfuse_v1_grouped_s42.pth`, `calibration.json` | trained in pipeline |
| **Fire / smoke boxes** | YOLO26s fine-tuned on CLAHE-enhanced D-Fire (test mAP@50 = 0.771); frames are CLAHE-enhanced before detection | `dfire_yolo26s_clahe.pt` | trained in pipeline |
```
Add a section after "How one RGB-or-thermal video is handled":
```markdown
### Stability and disagreement

* **Smoothing:** class probabilities are averaged over time,
  $\bar p_t = \bar p_{t'} + (1-e^{-\Delta t/\tau})(p_t-\bar p_{t'})$ with $\tau = 0.5$ s,
  and reset on seeking.
* **Box persistence:** a box is drawn only if the same class (IoU ≥ 0.3) is detected in
  2 of the last 3 analyzed frames.
* **"Models disagree"** appears when a persistent fire/smoke box with confidence ≥ 0.5
  contradicts a non-fire scene, or when the scene has said fire/smoke for ≥ 2 s without
  any fire/smoke box. Both outputs stay visible and nothing is overridden.
* The detector refuses to load a checkpoint whose class names are not
  `{0: smoke, 1: fire}`. An earlier bundled checkpoint had them swapped.
```
Replace the timing table's rows with the numbers measured in Steps 1–2. Add under Layout: `├── eval/  classifier_eval.md — stage-by-stage accuracy on the grouped split`, and under `scripts/` add `check_detector.py` and `eval_classifier.py`, and `tests/` (run `python -m pytest tests -q`).

- [ ] **Step 4: Update the root `README.md` provenance table**

Replace the first and third rows of the "Shown in the app" table:
```markdown
| Disaster / victim / nuisance labels, RUE reliability | AdapFuse-V1 ensemble (grouped seeds 41 + 42), trained in pipeline |
| Fire / smoke boxes | YOLO26s fine-tuned on CLAHE D-Fire, trained in pipeline |
```

- [ ] **Step 5: Final test run and commit**

Run: `python -m pytest tests -q`, expecting all pass.
```bash
git add ../../../README.md README.md weights/calibration.json eval/classifier_eval.md
git commit -m "Document detector fix, classifier ensemble and stability rules; record speed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
