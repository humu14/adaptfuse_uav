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
