"""
Export the slim inference weights used by the demo app into demo_app/weights/.

Training checkpoints also carry optimizer/scheduler state (adaptfuse_v1_best.pth
is ~155 MB). The demo only needs model weights, so this script keeps
`model_state` + `config` and copies the detector / PANNs weights alongside.

Run from code_p2/adaptfuse_uav:
    python demo_app/scripts/export_weights.py
"""

import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]          # code_p2/adaptfuse_uav
REPO = ROOT.parents[1]                              # Co-Sup
OUT = ROOT / "demo_app" / "weights"

CLASSIFIERS = {
    "adaptfuse_v1_grouped_s41.pth": ROOT / "outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth",
    "adaptfuse_v1_grouped_s42.pth": ROOT / "outputs/grouped_repeated/seed_42/checkpoints/adaptfuse_v1_grouped_best.pth",
    "baseline_audio.pth": ROOT / "outputs/checkpoints/baseline_audio_best.pth",
}
FP16 = {"adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth"}   # halves repo size
COPIES = {
    "dfire_yolo26s_clahe.pt": REPO / "adaptfuse_od/outputs/accuracy_study/accuracy_yolo26s_clahe_640/weights/best.pt",
    "yolo26n_coco.pt": REPO / "adaptfuse_od/yolo26n.pt",
    "panns_cnn6.pth": Path(os.path.expanduser("~/.cache/panns/Cnn6_mAP=0.343.pth")),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in CLASSIFIERS.items():
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        ck = torch.load(src, map_location="cpu", weights_only=False)
        slim = {
            "model_state": ({k: (v.half() if v.is_floating_point() else v) for k, v in ck["model_state"].items()}
                            if name in FP16 else ck["model_state"]),
            "config": ck["config"],
            "metrics": ck.get("metrics"),
            "epoch": ck.get("epoch"),
        }
        torch.save(slim, OUT / name)
        print(f"[ok] {name}: {(OUT / name).stat().st_size / 1e6:.1f} MB")
    for name, src in COPIES.items():
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        if name == "panns_cnn6.pth":
            # keep only the model tensors (drops training iteration counter)
            ck = torch.load(src, map_location="cpu", weights_only=False)
            torch.save({"model": ck["model"]}, OUT / name)
        else:
            shutil.copy2(src, OUT / name)
        print(f"[ok] {name}: {(OUT / name).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
