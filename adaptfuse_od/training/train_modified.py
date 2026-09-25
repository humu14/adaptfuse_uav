"""Train the proposed hazard-aware detector.

This compatibility entry point replaces the previous prototype trainer, which
did not use ground-truth boxes. Training is delegated to Ultralytics so the
native YOLO26 end-to-end detection objective is optimized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_benchmark import train_single_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26n with HPG")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "efficient_study" / "hazard_yolo26n.yaml"),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--no_fire_prior", action="store_true")
    parser.add_argument("--no_smoke_prior", action="store_true")
    parser.add_argument("--no_detail_prior", action="store_true")
    args = parser.parse_args()

    overrides = {"model": {"hazard_prior_gate": {}}}
    gate = overrides["model"]["hazard_prior_gate"]
    if args.no_fire_prior:
        gate["use_fire_prior"] = False
    if args.no_smoke_prior:
        gate["use_smoke_prior"] = False
    if args.no_detail_prior:
        gate["use_detail_prior"] = False
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.image_size is not None:
        overrides.setdefault("dataset", {})["image_size"] = args.image_size

    train_single_model(args.config, overrides)


if __name__ == "__main__":
    main()
