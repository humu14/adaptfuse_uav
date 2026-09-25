"""Run the reduced object-detection study.

The core comparison has three focused runs:

1. pretrained YOLOv10n historical baseline;
2. pretrained YOLO26n current baseline;
3. the same YOLO26n with the proposed Hazard Prior Gate (HPG).

Two short diagnostic ablations are available explicitly. They are not part of
the default run, which prevents accidental multi-day benchmark sweeps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from training.train_benchmark import train_single_model


CORE_EXPERIMENTS = (
    "configs/efficient_study/yolov10n_baseline.yaml",
    "configs/efficient_study/yolo26n_baseline.yaml",
    "configs/efficient_study/hazard_yolo26n.yaml",
)
ABLATION_EXPERIMENTS = (
    "configs/efficient_study/ablation_no_spectral.yaml",
    "configs/efficient_study/ablation_no_detail.yaml",
)


def _prepare_smoke_dataset(source_root: Path) -> Path:
    """Create a tiny deterministic dataset that exercises all three splits."""
    smoke_root = PROJECT_ROOT / "outputs" / "efficient_study" / "_smoke_dataset"
    split_sizes = {"train": 24, "val": 12, "test": 12}
    for split, limit in split_sizes.items():
        image_target = smoke_root / "images" / split
        label_target = smoke_root / "labels" / split
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        for old_file in (*image_target.glob("*"), *label_target.glob("*")):
            if old_file.is_file():
                old_file.unlink()

        all_images = sorted((source_root / "images" / split).glob("*"))
        positive, background = [], []
        for image in all_images:
            label = source_root / "labels" / split / f"{image.stem}.txt"
            target_list = positive if label.exists() and label.stat().st_size > 0 else background
            target_list.append(image)
        positive_count = min(len(positive), limit // 2)
        images = positive[:positive_count] + background[: limit - positive_count]
        if len(images) < limit:
            raise FileNotFoundError(f"Not enough source images for smoke split: {split}")
        for image in images:
            shutil.copy2(image, image_target / image.name)
            label = source_root / "labels" / split / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, label_target / label.name)
            else:
                (label_target / f"{image.stem}.txt").touch()
    return smoke_root


def run_study(
    include_ablations: bool = False,
    skip_existing: bool = False,
    smoke_test: bool = False,
) -> dict:
    configs = list(CORE_EXPERIMENTS)
    if include_ablations:
        configs.extend(ABLATION_EXPERIMENTS)

    results = {}
    started = time.time()
    for index, relative_config in enumerate(configs, start=1):
        config_path = PROJECT_ROOT / relative_config
        config = load_config(str(config_path))
        summary_path = Path(config["output_path"]) / "training_summary.json"
        name = config["experiment_name"]

        if skip_existing and summary_path.exists():
            print(f"[{index}/{len(configs)}] Skipping completed run: {name}")
            with open(summary_path, encoding="utf-8") as handle:
                results[name] = json.load(handle)
            continue

        overrides = None
        if smoke_test:
            smoke_dataset = _prepare_smoke_dataset(Path(config["dataset"]["root"]))
            overrides = {
                "experiment_name": f"smoke_{name}",
                "output_dir": str(PROJECT_ROOT / "outputs" / "efficient_study" / "_smoke_runs"),
                "dataset": {"root": str(smoke_dataset), "image_size": 128},
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                    "cache": False,
                    "close_mosaic": 0,
                    "plots": False,
                },
                "evaluation": {"run_test_after_training": False},
            }

        print(f"\n[{index}/{len(configs)}] Starting: {name}")
        try:
            train_single_model(str(config_path), overrides)
            effective = load_config(str(config_path), overrides)
            completed_summary = Path(effective["output_path"]) / "training_summary.json"
            with open(completed_summary, encoding="utf-8") as handle:
                results[name] = json.load(handle)
        except Exception as error:
            results[name] = {"status": "failed", "error": str(error)}
            raise

    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_hours": (time.time() - started) / 3600,
        "core_run_count": len(CORE_EXPERIMENTS),
        "ablation_run_count": len(ABLATION_EXPERIMENTS) if include_ablations else 0,
        "results": results,
    }
    summary_folder = "_smoke_runs" if smoke_test else ""
    output = PROJECT_ROOT / "outputs" / "efficient_study" / summary_folder / "study_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the YOLOv10n/YOLO26n/HPG study")
    parser.add_argument(
        "--include_ablations",
        action="store_true",
        help="Also run the two optional 15-epoch cue ablations",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use a tiny deterministic subset for one epoch to verify the pipeline",
    )
    args = parser.parse_args()
    run_study(args.include_ablations, args.skip_existing, args.smoke_test)


if __name__ == "__main__":
    main()
