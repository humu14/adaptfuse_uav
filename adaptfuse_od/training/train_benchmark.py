"""
Benchmark Training Script — Train any of the 8 detector architectures
=======================================================================
Single unified entry point for training. Reads a config YAML, builds
the model, and trains on the specified dataset.

Usage:
    # Train a single model
    python training/train_benchmark.py --config configs/benchmark/yolov8n.yaml

    # Override parameters
    python training/train_benchmark.py --config configs/benchmark/yolov8n.yaml \\
        --epochs 50 --batch_size 8 --image_size 416

    # Quick test (1 epoch)
    python training/train_benchmark.py --config configs/benchmark/yolov8n.yaml --epochs 1
"""

import os
import sys
import json
import time
import argparse
import torch
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config, save_config
from models.build_model import build_model, get_model_info


def _prepare_dataset_yaml(config: dict, run_dir: Path) -> Path:
    """Write a portable Ultralytics dataset file with a resolved local root."""
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_cfg["root"]).resolve()
    split_dirs = {
        "train": dataset_root / "images" / dataset_cfg.get("train_split", "train"),
        "val": dataset_root / "images" / dataset_cfg.get("val_split", "val"),
        "test": dataset_root / "images" / dataset_cfg.get("test_split", "test"),
    }
    missing = [str(path) for path in split_dirs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Object-detection dataset is incomplete. Missing image directories: "
            + ", ".join(missing)
        )

    resolved = {
        "path": str(dataset_root),
        "train": str(Path("images") / dataset_cfg.get("train_split", "train")),
        "val": str(Path("images") / dataset_cfg.get("val_split", "val")),
        "test": str(Path("images") / dataset_cfg.get("test_split", "test")),
        "names": {
            index: name
            for index, name in enumerate(dataset_cfg.get("class_names", ["smoke", "fire"]))
        },
        "nc": dataset_cfg.get("num_classes", 2),
    }
    yaml_path = run_dir / "dataset.resolved.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    return yaml_path


def _extract_ultralytics_metrics(results) -> dict:
    """Normalize the metric keys returned by supported Ultralytics versions."""
    raw = getattr(results, "results_dict", {}) or {}
    key_map = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
    }
    metrics = {}
    for source, target in key_map.items():
        if source in raw:
            metrics[target] = float(raw[source])
    return metrics


def train_single_model(config_path: str, overrides: dict = None):
    """
    Train a single model from a config file.
    
    Args:
        config_path: Path to YAML config
        overrides: Dict of CLI overrides
    
    Returns:
        Training results dictionary
    """
    # Load config
    config = load_config(config_path, overrides)
    arch = config["model"]["architecture"]
    exp_name = config.get("experiment_name", arch)

    # Print header
    info = get_model_info(arch)
    print(f"\n{'='*70}")
    print(f"  AdapFuse-OD Benchmark Training")
    print(f"{'='*70}")
    print(f"  Model:       {arch}")
    print(f"  Family:      {info['family']}")
    print(f"  Params:      {info['params_m']:.1f}M")
    print(f"  FLOPs:       {info['flops_g']:.1f}G")
    print(f"  Input size:  {info['input']}")
    print(f"  Dataset:     {config['dataset']['name']}")
    print(f"  Epochs:      {config['training']['epochs']}")
    print(f"  Batch size:  {config['training']['batch_size']}")
    print(f"  LR:          {config['training']['lr']}")
    print(f"  Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Output:      {config['output_dir']}/{exp_name}")
    print(f"  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Save the config used for this run
    output_path = Path(config["output_dir"]) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    save_config(config, output_path / "config.yaml")

    # Build model
    print("Building model...")
    model = build_model(
        architecture=arch,
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        config=config,
    )

    dataset_yaml = _prepare_dataset_yaml(config, output_path)

    # Check if a partial run exists with last.pt to auto-resume after crash
    last_pt = output_path / "weights" / "last.pt"
    extra_train_kwargs = {}
    if last_pt.exists() and not (output_path / "training_summary.json").exists():
        print(f"[AUTO-RESUME] Found partial run checkpoint: {last_pt}. Resuming training.")
        extra_train_kwargs["resume"] = True

    # Train
    start_time = time.time()

    results = model.train(
        dataset_yaml=str(dataset_yaml),
        output_dir=config["output_dir"],
        **extra_train_kwargs,
    )

    total_time = time.time() - start_time

    validation_metrics = _extract_ultralytics_metrics(results)
    test_metrics = {}
    evaluation_cfg = config.get("evaluation", {})
    if evaluation_cfg.get("run_test_after_training", False):
        best_weights = output_path / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"Best checkpoint was not created: {best_weights}")
        model.load_weights(str(best_weights))
        test_metrics = model.evaluate(
            dataset_yaml=str(dataset_yaml),
            split="test",
            conf=evaluation_cfg.get("map_conf_threshold", 0.001),
            iou=evaluation_cfg.get("nms_iou_threshold", 0.7),
        )

    # Save training summary
    summary = {
        "status": "completed",
        "architecture": arch,
        "experiment_name": exp_name,
        "total_training_time_hours": total_time / 3600,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Training Complete: {arch}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train a detector model for fire/smoke detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--validation_batch_size", type=int, default=None,
                        help="Override validation batch size for custom trainers")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override input image size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Build overrides from CLI args
    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.validation_batch_size is not None:
        overrides.setdefault("training", {})["validation_batch_size"] = args.validation_batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.image_size is not None:
        overrides.setdefault("dataset", {})["image_size"] = args.image_size
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.resume is not None:
        overrides.setdefault("training", {})["resume"] = args.resume

    train_single_model(args.config, overrides)


if __name__ == "__main__":
    main()
