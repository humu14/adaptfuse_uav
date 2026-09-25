"""
Run All Benchmark Models — Sequential training of all 8 architectures
======================================================================
Trains each model one after another on the D-Fire dataset, then generates
a comparison table. Saves per-model results and a combined summary.

Usage:
    python training/run_all_benchmark.py                        # Run all 8
    python training/run_all_benchmark.py --models yolov8n yolov11n  # Subset
    python training/run_all_benchmark.py --skip_existing         # Resume
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from training.train_benchmark import train_single_model
from models.build_model import get_model_info, ALL_MODELS

# Reduced controlled pair (baseline and proposed HPG model).
# (lightest → heaviest, so you see results fast)
BENCHMARK_CONFIGS = [
    ("yolov10n", "configs/efficient_study/yolov10n_baseline.yaml"),
    ("yolo26n", "configs/efficient_study/yolo26n_baseline.yaml"),
    ("hazard_yolo26n", "configs/efficient_study/hazard_yolo26n.yaml"),
]


def run_all_benchmarks(models: list = None, output_dir: str = "outputs/efficient_study",
                       skip_existing: bool = False, overrides: dict = None):
    """
    Run all (or selected) benchmark models sequentially.
    
    Args:
        models: List of model names to train (None = both core models)
        output_dir: Base output directory
        skip_existing: Skip models that already have results
        overrides: Config overrides to apply to all models
    """
    configs_to_run = BENCHMARK_CONFIGS
    if models:
        models_set = set(m.lower() for m in models)
        configs_to_run = [(n, c) for n, c in BENCHMARK_CONFIGS if n in models_set]

    total = len(configs_to_run)
    results_all = {}
    start_all = time.time()

    print(f"\n{'='*70}")
    print(f"  AdapFuse-OD -- Controlled Baseline/HPG Study")
    print(f"  Models to train: {total}")
    print(f"  Output: {output_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for idx, (name, config_path) in enumerate(configs_to_run, 1):
        config_full = PROJECT_ROOT / config_path

        # Check if already trained
        preview = load_config(str(config_full), {"output_dir": output_dir})
        result_dir = Path(preview["output_path"])
        if skip_existing and (result_dir / "training_summary.json").exists():
            print(f"\n  [{idx}/{total}] SKIPPING {name} -- already trained")
            # Load existing results
            with open(result_dir / "training_summary.json") as f:
                results_all[name] = json.load(f)
            continue

        print(f"\n  [{idx}/{total}] Training: {name}")
        print(f"  {'-'*50}")

        try:
            merged_overrides = {"output_dir": output_dir}
            if overrides:
                merged_overrides.update(overrides)

            result = train_single_model(str(config_full), merged_overrides)
            results_all[name] = {
                "status": "completed",
                "architecture": name,
            }
        except Exception as e:
            print(f"\n  [X] ERROR training {name}: {e}")
            results_all[name] = {
                "status": "failed",
                "architecture": name,
                "error": str(e),
            }

    total_time = time.time() - start_all

    # Save combined summary
    summary = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "total_models": total,
        "total_time_hours": total_time / 3600,
        "models": results_all,
    }

    summary_path = Path(output_dir) / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Benchmark Complete -- {total} models in {total_time/3600:.2f} hours")
    print(f"{'='*70}")
    print(f"\n  {'Model':<30} {'Status':<12} {'Family':<10} {'Params':<8}")
    print(f"  {'-'*60}")
    for name, res in results_all.items():
        info = get_model_info(name)
        status = res.get("status", "unknown")
        icon = "[OK]" if status == "completed" else "[X]"
        print(f"  {icon} {name:<28} {status:<12} {info['family']:<10} {info['params_m']:.1f}M")

    print(f"\n  Results saved to: {summary_path}")
    print(f"  Next step: python evaluation/compare_models.py --results_dir {output_dir}")
    print()

    return results_all


def main():
    parser = argparse.ArgumentParser(description="Run the three-model focused study")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models (default: YOLOv10n, YOLO26n, YOLO26n+HPG)")
    parser.add_argument("--output_dir", type=str, default="outputs/efficient_study",
                        help="Base output directory")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models that already have results")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all models")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size for all models")
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["training"] = {"epochs": args.epochs}
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size

    run_all_benchmarks(
        models=args.models,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
