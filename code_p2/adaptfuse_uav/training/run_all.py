"""
AdapFuse-UAV: Run all 8 experiments sequentially.
This script trains all design alternatives in order and collects results.

Usage:
    python training/run_all.py
    python training/run_all.py --start_from adaptfuse_v1
    python training/run_all.py --skip_baselines
    python training/run_all.py --unimodal_backbones
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


EXPERIMENTS = [
    # (config_name, description, expected_time_min)
    ("baseline_rgb",       "A1: RGB-Only Baseline",                50),
    ("baseline_thermal",   "A2: Thermal-Only Baseline",            50),
    ("baseline_audio",     "A3: Audio-Only Baseline",              30),
    ("early_fusion",       "B: Early Fusion",                      60),
    ("late_fusion",        "C: Late Fusion",                       70),
    ("intermediate_fixed", "D: Intermediate Fusion (Fixed)",       80),
    ("adaptfuse_v1",       "E: AdapFuse v1 (Main Method)",         90),
    ("adaptfuse_v2_kd",    "F: AdapFuse v2 + KD (Student)",        60),
]

# Unimodal backbone sweep (3 backbones per modality), priority order.
# Visual runs use a 12-epoch budget; times measured-estimated on GTX 960.
UNIMODAL_BACKBONE_EXPERIMENTS = [
    ("baseline_rgb_efficientnet",    "A1b: RGB-Only EfficientNet-B0",         400),
    ("baseline_audio_panns",         "A3b: Audio-Only PANNS-CNN6",            130),
    ("baseline_thermal_efficientnet","A2b: Thermal-Only EfficientNet-B0",     370),
    ("baseline_audio_gdblock",       "A3c: Audio-Only GDBlock",               115),
    ("baseline_rgb_mobilevit",       "A1c: RGB-Only MobileViT-XXS",           400),
    ("baseline_thermal_mobilevit",   "A2c: Thermal-Only MobileViT-XXS",       370),
]

# Public-benchmark protocols (AIDER 4:1:2 subset, FLAME official Training/Test).
# Splits built by scripts/build_public_benchmark_splits.py.
PUBLIC_BENCHMARK_EXPERIMENTS = [
    ("public_aider_rgb_mobilenetv3",  "AIDER: RGB-Only MobileNetV3-Small",   20),
    ("public_aider_rgb_efficientnet", "AIDER: RGB-Only EfficientNet-B0",     40),
    ("public_aider_adaptfuse_v1",     "AIDER: AdapFuse-v1",                  40),
    ("public_flame_rgb_mobilenetv3",  "FLAME: RGB-Only MobileNetV3-Small",   40),
    ("public_flame_rgb_efficientnet", "FLAME: RGB-Only EfficientNet-B0",    120),
    ("public_flame_adaptfuse_v1",     "FLAME: AdapFuse-v1",                 100),
]


def run_experiment(config_name: str, data_dir: str, output_dir: str, dry_run: bool = False, dry_run_pipeline: bool = False, force: bool = False) -> dict:
    config_path = ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"[SKIP] Config not found: {config_path}")
        return {}

    results_path = Path(output_dir) / "logs" / f"{config_name}_test_results.json"
    history_path = Path(output_dir) / "logs" / f"{config_name}_history.json"
    ckpt_path = Path(output_dir) / "checkpoints" / f"{config_name}_latest.pth"

    is_completed = False
    if not force and results_path.exists() and history_path.exists():
        try:
            with open(history_path) as f:
                hist = json.load(f)
            if len(hist) > 1 or (len(hist) == 1 and hist[0].get("time_s", 0) > 10.0):
                is_completed = True
        except Exception:
            pass

    if is_completed:
        print(f"[SKIP] Experiment {config_name} already completed. Loading saved results.")
        try:
            with open(results_path) as f:
                return json.load(f)
        except Exception:
            pass

    cmd = [
        sys.executable,
        str(ROOT / "training" / "train.py"),
        "--config", str(config_path),
        "--data_dir", data_dir,
        "--output_dir", output_dir,
    ]
    if dry_run_pipeline:
        cmd += ["--epochs", "1", "--dry_run_batches", "2"]
    elif not force and ckpt_path.exists():
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu")
            epoch = ckpt.get("epoch", 0)
            is_dry_run_ckpt = False
            if history_path.exists():
                with open(history_path) as f:
                    hist = json.load(f)
                if len(hist) == 1 and hist[0].get("time_s", 0) < 10.0:
                    is_dry_run_ckpt = True
            if epoch > 1 or (epoch == 1 and not is_dry_run_ckpt):
                cmd += ["--resume", str(ckpt_path)]
                print(f"[RESUME] Found checkpoint for {config_name} at epoch {epoch}. Resuming training...")
        except Exception as e:
            print(f"[WARN] Could not check checkpoint for {config_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Skipping actual training.")
        return {"experiment": config_name, "dry_run": True}

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Experiment {config_name} failed with code {result.returncode}")
        return {"experiment": config_name, "status": "failed"}

    # Load test results if available
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {"experiment": config_name, "status": "complete"}


def main():
    parser = argparse.ArgumentParser(description="Run all AdapFuse-UAV experiments")
    parser.add_argument("--data_dir", type=str, default=str(ROOT))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--start_from", type=str, default=None,
                        help="Start from this experiment name (skip earlier ones)")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip A1-A3 unimodal baselines")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--dry_run_pipeline", action="store_true",
                        help="Run full pipeline end-to-end with 1 epoch and 2 batches per model")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment name")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain all experiments from scratch")
    parser.add_argument("--unimodal_backbones", action="store_true",
                        help="Run the unimodal backbone sweep instead of the main 8")
    parser.add_argument("--public_benchmarks", action="store_true",
                        help="Run the AIDER/FLAME public-protocol runs instead of the main 8")
    args = parser.parse_args()
    if args.public_benchmarks:
        experiments = PUBLIC_BENCHMARK_EXPERIMENTS
    elif args.unimodal_backbones:
        experiments = UNIMODAL_BACKBONE_EXPERIMENTS
    else:
        experiments = EXPERIMENTS

    print("=" * 60)
    print("AdapFuse-UAV: Full Experiment Pipeline")
    print(f"Output dir: {args.output_dir}")
    print(f"GPU: {'Available' if __import__('torch').cuda.is_available() else 'NOT AVAILABLE'}")
    print("=" * 60)

    all_results = {}
    start_found = args.start_from is None

    for config_name, desc, expected_min in experiments:
        # Only mode
        if args.only and config_name != args.only:
            continue
        # Start-from logic
        if not start_found:
            if config_name == args.start_from:
                start_found = True
            else:
                print(f"[SKIP] {config_name} (before start_from)")
                continue
        # Skip baselines
        if args.skip_baselines and config_name in ("baseline_rgb", "baseline_thermal", "baseline_audio"):
            print(f"[SKIP] {config_name} (--skip_baselines)")
            continue

        print(f"\n> {desc} (~{expected_min} min)")
        results = run_experiment(
            config_name, args.data_dir, args.output_dir,
            args.dry_run, args.dry_run_pipeline, args.force
        )
        all_results[config_name] = results

    # ── Print summary table ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'Disaster F1':<15} {'Victim F1':<12} {'Combined F1':<12}")
    print("-" * 80)
    for name, res in all_results.items():
        if "test" in res:
            t = res["test"]
            print(f"{name:<30} {t.get('disaster_f1', 0):<15.4f} {t.get('victim_f1', 0):<12.4f} {t.get('combined_f1', 0):<12.4f}")
        else:
            print(f"{name:<30} {'N/A':<15} {'N/A':<12} {'N/A':<12}")

    # Save summary
    summary_name = ("public_benchmarks_summary.json" if args.public_benchmarks
                    else "unimodal_backbones_summary.json" if args.unimodal_backbones
                    else "all_results_summary.json")
    summary_path = Path(args.output_dir) / "logs" / summary_name
    Path(args.output_dir, "logs").mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] Summary: {summary_path}")


if __name__ == "__main__":
    main()
