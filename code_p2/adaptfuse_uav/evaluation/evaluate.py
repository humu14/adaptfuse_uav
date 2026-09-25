"""
AdapFuse-UAV: Full evaluation script.
Runs standard metrics, corruption robustness, and missing-modality evaluation.

Usage:
    python evaluation/evaluate.py --config configs/adaptfuse_v1.yaml --checkpoint outputs/checkpoints/adaptfuse_v1_best.pth
    python evaluation/evaluate.py --config configs/adaptfuse_v1.yaml --all_models
"""

import sys
import os
import json
import argparse
import time
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.multimodal_dataset import MultimodalDisasterDataset, build_dataloaders
from datasets.corruptions import CORRUPTION_PARAMS
from models.full_model import build_model
from evaluation.metrics import (
    compute_metrics, format_metrics_table, print_classification_report,
    expected_calibration_error, false_positive_rate,
)


DISASTER_CLASS_NAMES = ["normal", "fire/smoke", "collapse/flood", "other_disaster"]
VICTIM_CLASS_NAMES = ["no_person", "person"]


def load_model(config: dict, ckpt_path: str, device: torch.device):
    model = build_model(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Standard Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_standard(model, loader, device, use_amp=True) -> dict:
    """Full evaluation: accuracy, F1, AUROC for disaster and victim tasks."""
    all_d_preds, all_d_gt, all_d_proba = [], [], []
    all_v_preds, all_v_gt = [], []
    all_reliability = []

    for batch in tqdm(loader, desc="Standard Eval", leave=False):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast(enabled=use_amp):
            d_out, v_out, reliability = model(
                rgb=batch["rgb"], thermal=batch["thermal"], audio=batch["audio"],
                has_rgb=batch["has_rgb"], has_thermal=batch["has_thermal"],
                has_audio=batch["has_audio"],
            )

        all_d_preds.extend(d_out.argmax(1).cpu().numpy())
        all_d_gt.extend(batch["disaster_label"].cpu().numpy())
        all_d_proba.extend(F.softmax(d_out, dim=-1).cpu().numpy())
        all_v_preds.extend(v_out.argmax(1).cpu().numpy())
        all_v_gt.extend(batch["victim_label"].cpu().numpy())
        all_reliability.append(reliability.cpu().numpy())

    disaster_metrics = compute_metrics(all_d_gt, all_d_preds, all_d_proba, num_classes=4, prefix="disaster_")
    victim_metrics = compute_metrics(all_v_gt, all_v_preds, num_classes=2, prefix="victim_")

    avg_reliability = np.concatenate(all_reliability, axis=0).mean(axis=0)

    # EXP-5: False Positive Rate (clean samples misclassified as disaster)
    fpr = false_positive_rate(np.array(all_d_gt), np.array(all_d_preds), num_classes=4)

    # EXP-9: Expected Calibration Error
    ece = expected_calibration_error(np.array(all_d_proba), np.array(all_d_gt))

    print(format_metrics_table(disaster_metrics, "Disaster Task"))
    print(format_metrics_table(victim_metrics, "Victim Task"))
    print(f"\n  Avg Reliability: RGB={avg_reliability[0]:.3f} | "
          f"Thermal={avg_reliability[1]:.3f} | Audio={avg_reliability[2]:.3f}")
    print(f"  False Positive Rate (clean->disaster): {fpr:.4f}")
    print(f"  Expected Calibration Error (ECE):     {ece:.4f}")

    print("\n  Disaster Classification Report:")
    print_classification_report(all_d_gt, all_d_preds, DISASTER_CLASS_NAMES)

    return {
        **disaster_metrics,
        **victim_metrics,
        "avg_reliability": avg_reliability.tolist(),
        "false_positive_rate": fpr,
        "ece": ece,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Corruption Robustness Evaluation (Section 7.2 of plan)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_corruption_robustness(model, test_csv: str, device, use_amp=True) -> dict:
    """
    Evaluate model under all corruption types × 5 severity levels.
    Returns 25-point robustness matrix.
    """
    results = {}
    corruptions = {
        'smoke_overlay': 'rgb',
        'motion_blur': 'rgb',
        'low_light': 'rgb',
        'thermal_drift': 'thermal',
        'rotor_noise_snr': 'audio',
    }

    print("\n[EVAL] Corruption Robustness (5 corruptions × 5 severities)")

    for corruption_name, modality in corruptions.items():
        results[corruption_name] = {}
        for severity in range(1, 6):
            ds = MultimodalDisasterDataset(
                csv_path=test_csv,
                split="test",
                modalities=["rgb", "thermal", "audio"],
                corruption_prob=1.0,     # always apply
                corruption_type=corruption_name,
                corruption_severity=severity,
            )
            loader = torch.utils.data.DataLoader(
                ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
            )
            all_d_preds, all_d_gt = [], []
            for batch in tqdm(loader, desc=f"  {corruption_name} sev={severity}", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with autocast(enabled=use_amp):
                    d_out, _, _ = model(
                        rgb=batch["rgb"], thermal=batch["thermal"], audio=batch["audio"],
                        has_rgb=batch["has_rgb"], has_thermal=batch["has_thermal"],
                        has_audio=batch["has_audio"],
                    )
                all_d_preds.extend(d_out.argmax(1).cpu().numpy())
                all_d_gt.extend(batch["disaster_label"].cpu().numpy())

            from sklearn.metrics import f1_score
            f1 = f1_score(all_d_gt, all_d_preds, average="macro", zero_division=0)
            results[corruption_name][severity] = f1
            print(f"  {corruption_name} sev={severity}: F1={f1:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Missing Modality Evaluation (Section 7.3 of plan)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_missing_modality(model, test_csv: str, device, use_amp=True) -> dict:
    """
    Evaluate model under 6 missing-modality configurations.
    """
    missing_configs = [
        {"name": "full",           "has_rgb": 1, "has_thermal": 1, "has_audio": 1},
        {"name": "no_rgb",         "has_rgb": 0, "has_thermal": 1, "has_audio": 1},
        {"name": "no_thermal",     "has_rgb": 1, "has_thermal": 0, "has_audio": 1},
        {"name": "no_audio",       "has_rgb": 1, "has_thermal": 1, "has_audio": 0},
        {"name": "audio_only",     "has_rgb": 0, "has_thermal": 0, "has_audio": 1},
        {"name": "rgb_only",       "has_rgb": 1, "has_thermal": 0, "has_audio": 0},
        {"name": "rgb_thermal",    "has_rgb": 1, "has_thermal": 1, "has_audio": 0},
    ]

    print("\n[EVAL] Missing Modality Robustness (7 configurations)")
    results = {}
    ds_base = MultimodalDisasterDataset(
        csv_path=test_csv,
        split="test",
        modalities=["rgb", "thermal", "audio"],
    )
    loader = torch.utils.data.DataLoader(
        ds_base, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    for cfg in missing_configs:
        name = cfg["name"]
        all_d_preds, all_d_gt = [], []

        for batch in tqdm(loader, desc=f"  {name}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Override availability flags
            B = batch["rgb"].size(0)
            has_rgb = torch.full((B,), float(cfg["has_rgb"]), device=device)
            has_thermal = torch.full((B,), float(cfg["has_thermal"]), device=device)
            has_audio = torch.full((B,), float(cfg["has_audio"]), device=device)

            # Zero out missing tensors
            rgb = batch["rgb"] * has_rgb.view(-1, 1, 1, 1)
            thermal = batch["thermal"] * has_thermal.view(-1, 1, 1, 1)
            audio = batch["audio"] * has_audio.view(-1, 1, 1, 1)

            with autocast(enabled=use_amp):
                d_out, _, _ = model(
                    rgb=rgb, thermal=thermal, audio=audio,
                    has_rgb=has_rgb, has_thermal=has_thermal, has_audio=has_audio,
                )
            all_d_preds.extend(d_out.argmax(1).cpu().numpy())
            all_d_gt.extend(batch["disaster_label"].cpu().numpy())

        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(all_d_gt, all_d_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_d_gt, all_d_preds)
        results[name] = {"f1_macro": f1, "accuracy": acc}
        print(f"  {name:<20}: F1={f1:.4f}  Acc={acc:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Latency & Parameter Efficiency (Section 7.4 of plan)
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(model, device, n_runs=100, batch_size=1) -> dict:
    """Measure inference latency and count parameters."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy inputs
    dummy = {
        "rgb": torch.randn(batch_size, 3, 224, 224, device=device),
        "thermal": torch.randn(batch_size, 1, 224, 224, device=device),
        "audio": torch.randn(batch_size, 1, 64, 63, device=device),
        "has_rgb": torch.ones(batch_size, device=device),
        "has_thermal": torch.ones(batch_size, device=device),
        "has_audio": torch.ones(batch_size, device=device),
    }

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(**dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(**dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

    print(f"\n[LATENCY] Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"          Latency (batch=1): {elapsed_ms:.2f} ms")
    print(f"          Throughput: {1000/elapsed_ms:.1f} fps")

    return {
        "params": n_params,
        "params_M": n_params / 1e6,
        "latency_ms": elapsed_ms,
        "fps": 1000 / elapsed_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AdapFuse-UAV Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=str(ROOT))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--skip_corruption", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--skip_latency", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    exp_name = config.get("experiment_name", "experiment")

    # Auto-find checkpoint if not specified
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = str(Path(args.output_dir) / "checkpoints" / f"{exp_name}_best.pth")
    if not Path(ckpt_path).exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    print(f"\n[EVAL] Loading model: {config['model']} from {ckpt_path}")
    model = load_model(config, ckpt_path, device)

    test_csv = str(Path(args.data_dir) / "data" / "metadata" / "test.csv")
    _, _, test_loader = build_dataloaders(args.data_dir, config)

    all_results = {"experiment": exp_name, "config": config}

    # ── Standard eval ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STANDARD EVALUATION")
    std_results = evaluate_standard(model, test_loader, device, use_amp)
    all_results["standard"] = std_results

    # ── Latency ───────────────────────────────────────────────────
    if not args.skip_latency:
        print(f"\n{'='*60}")
        print("LATENCY & EFFICIENCY")
        lat_results = measure_latency(model, device)
        all_results["latency"] = lat_results

    # ── Corruption robustness ─────────────────────────────────────
    if not args.skip_corruption and Path(test_csv).exists():
        print(f"\n{'='*60}")
        corr_results = evaluate_corruption_robustness(model, test_csv, device, use_amp)
        all_results["corruption"] = corr_results

    # ── Missing modality ──────────────────────────────────────────
    if not args.skip_missing and Path(test_csv).exists():
        print(f"\n{'='*60}")
        miss_results = evaluate_missing_modality(model, test_csv, device, use_amp)
        all_results["missing_modality"] = miss_results

    # ── Save ──────────────────────────────────────────────────────
    out_path = Path(args.output_dir) / "logs" / f"{exp_name}_full_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy to Python types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\n[SAVED] Full eval results: {out_path}")


if __name__ == "__main__":
    main()
