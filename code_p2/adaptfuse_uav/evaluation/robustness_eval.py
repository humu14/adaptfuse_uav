"""
NOW-4: Corruption + Missing Modality Robustness Evaluation (inference only, no training).

Sweeps all 5 corruption types × 5 severity levels and all 7 missing-modality combos.
Generates Tables 3 & 4 for Chapter 4 of the thesis.

Usage:
    python evaluation/robustness_eval.py \
        --config configs/adaptfuse_v1.yaml \
        --checkpoint outputs/checkpoints/adaptfuse_v1_best.pth

    # Compare multiple models:
    python evaluation/robustness_eval.py \
        --config configs/adaptfuse_v1.yaml \
        --checkpoint outputs/checkpoints/adaptfuse_v1_best.pth \
        --compare configs/late_fusion.yaml outputs/checkpoints/late_fusion_best.pth

Output: outputs/logs/<exp>_robustness.json + printed tables.
"""

import sys
import json
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.multimodal_dataset import MultimodalDisasterDataset
from models.full_model import build_model


SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS = [
    ('smoke_overlay',   'rgb'),
    ('motion_blur',     'rgb'),
    ('low_light',       'rgb'),
    ('thermal_drift',   'thermal'),
    ('rotor_noise_snr', 'audio'),
]

MISSING_COMBOS = [
    {"name": "full",          "has_rgb": 1, "has_thermal": 1, "has_audio": 1},
    {"name": "no_rgb",        "has_rgb": 0, "has_thermal": 1, "has_audio": 1},
    {"name": "no_thermal",    "has_rgb": 1, "has_thermal": 0, "has_audio": 1},
    {"name": "no_audio",      "has_rgb": 1, "has_thermal": 1, "has_audio": 0},
    {"name": "rgb_only",      "has_rgb": 1, "has_thermal": 0, "has_audio": 0},
    {"name": "thermal_only",  "has_rgb": 0, "has_thermal": 1, "has_audio": 0},
    {"name": "audio_only",    "has_rgb": 0, "has_thermal": 0, "has_audio": 1},
]


def load_model(config: dict, ckpt_path: str, device: torch.device):
    model = build_model(config)
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path} — using random weights")
    model.to(device).eval()
    return model


@torch.no_grad()
def eval_loader(model, loader, device) -> tuple:
    preds, gts = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        d_out, _, _ = model(
            batch["rgb"], batch["thermal"], batch["audio"],
            batch["has_rgb"], batch["has_thermal"], batch["has_audio"],
        )
        preds.extend(d_out.argmax(1).cpu().numpy())
        gts.extend(batch["disaster_label"].cpu().numpy())
    return np.array(gts), np.array(preds)


@torch.no_grad()
def eval_missing(model, loader, device, cfg: dict) -> tuple:
    B_flag = cfg["has_rgb"], cfg["has_thermal"], cfg["has_audio"]
    preds, gts = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B = batch["rgb"].size(0)
        h_rgb = torch.full((B,), float(B_flag[0]), device=device)
        h_th  = torch.full((B,), float(B_flag[1]), device=device)
        h_au  = torch.full((B,), float(B_flag[2]), device=device)
        rgb     = batch["rgb"]     * h_rgb.view(-1, 1, 1, 1)
        thermal = batch["thermal"] * h_th.view(-1, 1, 1, 1)
        audio   = batch["audio"]   * h_au.view(-1, 1, 1, 1)
        d_out, _, _ = model(rgb, thermal, audio, h_rgb, h_th, h_au)
        preds.extend(d_out.argmax(1).cpu().numpy())
        gts.extend(batch["disaster_label"].cpu().numpy())
    return np.array(gts), np.array(preds)


def run_corruption_sweep(model, test_csv: str, device: torch.device) -> dict:
    results = {}
    print("\n[TABLE 3] Corruption Robustness (5 corruptions × 5 severities)")
    for corruption_name, modality in CORRUPTIONS:
        results[corruption_name] = {}
        for severity in SEVERITIES:
            ds = MultimodalDisasterDataset(
                csv_path=test_csv,
                split="test",
                modalities=["rgb", "thermal", "audio"],
                corruption_prob=1.0,
                corruption_type=corruption_name,
                corruption_severity=severity,
            )
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
            gts, preds = eval_loader(model, loader, device)
            f1 = f1_score(gts, preds, average="macro", zero_division=0)
            results[corruption_name][severity] = round(float(f1), 4)
    return results


def run_missing_modality_sweep(model, test_csv: str, device: torch.device) -> dict:
    results = {}
    print("\n[TABLE 4] Missing Modality Robustness (7 combos)")
    ds_base = MultimodalDisasterDataset(
        csv_path=test_csv,
        split="test",
        modalities=["rgb", "thermal", "audio"],
    )
    loader = DataLoader(ds_base, batch_size=32, shuffle=False, num_workers=0)
    for cfg in MISSING_COMBOS:
        gts, preds = eval_missing(model, loader, device, cfg)
        f1 = f1_score(gts, preds, average="macro", zero_division=0)
        acc = accuracy_score(gts, preds)
        results[cfg["name"]] = {"f1_macro": round(float(f1), 4), "accuracy": round(float(acc), 4)}
    return results


def print_corruption_table(results: dict):
    corruptions = list(results.keys())
    print(f"\n{'='*70}")
    print("Table 3: Corruption Robustness (macro F1)")
    print(f"{'Corruption':<22}" + "".join(f" sev{s:>2}" for s in SEVERITIES))
    print("-" * 70)
    for c in corruptions:
        row = f"  {c:<20}"
        for s in SEVERITIES:
            row += f"  {results[c].get(s, 0.0):.3f}"
        print(row)
    print("=" * 70)


def print_missing_table(results: dict):
    print(f"\n{'='*50}")
    print("Table 4: Missing Modality Robustness")
    print(f"  {'Config':<20} {'F1 macro':>9} {'Accuracy':>9}")
    print("-" * 45)
    for name, vals in results.items():
        print(f"  {name:<20} {vals['f1_macro']:>9.4f} {vals['accuracy']:>9.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="NOW-4: Corruption + missing-modality robustness sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=str(ROOT))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--skip_corruption", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    # Optional second model for comparison
    parser.add_argument("--compare", nargs=2, metavar=("CONFIG2", "CKPT2"),
                        help="Compare with a second model: --compare config.yaml ckpt.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = config.get("experiment_name", "experiment")

    ckpt_path = args.checkpoint or str(
        Path(args.output_dir) / "checkpoints" / f"{exp_name}_best.pth"
    )

    test_csv = str(Path(args.data_dir) / "data" / "metadata" / "test.csv")
    if not Path(test_csv).exists():
        print(f"[ERROR] test.csv not found: {test_csv}")
        return

    print(f"[NOW-4] Loading model: {config['model']}")
    model = load_model(config, ckpt_path, device)

    all_results = {"experiment": exp_name}

    if not args.skip_corruption:
        corr = run_corruption_sweep(model, test_csv, device)
        all_results["corruption"] = corr
        print_corruption_table(corr)

    if not args.skip_missing:
        miss = run_missing_modality_sweep(model, test_csv, device)
        all_results["missing_modality"] = miss
        print_missing_table(miss)

    if args.compare:
        config2_path, ckpt2_path = args.compare
        with open(config2_path) as f:
            config2 = yaml.safe_load(f)
        exp2 = config2.get("experiment_name", "compare")
        print(f"\n[NOW-4] Loading comparison model: {config2['model']}")
        model2 = load_model(config2, ckpt2_path, device)
        all_results[f"compare_{exp2}"] = {}
        if not args.skip_corruption:
            c2 = run_corruption_sweep(model2, test_csv, device)
            all_results[f"compare_{exp2}"]["corruption"] = c2
            print(f"\n--- Comparison: {exp2} ---")
            print_corruption_table(c2)
        if not args.skip_missing:
            m2 = run_missing_modality_sweep(model2, test_csv, device)
            all_results[f"compare_{exp2}"]["missing_modality"] = m2
            print_missing_table(m2)

    out_path = Path(args.output_dir) / "logs" / f"{exp_name}_robustness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
