"""
EXP-2: Reliability Score Behavior Analysis (eval-only, no training needed).

Verifies that RUE actually learns to downweight corrupted sensors:
when smoke_overlay is applied to RGB, r_rgb should drop while r_thermal/r_audio stay stable.

Usage:
    python evaluation/reliability_analysis.py \
        --config configs/adaptfuse_v1.yaml \
        --checkpoint outputs/checkpoints/adaptfuse_v1_best.pth

Output: outputs/reliability_behavior.json + printed table.
"""

import sys
import json
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.multimodal_dataset import MultimodalDisasterDataset
from models.full_model import build_model

CORRUPTIONS = {
    'rgb':     [('smoke_overlay', 0), ('smoke_overlay', 1), ('smoke_overlay', 3), ('smoke_overlay', 5)],
    'thermal': [('thermal_drift', 0), ('thermal_drift', 1), ('thermal_drift', 3), ('thermal_drift', 5)],
    'audio':   [('rotor_noise_snr', 0), ('rotor_noise_snr', 1), ('rotor_noise_snr', 3), ('rotor_noise_snr', 5)],
}

MODALITY_NAMES = ['r_rgb', 'r_thermal', 'r_audio']


@torch.no_grad()
def analyze_reliability(model, test_csv: str, device: torch.device) -> dict:
    results = {}
    for target_modality, corruption_list in CORRUPTIONS.items():
        results[target_modality] = []
        for corruption_type, severity in corruption_list:
            ds = MultimodalDisasterDataset(
                csv_path=test_csv,
                split='test',
                modalities=['rgb', 'thermal', 'audio'],
                corruption_prob=1.0 if severity > 0 else 0.0,
                corruption_type=corruption_type,
                corruption_severity=severity,
            )
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
            all_rel = []
            for batch in tqdm(loader, desc=f"  {target_modality} | {corruption_type} sev={severity}", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                _, _, reliability = model(
                    batch['rgb'], batch['thermal'], batch['audio'],
                    batch['has_rgb'], batch['has_thermal'], batch['has_audio'],
                )
                all_rel.append(reliability.cpu())
            mean_rel = torch.cat(all_rel).mean(0)  # (3,)
            results[target_modality].append({
                'severity': severity,
                'corruption': corruption_type,
                'r_rgb':     round(mean_rel[0].item(), 4),
                'r_thermal': round(mean_rel[1].item(), 4),
                'r_audio':   round(mean_rel[2].item(), 4),
            })

    return results


def print_results(results: dict):
    header = f"{'Target':<10} {'Sev':>4}  {'r_rgb':>8} {'r_thermal':>10} {'r_audio':>8}  Note"
    sep = '-' * 65
    print(f"\n{'='*65}")
    print("RUE Reliability Score Behavior (EXP-2)")
    print(f"{'='*65}")
    for target, rows in results.items():
        print(f"\n  Corrupted modality: {target.upper()}")
        print(header)
        print(sep)
        for r in rows:
            note = ""
            if r['severity'] > 0:
                if target == 'rgb' and r['r_rgb'] < rows[0]['r_rgb'] - 0.02:
                    note = "<-- drops (EXPECTED)"
                elif target == 'thermal' and r['r_thermal'] < rows[0]['r_thermal'] - 0.02:
                    note = "<-- drops (EXPECTED)"
                elif target == 'audio' and r['r_audio'] < rows[0]['r_audio'] - 0.02:
                    note = "<-- drops (EXPECTED)"
            print(f"  {target:<10} {r['severity']:>4}  {r['r_rgb']:>8.4f} {r['r_thermal']:>10.4f} {r['r_audio']:>8.4f}  {note}")


def main():
    parser = argparse.ArgumentParser(description="EXP-2: Reliability score behavior analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=str(ROOT))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = config.get("experiment_name", "experiment")

    ckpt_path = args.checkpoint or str(
        Path(args.output_dir) / "checkpoints" / f"{exp_name}_best.pth"
    )
    if not Path(ckpt_path).exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    print(f"[EXP-2] Loading {config['model']} from {ckpt_path}")
    model = build_model(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device).eval()

    test_csv = str(Path(args.data_dir) / "data" / "metadata" / "test.csv")
    if not Path(test_csv).exists():
        print(f"[ERROR] test.csv not found: {test_csv}")
        return

    results = analyze_reliability(model, test_csv, device)
    print_results(results)

    out_path = Path(args.output_dir) / "logs" / f"{exp_name}_reliability_behavior.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
