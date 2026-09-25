"""
AdapFuse-UAV: Main training entry point.
Run all 8 experimental configurations sequentially or individually.
Usage:
    python training/train.py --config configs/adaptfuse_v1.yaml
    python training/train.py --config configs/baseline_rgb.yaml
    python training/run_all.py   ← run all experiments in order
"""

import sys
import os
import argparse
import json
import random
import time

PROCESS_STARTED_UNIX = time.time()

import yaml
import numpy as np
import torch
from pathlib import Path

# Prevent an unnecessary network version check during reproducible/offline runs.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# ── Ensure project root is in path ───────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.multimodal_dataset import build_dataloaders
from models.full_model import build_model
from training.trainer import Trainer
from training.run_artifacts import (
    finalize_run_artifacts,
    initialize_run_artifacts,
    refresh_resolved_config,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def main():
    run_started_unix = PROCESS_STARTED_UNIX
    parser = argparse.ArgumentParser(description="AdapFuse-UAV Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(ROOT),
        help="Project root (contains data/metadata/)",
    )
    parser.add_argument(
        "--metadata_dir", type=str, default=None,
        help="Optional metadata directory containing train.csv, val.csv, and test.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--teacher_ckpt", type=str, default=None,
        help="Override teacher checkpoint path for KD",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the training seed",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Override DataLoader worker count (use 0 for deterministic Windows benchmarking)",
    )
    parser.add_argument(
        "--audio_cache_dir", type=str, default=None,
        help="Override the directory containing cached log-mel features",
    )
    parser.add_argument(
        "--dry_run_batches", type=int, default=None,
        help="Limit number of batches per epoch for dry run",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    config = load_config(args.config)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.audio_cache_dir is not None:
        config["audio_cache_dir"] = args.audio_cache_dir
    if args.metadata_dir:
        config["metadata_dir"] = args.metadata_dir
    if args.teacher_ckpt:
        config["kd_teacher_ckpt"] = args.teacher_ckpt
    if args.dry_run_batches:
        config["dry_run_batches"] = args.dry_run_batches

    # ── Seed ──────────────────────────────────────────────────────
    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    # ── Device ────────────────────────────────────────────────────
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Output dir ────────────────────────────────────────────────
    output_dir = args.output_dir or str(ROOT / config.get("output_dir", "outputs"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_context = initialize_run_artifacts(
        output_dir=Path(output_dir),
        root=ROOT,
        config=config,
        config_path=Path(args.config).resolve(),
        data_dir=args.data_dir,
        device=device,
        started_unix=run_started_unix,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[ARTIFACTS] Run provenance initialized: {run_context['artifact_dir']}")

    # ── Experiment name and automatic resume if available ──────────
    exp_name = config.get("experiment_name", Path(args.config).stem)
    # If user did not pass --resume, auto-resume from latest (most recent epoch) or best checkpoint
    if not args.resume:
        candidate_latest = Path(output_dir) / "checkpoints" / f"{exp_name}_latest.pth"
        candidate_best = Path(output_dir) / "checkpoints" / f"{exp_name}_best.pth"
        if candidate_latest.exists():
            args.resume = str(candidate_latest)
            print(f"[AUTO-RESUME] Found latest checkpoint for {exp_name}: {args.resume}. Resuming from most recent epoch.")
        elif candidate_best.exists():
            args.resume = str(candidate_best)
            print(f"[AUTO-RESUME] Found best checkpoint for {exp_name}: {args.resume}. Resuming from it.")

    # ── Data ──────────────────────────────────────────────────────
    print(f"\n[DATA] Loading from: {args.data_dir}")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        config=config,
    )
    print(
        f"  Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n[MODEL] Building: {config['model']}")
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # ── Teacher (for KD) ──────────────────────────────────────────
    teacher = None
    if config.get("use_kd", False):
        teacher_ckpt = config.get("kd_teacher_ckpt")
        if teacher_ckpt:
            tp = Path(teacher_ckpt)
            if not tp.is_absolute():
                teacher_ckpt = str(ROOT / tp)
        if teacher_ckpt and Path(teacher_ckpt).exists():
            print(f"[KD] Loading teacher from: {teacher_ckpt}")
            teacher_config = dict(config)
            teacher_config["model"] = "adaptfuse_v1"
            teacher_config["width_multiplier"] = 1.0
            teacher = build_model(teacher_config)
            ckpt = torch.load(teacher_ckpt, map_location="cpu")
            teacher.load_state_dict(ckpt["model_state"])
            teacher.eval()
            print(f"  Teacher loaded (epoch {ckpt.get('epoch', '?')})")
        else:
            print(f"[WARN] KD enabled but teacher checkpoint not found: {teacher_ckpt}")
            print("       Training without KD.")
            config["use_kd"] = False

    # Capture runtime fallbacks (for example, unavailable KD teacher) in the exact config.
    refresh_resolved_config(run_context, config)

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        output_dir=output_dir,
        teacher=teacher,
        device=device,
    )

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        print(f"[RESUME] Loading checkpoint: {args.resume}")
        start_epoch, metrics = trainer.load_checkpoint(args.resume)
        start_epoch += 1
        print(f"  Resumed from epoch {start_epoch - 1}")

    # ── Train ─────────────────────────────────────────────────────
    history = trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    if history:
        final_epoch = int(history[-1]["epoch"])
        final_metrics = {
            key.removeprefix("val_"): value
            for key, value in history[-1].items()
            if key.startswith("val_")
        }
    else:
        final_epoch = max(start_epoch - 1, 0)
        final_metrics = {}
    trainer.save_final_checkpoint(final_epoch, final_metrics)
    expected_best = Path(output_dir) / "checkpoints" / f"{exp_name}_best.pth"
    if not expected_best.exists():
        print("[WARN] No best checkpoint was present; saving the final state as best available.")
        trainer.save_checkpoint(final_epoch, final_metrics, is_best=True)

    # ── Final Test Evaluation ─────────────────────────────────────
    print("\n[EVAL] Final test evaluation...")
    exp_name = config.get("experiment_name", "experiment")
    best_ckpt = Path(output_dir) / "checkpoints" / f"{exp_name}_best.pth"

    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))
        print(f"  Loaded best checkpoint: {best_ckpt.name}")

    test_metrics, test_predictions = trainer.eval_epoch(
        test_loader,
        epoch=0,
        split="test",
        return_predictions=True,
    )
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {exp_name}")
    print(f"  Disaster F1 (macro): {test_metrics['disaster_f1']:.4f}")
    print(f"  Victim   F1 (macro): {test_metrics['victim_f1']:.4f}")
    print(f"  Combined F1:         {test_metrics['combined_f1']:.4f}")
    print(f"  Disaster Acc:        {test_metrics['disaster_acc']:.4f}")
    print(f"  Victim   Acc:        {test_metrics['victim_acc']:.4f}")
    print(f"  Det mAP@0.5:         {test_metrics.get('map50', 0.0):.4f}")
    print(f"{'='*60}\n")

    # Save test metrics
    test_path = Path(output_dir) / "logs" / f"{exp_name}_test_results.json"
    with open(test_path, "w") as f:
        json.dump({"experiment": exp_name, "test": test_metrics}, f, indent=2)
    print(f"[SAVED] Test results: {test_path}")

    run_summary = finalize_run_artifacts(
        context=run_context,
        predictions=test_predictions,
        test_metrics=test_metrics,
        output_dir=Path(output_dir),
        exp_name=exp_name,
        device=device,
    )
    print(f"[SAVED] Reproducibility bundle: {run_context['artifact_dir']}")
    print(
        f"        Wall time: {run_summary['wall_clock_seconds'] / 3600:.2f} h | "
        f"Peak VRAM: {run_summary['peak_vram_bytes'] / (1024 ** 3):.2f} GiB"
    )


if __name__ == "__main__":
    main()
