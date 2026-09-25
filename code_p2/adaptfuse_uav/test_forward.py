"""
AdapFuse-UAV unit tests and model verification.
Run this FIRST before any training to verify everything works.

Usage:
    python test_forward.py
    python test_forward.py --device cuda
"""

import sys
import os
import argparse
import torch
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def make_dummy_batch(B=4, device='cuda'):
    """Create a dummy batch with missing modalities."""
    return {
        "rgb": torch.randn(B, 3, 224, 224, device=device),
        "thermal": torch.randn(B, 1, 224, 224, device=device),
        "audio": torch.randn(B, 1, 64, 63, device=device),
        "has_rgb": torch.tensor([1., 1., 0., 0.], device=device),
        "has_thermal": torch.tensor([1., 0., 1., 0.], device=device),
        "has_audio": torch.tensor([1., 1., 1., 0.], device=device),
        "disaster_label": torch.randint(0, 4, (B,), device=device),
        "victim_label": torch.randint(0, 2, (B,), device=device),
    }


def test_model(model_name, config_overrides, device, B=4):
    from models.full_model import build_model
    config = {
        "model": model_name,
        "num_classes_disaster": 4,
        "num_classes_victim": 2,
        **config_overrides,
    }
    print(f"  Testing {model_name}...", end=" ", flush=True)
    try:
        model = build_model(config).to(device)
        model.eval()
        batch = make_dummy_batch(B, device)

        with torch.no_grad():
            d_out, v_out, reliability = model(
                rgb=batch["rgb"],
                thermal=batch["thermal"],
                audio=batch["audio"],
                has_rgb=batch["has_rgb"],
                has_thermal=batch["has_thermal"],
                has_audio=batch["has_audio"],
            )

        assert d_out.shape == (B, 4), f"Disaster shape: {d_out.shape}"
        assert v_out.shape == (B, 2), f"Victim shape: {v_out.shape}"
        assert reliability.shape == (B, 3), f"Reliability shape: {reliability.shape}"

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[PASS] ({n_params/1e6:.2f}M params)")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_losses(device):
    from training.losses import CombinedLoss
    print("  Testing CombinedLoss...", end=" ", flush=True)
    try:
        B, C_d, C_v = 4, 4, 2
        criterion = CombinedLoss(use_kd=False)
        d_out = torch.randn(B, C_d, device=device, requires_grad=True)
        v_out = torch.randn(B, C_v, device=device, requires_grad=True)
        d_gt = torch.randint(0, C_d, (B,), device=device)
        v_gt = torch.randint(0, C_v, (B,), device=device)
        reliability = torch.rand(B, 3, device=device)

        losses = criterion(d_out, v_out, d_gt, v_gt, reliability)
        assert 'total' in losses
        assert losses['total'].item() > 0
        losses['total'].backward()
        print(f"[PASS] (loss={losses['total'].item():.4f})")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_kd_losses(device):
    from training.losses import CombinedLoss
    print("  Testing KD Loss...", end=" ", flush=True)
    try:
        B, C_d = 4, 4
        criterion = CombinedLoss(use_kd=True, kd_temperature=4.0)
        d_out = torch.randn(B, C_d, device=device, requires_grad=True)
        v_out = torch.randn(B, 2, device=device, requires_grad=True)
        d_gt = torch.randint(0, C_d, (B,), device=device)
        v_gt = torch.randint(0, 2, (B,), device=device)
        t_disaster = torch.randn(B, C_d, device=device)
        teacher_feat = torch.randn(B, 192, device=device)
        student_feat = torch.randn(B, 192, device=device, requires_grad=True)

        losses = criterion(
            d_out, v_out, d_gt, v_gt,
            teacher_disaster=t_disaster,
            fused_feat=student_feat,
            teacher_feat=teacher_feat,
        )
        losses['total'].backward()
        print(f"[PASS] (kd_logit={losses['kd_logit'].item():.4f})")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_dataset_loading(device):
    from datasets.multimodal_dataset import MultimodalDisasterDataset
    print("  Testing dataset loading...", end=" ", flush=True)
    meta_dir = ROOT / "data" / "metadata"
    train_csv = meta_dir / "train.csv"
    if not train_csv.exists():
        print("[SKIP] (no train.csv — run build_metadata.py first)")
        return None

    try:
        ds = MultimodalDisasterDataset(
            csv_path=str(train_csv),
            split="train",
            modalities=["rgb", "thermal", "audio"],
            corruption_prob=0.3,
            missing_modality_prob=0.2,
        )
        sample = ds[0]
        assert "rgb" in sample
        assert "disaster_label" in sample
        assert sample["rgb"].shape == (3, 224, 224)
        print(f"[PASS] ({len(ds)} samples, keys: {list(sample.keys())})")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_step(device):
    from models.full_model import build_model
    from training.losses import CombinedLoss
    print("  Testing full training step (AdapFuse v1)...", end=" ", flush=True)
    try:
        config = {
            "model": "adaptfuse_v1",
            "num_classes_disaster": 4,
            "num_classes_victim": 2,
            "proj_dim": 192,
            "num_heads": 4,
        }
        model = build_model(config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = CombinedLoss()

        batch = make_dummy_batch(4, device)
        optimizer.zero_grad()

        d_out, v_out, reliability = model(
            rgb=batch["rgb"], thermal=batch["thermal"], audio=batch["audio"],
            has_rgb=batch["has_rgb"], has_thermal=batch["has_thermal"],
            has_audio=batch["has_audio"],
        )
        losses = criterion(d_out, v_out, batch["disaster_label"], batch["victim_label"], reliability)
        losses["total"].backward()
        optimizer.step()

        print(f"[PASS] (loss={losses['total'].item():.4f})")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    print("=" * 60)
    print("AdapFuse-UAV: Forward Pass Tests")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    results = {}

    print("\n[1] Model Forward Passes:")
    models_to_test = [
        ("baseline_rgb",       {}),
        ("baseline_thermal",   {}),
        ("baseline_audio",     {}),
        ("early_fusion",       {}),
        ("late_fusion",        {}),
        ("intermediate_fixed", {"proj_dim": 192}),
        ("adaptfuse_v1",       {"proj_dim": 192, "num_heads": 4}),
        ("adaptfuse_v2_kd",    {"proj_dim": 192, "num_heads": 4, "width_multiplier": 0.5}),
    ]
    for name, overrides in models_to_test:
        results[name] = test_model(name, overrides, device)

    print("\n[2] Loss Function Tests:")
    results["loss_basic"] = test_losses(device)
    results["loss_kd"]    = test_kd_losses(device)

    print("\n[3] Dataset Loading:")
    results["dataset"] = test_dataset_loading(device)

    print("\n[4] Full Training Step:")
    results["training_step"] = test_full_training_step(device)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed  = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed  = sum(1 for v in results.values() if v is False)
    for name, res in results.items():
        status = "[PASS]" if res is True else ("[SKIP]" if res is None else "[FAIL]")
        print(f"  {status}  {name}")
    print(f"\n  Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    print("=" * 60)

    if failed > 0:
        print("\n[!] Some tests failed. Fix errors before training.")
        sys.exit(1)
    else:
        print("\n[OK] All tests passed! Ready to train.")


if __name__ == "__main__":
    main()
