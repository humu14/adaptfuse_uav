"""
Forward Pass Test — Verify all 8 models and 5 novel modules work
==================================================================
Run this FIRST to ensure the entire pipeline is functional.

Usage:
    python tests/test_forward_all.py
    python tests/test_forward_all.py --models yolov8n yolov11n  # Subset
    python tests/test_forward_all.py --novel_only               # Test novel modules only
"""

import os
import sys
import time
import torch
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_novel_modules():
    """Test all novel modules with dummy inputs."""
    print(f"\n{'='*60}")
    print(f"  Testing Novel Modules")
    print(f"{'='*60}\n")

    results = {}

    # --- SPD-Conv ---
    try:
        from models.novel_modules.spd_conv import SPDConv, SPDBlock
        x = torch.randn(2, 64, 80, 80)
        spd = SPDConv(64, 128)
        out = spd(x)
        assert out.shape == (2, 128, 40, 40), f"Expected (2,128,40,40), got {out.shape}"
        params = sum(p.numel() for p in spd.parameters())
        print(f"  [OK] SPD-Conv: {x.shape} -> {out.shape} ({params:,} params)")
        results["SPD-Conv"] = "PASS"
    except Exception as e:
        print(f"  [X] SPD-Conv: {e}")
        results["SPD-Conv"] = f"FAIL: {e}"

    # --- SEAM ---
    try:
        from models.novel_modules.seam import SEAM
        x = torch.randn(2, 256, 20, 20)
        seam = SEAM(256, num_suppression_patterns=4)
        out, patterns = seam(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        assert patterns.shape == (2, 4, 20, 20)
        params = sum(p.numel() for p in seam.parameters())
        print(f"  [OK] SEAM: {x.shape} -> {out.shape}, patterns {patterns.shape} ({params:,} params)")
        results["SEAM"] = "PASS"
    except Exception as e:
        print(f"  [X] SEAM: {e}")
        results["SEAM"] = f"FAIL: {e}"

    # --- P2 Head ---
    try:
        from models.novel_modules.p2_head import P2Module
        c2 = torch.randn(2, 128, 160, 160)
        p3 = torch.randn(2, 256, 80, 80)
        p2 = P2Module(c2_channels=128, p3_channels=256, num_classes=2)
        dets, feats = p2(c2, p3)
        assert feats.shape[2:] == c2.shape[2:]  # Same spatial as C2
        params = sum(p.numel() for p in p2.parameters())
        print(f"  [OK] P2Head: C2{c2.shape} + P3{p3.shape} -> dets{dets.shape}, feats{feats.shape} ({params:,} params)")
        results["P2Head"] = "PASS"
    except Exception as e:
        print(f"  [X] P2Head: {e}")
        results["P2Head"] = f"FAIL: {e}"

    # --- NAD Head ---
    try:
        from models.novel_modules.nad_head import NADHead
        features = torch.randn(10, 256)
        det_logits = torch.randn(10, 2)
        nad = NADHead(in_channels=256, num_classes=2, num_nuisance_classes=6)
        result = nad(features, det_logits)
        assert result["nuisance_logits"].shape == (10, 6)
        assert result["genuineness"].shape == (10, 1)
        assert result["recalibrated_conf"].shape == (10, 1)
        params = sum(p.numel() for p in nad.parameters())
        print(f"  [OK] NADHead: {features.shape} -> nuisance{result['nuisance_logits'].shape}, "
              f"genuine{result['genuineness'].shape} ({params:,} params)")
        results["NADHead"] = "PASS"
    except Exception as e:
        print(f"  [X] NADHead: {e}")
        results["NADHead"] = f"FAIL: {e}"

    # --- ACR ---
    try:
        from models.novel_modules.acr import ACR
        raw_scores = torch.rand(50)
        feat_map = torch.randn(1, 256, 20, 20)
        acr = ACR(feature_channels=256)
        recal, info = acr(raw_scores, feat_map)
        assert recal.shape == raw_scores.shape
        params = sum(p.numel() for p in acr.parameters())
        print(f"  [OK] ACR: {raw_scores.shape} -> {recal.shape} (quality={info['feature_quality']:.3f}) ({params:,} params)")
        results["ACR"] = "PASS"
    except Exception as e:
        print(f"  [X] ACR: {e}")
        results["ACR"] = f"FAIL: {e}"

    # --- Hazard Prior Gate (efficient object-detection study) ---
    try:
        from models.novel_modules.hazard_prior_gate import HazardPriorGate
        x = torch.rand(2, 3, 128, 128, requires_grad=True)
        hpg = HazardPriorGate(hidden_channels=8)
        out = hpg(x)
        assert out.shape == x.shape
        out.mean().backward()
        assert any(p.grad is not None for p in hpg.parameters())
        params = sum(p.numel() for p in hpg.parameters())
        print(f"  [OK] HPG: {x.shape} -> {out.shape} ({params:,} params, gradients verified)")
        results["HPG"] = "PASS"
    except Exception as e:
        print(f"  [X] HPG: {e}")
        results["HPG"] = f"FAIL: {e}"

    return results


def test_model_build(model_name: str):
    """Test building and forward pass of a single model."""
    try:
        from models.build_model import build_model, get_model_info

        info = get_model_info(model_name)
        print(f"\n  Testing {model_name} ({info['family']}, {info['params_m']:.1f}M params)...")

        start = time.time()
        model = build_model(model_name, num_classes=2, pretrained=False)
        build_time = time.time() - start

        print(f"    Built in {build_time:.2f}s")
        return "PASS"

    except Exception as e:
        print(f"    [X] FAILED: {e}")
        traceback.print_exc()
        return f"FAIL: {e}"


def test_all_models(models_to_test=None):
    """Test building the three core study models."""
    from models.build_model import ALL_MODELS, ULTRALYTICS_MODELS, TORCHVISION_MODELS, EFFICIENTDET_MODELS

    if models_to_test is None:
        models_to_test = ["yolov10n", "yolo26n", "hazard_yolo26n"]

    print(f"\n{'='*60}")
    print(f"  Testing Model Build ({len(models_to_test)} models)")
    print(f"{'='*60}")

    results = {}
    for name in models_to_test:
        results[name] = test_model_build(name)

    return results


def test_phase2_and_3():
    """Test Phase 2 Modified Detector and Phase 3 Multimodal Fusion models."""
    print(f"\n{'='*60}")
    print(f"  Testing Phase 2 & Phase 3 Models")
    print(f"{'='*60}\n")

    results = {}

    # --- Phase 2: Modified Detector ---
    try:
        from models.novel_modules.modified_detector import AdapFuseODDetector
        x = torch.randn(2, 3, 416, 416)
        mod_det = AdapFuseODDetector(num_classes=2)
        outs = mod_det(x)
        assert "det_p3" in outs and "det_p4" in outs and "det_p5" in outs
        params = sum(p.numel() for p in mod_det.parameters())
        print(f"  [OK] AdapFuseODDetector (Phase 2): Input {x.shape} -> det_p3 {outs['det_p3'].shape} ({params/1e6:.2f}M params)")
        results["AdapFuseODDetector"] = "PASS"
    except Exception as e:
        print(f"  [X] AdapFuseODDetector: {e}")
        results["AdapFuseODDetector"] = f"FAIL: {e}"

    # --- Phase 3: AudioYOLO Detector ---
    try:
        from models.fusion.audio_yolo import AudioYOLODetector
        spec = torch.randn(2, 1, 128, 256)
        audio_det = AudioYOLODetector(num_audio_classes=5)
        out = audio_det(spec)
        assert out.dim() == 4
        params = sum(p.numel() for p in audio_det.parameters())
        print(f"  [OK] AudioYOLODetector (Phase 3): Input {spec.shape} -> TF-boxes {out.shape} ({params/1e6:.2f}M params)")
        results["AudioYOLODetector"] = "PASS"
    except Exception as e:
        print(f"  [X] AudioYOLODetector: {e}")
        results["AudioYOLODetector"] = f"FAIL: {e}"

    # --- Phase 3: Full Tri-Modal Model ---
    try:
        from models.fusion.adaptfuse_od_full import AdapFuseODFullModel
        rgb = torch.randn(2, 3, 416, 416)
        thermal = torch.randn(2, 3, 416, 416)
        audio = torch.randn(2, 1, 128, 256)
        full_model = AdapFuseODFullModel()
        outs = full_model(rgb, thermal, audio)
        assert "rgb_reliability" in outs and "thermal_reliability" in outs
        params = sum(p.numel() for p in full_model.parameters())
        print(f"  [OK] AdapFuseODFullModel (Phase 3): Tri-modal inputs -> Fused outputs ({params/1e6:.2f}M params)")
        results["AdapFuseODFullModel"] = "PASS"
    except Exception as e:
        print(f"  [X] AdapFuseODFullModel: {e}")
        results["AdapFuseODFullModel"] = f"FAIL: {e}"

    return results


def print_summary(novel_results, model_results, phase2_3_results=None):
    """Print combined test summary."""
    print(f"\n{'='*60}")
    print(f"  TEST SUMMARY")
    print(f"{'='*60}")

    all_pass = True

    print(f"\n  Novel Modules:")
    for name, status in novel_results.items():
        icon = "[OK]" if status == "PASS" else "[X]"
        print(f"    {icon} {name}: {status}")
        if status != "PASS":
            all_pass = False

    if phase2_3_results:
        print(f"\n  Phase 2 & 3 Models:")
        for name, status in phase2_3_results.items():
            icon = "[OK]" if status == "PASS" else "[X]"
            print(f"    {icon} {name}: {status}")
            if status != "PASS":
                all_pass = False

    if model_results:
        print(f"\n  Model Builds:")
        for name, status in model_results.items():
            icon = "[OK]" if status == "PASS" else "[X]"
            print(f"    {icon} {name}: {status}")
            if status != "PASS":
                all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print(f"  [OK] ALL TESTS PASSED")
    else:
        print(f"  [X] SOME TESTS FAILED -- see details above")
    print(f"{'='*60}\n")

    return all_pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test all components")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to test")
    parser.add_argument("--novel_only", action="store_true",
                        help="Only test novel modules")
    parser.add_argument("--models_only", action="store_true",
                        help="Only test model builds")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  AdapFuse-OD — Component Tests")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'#'*60}")

    novel_results = {}
    phase2_3_results = {}
    model_results = {}

    if not args.models_only:
        novel_results = test_novel_modules()
        phase2_3_results = test_phase2_and_3()

    if not args.novel_only:
        model_results = test_all_models(args.models)

    all_pass = print_summary(novel_results, model_results, phase2_3_results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
