"""
EXP-13: Per-Module Latency Breakdown (eval-only, no training needed).

Times each stage of AdapFuseV1 individually to identify the bottleneck.
Expected: backbone dominates; RUE adds <1ms (tiny MLP).

Usage:
    python evaluation/profile_latency.py --config configs/adaptfuse_v1.yaml \
        --checkpoint outputs/checkpoints/adaptfuse_v1_best.pth

Output: printed table + outputs/logs/<exp>_latency_profile.json
"""

import sys
import json
import argparse
import time
import yaml
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.full_model import build_model, AdapFuseV1


def time_fn(fn, *args, warmup=10, n=100, device=None):
    for _ in range(warmup):
        with torch.no_grad():
            fn(*args)
    if device and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            fn(*args)
    if device and device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms


def profile_adaptfuse(model: AdapFuseV1, device: torch.device, batch_size: int = 1) -> dict:
    B = batch_size
    rgb_t    = torch.randn(B, 3, 224, 224, device=device)
    th_t     = torch.randn(B, 1, 224, 224, device=device)
    au_t     = torch.randn(B, 1, 64, 63, device=device)
    has_ones = torch.ones(B, device=device)

    # Stage inputs built from model internals
    with torch.no_grad():
        f_rgb_raw = model.rgb_enc(rgb_t)
        f_th_raw  = model.thermal_enc(th_t)
        f_au_raw  = model.audio_enc(au_t)
        f_rgb = model.rgb_proj(f_rgb_raw)
        f_th  = model.thermal_proj(f_th_raw)
        f_au  = model.audio_proj(f_au_raw)
        q_rgb = model.rgb_quality(f_rgb_raw)
        q_th  = model.thermal_quality(f_th_raw)
        q_au  = model.audio_quality(f_au_raw)
        features = {'rgb': f_rgb, 'thermal': f_th, 'audio': f_au}
        quality_tokens = {'rgb': q_rgb, 'thermal': q_th, 'audio': q_au}
        has_modality = {'rgb': has_ones, 'thermal': has_ones, 'audio': has_ones}
        rel, _, _ = model.rue(features, quality_tokens, has_modality)
        tokens = torch.stack([f_rgb, f_th, f_au], dim=1)

    results = {}

    results["rgb_backbone"]     = time_fn(model.rgb_enc, rgb_t, device=device)
    results["thermal_backbone"] = time_fn(model.thermal_enc, th_t, device=device)
    results["audio_backbone"]   = time_fn(model.audio_enc, au_t, device=device)

    def proj_all(*_):
        model.rgb_proj(f_rgb_raw); model.thermal_proj(f_th_raw); model.audio_proj(f_au_raw)
    results["projectors_all"] = time_fn(proj_all, device=device)

    def quality_all(*_):
        model.rgb_quality(f_rgb_raw); model.thermal_quality(f_th_raw); model.audio_quality(f_au_raw)
    results["quality_heads_all"] = time_fn(quality_all, device=device)

    results["rue"] = time_fn(model.rue, features, quality_tokens, has_modality, device=device)

    def cross_attn_fn(*_):
        model.cross_attn(tokens, tokens, tokens)
    results["cross_attention"] = time_fn(cross_attn_fn, device=device)

    fused = tokens.mean(dim=1)
    results["gru"] = time_fn(lambda *_: model.gru(fused.unsqueeze(1)), device=device)

    results["head"] = time_fn(model.head, fused, device=device)

    # Total pipeline
    def full_forward(*_):
        model(rgb_t, th_t, au_t, has_ones, has_ones, has_ones)
    results["total_pipeline"] = time_fn(full_forward, device=device)

    results["_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    results["_batch_size"] = B

    return results


def print_profile(results: dict):
    total = results.get("total_pipeline", 1.0)
    print(f"\n{'='*55}")
    print(f"Per-Module Latency Profile (batch={results['_batch_size']})")
    print(f"{'='*55}")
    print(f"  {'Module':<25} {'ms':>8}  {'%total':>7}")
    print(f"  {'-'*45}")
    stages = [
        ("rgb_backbone",     "RGB backbone"),
        ("thermal_backbone", "Thermal backbone"),
        ("audio_backbone",   "Audio backbone"),
        ("projectors_all",   "Projectors (all 3)"),
        ("quality_heads_all","Quality heads (all)"),
        ("rue",              "RUE"),
        ("cross_attention",  "Cross-modal attention"),
        ("gru",              "GRU"),
        ("head",             "NuisanceHead"),
        ("total_pipeline",   "TOTAL pipeline"),
    ]
    for key, label in stages:
        ms = results.get(key, 0.0)
        pct = ms / total * 100 if total > 0 else 0
        print(f"  {label:<25} {ms:>8.2f}  {pct:>6.1f}%")
    print(f"\n  Parameters: {results['_params_M']:.2f}M")
    print(f"  Throughput: {1000/total:.1f} fps (batch={results['_batch_size']})")


def main():
    parser = argparse.ArgumentParser(description="EXP-13: Per-module latency profiling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = config.get("experiment_name", "experiment")

    if config.get("model") != "adaptfuse_v1":
        print("[WARN] profile_latency.py is designed for adaptfuse_v1 — running full-model timing only.")

    ckpt_path = args.checkpoint or str(
        Path(args.output_dir) / "checkpoints" / f"{exp_name}_best.pth"
    )

    model = build_model(config)
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"[EXP-13] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[EXP-13] No checkpoint found — profiling random-init model (structure is same).")

    model.to(device).eval()

    if isinstance(model, AdapFuseV1):
        results = profile_adaptfuse(model, device, args.batch_size)
    else:
        # Fallback: time full forward only
        B = args.batch_size
        dummy_args = (
            torch.randn(B, 3, 224, 224, device=device),
            torch.randn(B, 1, 224, 224, device=device),
            torch.randn(B, 1, 64, 63, device=device),
            torch.ones(B, device=device),
            torch.ones(B, device=device),
            torch.ones(B, device=device),
        )
        ms = time_fn(model, *dummy_args, device=device)
        results = {
            "total_pipeline": ms,
            "_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "_batch_size": B,
        }

    print_profile(results)

    out_path = Path(args.output_dir) / "logs" / f"{exp_name}_latency_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: round(v, 4) if isinstance(v, float) else v
                   for k, v in results.items()}, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
