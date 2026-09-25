"""
Backbone candidate cost benchmark (backbone-selection justification).

For each candidate encoder, measures on the local GPU:
  - parameters (M)
  - forward GFLOPs for one input (torch FlopCounterMode; 1 MAC = 2 FLOPs)
  - peak training memory (MB) for one AMP fwd+bwd+step at the training batch size
  - batch-1 fp32 inference latency (ms, mean of 100 runs after warm-up)
ImageNet top-1 for torchvision models is read from the weights metadata.

Usage:
    python scripts/benchmark_backbone_candidates.py
Writes outputs/logs/backbone_candidate_benchmark.json
"""

import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.backbones.audio_backbone import AudioCNN, AudioCNNGD, PANNsAudioBackbone


def tv(name, weights_enum):
    """torchvision classifier with its head removed; top-1 from weights metadata."""
    def build():
        m = getattr(tvm, name)(weights=None)
        for attr in ("classifier", "fc", "heads", "head"):
            if hasattr(m, attr):
                setattr(m, attr, nn.Identity())
        return m
    w = getattr(tvm, weights_enum).IMAGENET1K_V1
    top1 = w.meta["_metrics"]["ImageNet-1K"]["acc@1"]
    return build, top1


def timm_model(name, top1):
    def build():
        import timm
        return timm.create_model(name, pretrained=False, num_classes=0)
    return build, top1


VISUAL = {
    # name: (builder, ImageNet top-1)
    "MobileNetV3-Small": tv("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "MobileNetV3-Large": tv("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "MobileNetV2": tv("mobilenet_v2", "MobileNet_V2_Weights"),
    "ShuffleNetV2-x1.0": tv("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights"),
    "MobileViT-XXS": timm_model("mobilevit_xxs", 69.0),   # Mehta & Rastegari 2022
    "MobileViT-S": timm_model("mobilevit_s", 78.4),       # Mehta & Rastegari 2022
    "EfficientNet-B0": tv("efficientnet_b0", "EfficientNet_B0_Weights"),
    "EfficientNet-B3": tv("efficientnet_b3", "EfficientNet_B3_Weights"),
    "ResNet-18": tv("resnet18", "ResNet18_Weights"),
    "ResNet-50": tv("resnet50", "ResNet50_Weights"),
    "ConvNeXt-Tiny": tv("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "Swin-T": tv("swin_t", "Swin_T_Weights"),
    "ViT-B/16": tv("vit_b_16", "ViT_B_16_Weights"),
}

AUDIO = {
    "AudioCNN": (lambda: AudioCNN(), None),
    "AudioCNN + GDBlock": (lambda: AudioCNNGD(), None),
    "PANNS-CNN6 encoder": (lambda: PANNsAudioBackbone(pretrained=False), None),
}


def gflops(model, x):
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def peak_train_mb(model, shape, batch, device):
    model.train().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(2):
            x = torch.randn(batch, *shape, device=device)
            with torch.autocast("cuda"):
                out = model(x)
                loss = out.float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 2**20
    except torch.cuda.OutOfMemoryError:
        return float("inf")
    finally:
        del opt
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def latency_ms(model, shape, device, n=100):
    model.eval().to(device)
    x = torch.randn(1, *shape, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def run(table, shape, batches, device):
    rows = []
    for name, (build, top1) in table.items():
        model = build()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        gf = gflops(model, torch.randn(1, *shape))
        mem = {f"train_mb_bs{b}": peak_train_mb(model, shape, b, device) for b in batches}
        lat = latency_ms(model, shape, device)
        row = {"backbone": name, "params_m": round(params, 2), "gflops": round(gf, 3),
               "imagenet_top1": top1, "latency_ms_bs1": round(lat, 2),
               **{k: (round(v) if v != float("inf") else "OOM") for k, v in mem.items()}}
        print(row)
        rows.append(row)
        model.cpu()
        del model
        torch.cuda.empty_cache()
    return rows


def main():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    visual = run(VISUAL, (3, 224, 224), (8, 16), device)
    audio = run(AUDIO, (1, 64, 63), (16,), device)
    out = ROOT / "outputs" / "logs" / "backbone_candidate_benchmark.json"
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
                               "visual": visual, "audio": audio}, indent=2))
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
