# Backbone Experiment Results

_Updated: 2026-06-01 14:20_

## Test Set Results

| EXP | Description | Disaster F1 | Victim F1 | Combined F1 | Disaster Acc | Expected gain |
|-----|-------------|-------------|-----------|-------------|--------------|---------------|
| baseline | AdapFuse v1 (MobileNetV3 + ResNet18 + AudioCNN) | 0.9913 | 0.9836 | 0.9874 | 0.9944 | — |
| EXP-14 | PANNS-CNN6 audio (AudioSet pretrained) | 0.9858 | 0.9792 | 0.9825 | 0.9907 | +5–10% audio F1 |
| EXP-15 | EfficientNet-B0 RGB (1280d) | pending | pending | pending | pending | +2–4% disaster F1 |
| EXP-16 | EfficientNet-B0 thermal (1280d, 1-ch) | pending | pending | pending | pending | symmetric backbones |
| EXP-17 | MobileViT-XXS RGB (CNN+ViT, 320d) | pending | pending | pending | pending | aerial context |
| EXP-18 | GDBlock audio (multi-scale dilated) | 0.9888 | 0.9838 | 0.9863 | 0.9924 | efficiency + citable |
| EXP-21 | Domain prompt tokens (DPMamba-inspired) | verified | verified | verified | verified | +2–5% missing-modal |
| EXP-22 | Spatial alignment attention before GAP | pending | pending | pending | pending | +1–2% under smoke |
| **UniAdapFuse** | Unified Dual-Scale Cls + Dense Det (FPN+P2+RUE) | verified | verified | verified | verified | **End-to-End Joint Perception** |

## Status

- **baseline** AdapFuse v1 (MobileNetV3 + ResNet18 + AudioCNN): ✅ done
- **EXP-14** PANNS-CNN6 audio (AudioSet pretrained): ✅ done
- **EXP-15** EfficientNet-B0 RGB (1280d): ⏳ pending
- **EXP-16** EfficientNet-B0 thermal (1280d, 1-ch): ⏳ pending
- **EXP-17** MobileViT-XXS RGB (CNN+ViT, 320d): ⏳ pending
- **EXP-18** GDBlock audio (multi-scale dilated): ✅ done
- **EXP-21** Domain prompt tokens (DPMamba-inspired): ✅ verified in foreground
- **EXP-22** Spatial alignment attention before GAP: ⏳ pending
- **UniAdapFuse** Unified Multi-Task Perception: ✅ verified in foreground

## Run order

```powershell
cd adaptfuse_uav
# EXP-14 (overnight first — safest, highest impact)
python training/train.py --config configs/adaptfuse_v1_panns.yaml
# EXP-18 (fast, ~1.5h)
python training/train.py --config configs/adaptfuse_v1_gdblock.yaml
# EXP-21 (fast, ~2h)
python training/train.py --config configs/adaptfuse_v1_domain_prompts.yaml
# EXP-22 (fast, ~2h)
python training/train.py --config configs/adaptfuse_v1_spatial_attn.yaml
# EXP-15 (batch=8, ~4-5h)
python training/train.py --config configs/adaptfuse_v1_efficientnet_rgb.yaml
# EXP-16 (batch=8, ~4-5h)
python training/train.py --config configs/adaptfuse_v1_efficientnet_thermal.yaml
# EXP-17 (batch=8, ~5-6h, requires: pip install timm)
python training/train.py --config configs/adaptfuse_v1_mobilevit.yaml
```
