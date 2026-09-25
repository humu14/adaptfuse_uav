"""
RGB backbone: MobileNetV3-Small with customizable output.
Design A1 unimodal baseline + shared encoder for fusion models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# EXP-22: CBAM-style spatial alignment attention (Zhang et al. 2025 inspired)
# Applied before GAP to focus on spatially consistent regions.
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAlignmentAttention(nn.Module):
    """Channel + spatial attention gate applied before GAP. ~50K params."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // 4), channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        ca = self.channel_gate(x).unsqueeze(-1).unsqueeze(-1)
        x = x * ca
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        sa = self.spatial_gate(torch.cat([avg, mx], dim=1))
        return x * sa


class RGBBackbone(nn.Module):
    """
    MobileNetV3-Small backbone for RGB images.
    Output feature dim: 576 (before classification head).
    """

    def __init__(self, pretrained: bool = True, freeze_early: bool = False,
                 use_spatial_attn: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)

        # Remove the classifier (avgpool + flatten + linear)
        # Keep features (conv layers) up to adaptive avgpool
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool  # AdaptiveAvgPool2d(1)
        self.out_dim = 576
        self.spatial_attn = SpatialAlignmentAttention(576) if use_spatial_attn else None

        if freeze_early:
            # Freeze first 6 layers for transfer learning
            for i, layer in enumerate(self.features):
                if i < 6:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224)
        returns: (B, 576)
        """
        x = self.features(x)         # (B, 576, 7, 7)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)
        x = self.avgpool(x)          # (B, 576, 1, 1)
        x = torch.flatten(x, 1)      # (B, 576)
        return x


class RGBOnlyModel(nn.Module):
    """
    Design A1: RGB-only classification model.
    MobileNetV3-Small → Linear(576,256) → ReLU → Linear(256, num_classes)
    """

    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone: str = "mobilenet_v3_small",
    ):
        super().__init__()
        if backbone == "efficientnet_b0":
            self.backbone = EfficientNetB0RGBBackbone(pretrained=pretrained)
        elif backbone == "mobilevit_xxs":
            self.backbone = MobileViTXXSBackbone(pretrained=pretrained)
        else:
            self.backbone = RGBBackbone(pretrained=pretrained)
        feat_dim = self.backbone.out_dim  # 576 / 1280 / 320

        self.disaster_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_victim_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ):
        feat = self.backbone(rgb)
        disaster_out = self.disaster_head(feat)
        victim_out = self.victim_head(feat)
        reliability = torch.ones(rgb.size(0), 3, device=rgb.device) / 3.0
        return disaster_out, victim_out, reliability


# ─────────────────────────────────────────────────────────────────────────────
# EXP-15: EfficientNet-B0 RGB backbone
# +11% ImageNet acc vs MobileNetV3-Small, 5.3M params, out_dim=1280
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0RGBBackbone(nn.Module):
    """EfficientNet-B0 for RGB. out_dim=1280."""

    def __init__(self, pretrained: bool = True, use_spatial_attn: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff = models.efficientnet_b0(weights=weights)
        self.features = eff.features
        self.avgpool = eff.avgpool
        self.out_dim = 1280
        self.spatial_attn = SpatialAlignmentAttention(1280) if use_spatial_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, 1280)"""
        x = self.features(x)      # (B, 1280, 7, 7)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)
        x = self.avgpool(x)       # (B, 1280, 1, 1)
        return torch.flatten(x, 1)


# ─────────────────────────────────────────────────────────────────────────────
# EXP-17: MobileViT-XXS RGB backbone (CNN+ViT hybrid)
# 1.3M params, out_dim=320. Requires: pip install timm
# ─────────────────────────────────────────────────────────────────────────────

class MobileViTXXSBackbone(nn.Module):
    """MobileViT-XXS hybrid CNN-ViT. out_dim=320. Requires timm."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("MobileViT requires timm: pip install timm")
        self.model = timm.create_model('mobilevit_xxs', pretrained=pretrained, num_classes=0)
        self.out_dim = self.model.num_features  # 320

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, 320)"""
        return self.model(x)
