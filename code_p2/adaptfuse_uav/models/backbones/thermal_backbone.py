"""
Thermal backbone: ResNet18 with 1-channel input.
Design A2 unimodal baseline + shared encoder for fusion models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# EXP-22: CBAM-style spatial alignment attention (shared with rgb_backbone)
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


class ThermalBackbone(nn.Module):
    """
    ResNet18 adapted for single-channel (grayscale/thermal) input.
    First conv layer is modified: 3→1 channel, weights averaged.
    Output feature dim: 512.
    """

    def __init__(self, pretrained: bool = True, use_spatial_attn: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Modify first conv: 3→1 channel
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            1, 64,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Average pretrained weights across RGB channel dim
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))

        resnet.conv1 = new_conv

        if use_spatial_attn:
            # Split out avgpool so spatial attn can be applied before it
            self.net = None
            self.backbone_features = nn.Sequential(*list(resnet.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.spatial_attn = SpatialAlignmentAttention(512)
        else:
            # Original structure: conv→...→avgpool fused in self.net
            self.net = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone_features = None
            self.avgpool = None
            self.spatial_attn = None

        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 224, 224)
        returns: (B, 512)
        """
        if self.spatial_attn is not None:
            x = self.backbone_features(x)  # (B, 512, 7, 7)
            x = self.spatial_attn(x)
            x = self.avgpool(x)            # (B, 512, 1, 1)
        else:
            x = self.net(x)                # (B, 512, 1, 1)
        return torch.flatten(x, 1)


class ThermalOnlyModel(nn.Module):
    """
    Design A2: Thermal-only classification model.
    ResNet18(1ch) → GAP → Linear(512,256) → ReLU → Linear(256, num_classes)
    """

    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone: str = "resnet18",
    ):
        super().__init__()
        if backbone == "efficientnet_b0":
            self.backbone = EfficientNetB0ThermalBackbone(pretrained=pretrained)
        elif backbone == "mobilevit_xxs":
            self.backbone = MobileViTXXSThermalBackbone(pretrained=pretrained)
        else:
            self.backbone = ThermalBackbone(pretrained=pretrained)
        feat_dim = self.backbone.out_dim  # 512 / 1280 / 320

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
        rgb: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ):
        feat = self.backbone(thermal)
        disaster_out = self.disaster_head(feat)
        victim_out = self.victim_head(feat)
        reliability = torch.ones(thermal.size(0), 3, device=thermal.device) / 3.0
        return disaster_out, victim_out, reliability


# ─────────────────────────────────────────────────────────────────────────────
# EXP-16: EfficientNet-B0 thermal backbone (1-channel, symmetric with RGB)
# Compound scaling handles thermal spatial features well. out_dim=1280
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0ThermalBackbone(nn.Module):
    """EfficientNet-B0 adapted for 1-channel thermal input. out_dim=1280."""

    def __init__(self, pretrained: bool = True, use_spatial_attn: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff = models.efficientnet_b0(weights=weights)

        # Replace first conv: 3→1 channel, average RGB weights
        old_conv = eff.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        eff.features[0][0] = new_conv

        self.features = eff.features
        self.avgpool = eff.avgpool
        self.out_dim = 1280
        self.spatial_attn = SpatialAlignmentAttention(1280) if use_spatial_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 224, 224) → (B, 1280)"""
        x = self.features(x)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MobileViT-XXS thermal backbone (1-channel, symmetric with RGB EXP-17)
# 1.3M params, out_dim=320. Requires: pip install timm
# ─────────────────────────────────────────────────────────────────────────────

class MobileViTXXSThermalBackbone(nn.Module):
    """MobileViT-XXS for 1-channel thermal input. out_dim=320. Requires timm.
    timm adapts the pretrained stem conv to in_chans=1 by summing RGB weights."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("MobileViT requires timm: pip install timm")
        self.model = timm.create_model(
            'mobilevit_xxs', pretrained=pretrained, num_classes=0, in_chans=1)
        self.out_dim = self.model.num_features  # 320

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, 320)"""
        return self.model(x)
