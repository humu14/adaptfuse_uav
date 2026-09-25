"""
External SOTA Baseline Architectures for UAV Multimodal Disaster Detection Benchmark.
Implements:
  1. DeepFire (Khan et al., 2022): Dual-stream ResNet (RGB + Thermal) with fixed intermediate concatenation.
  2. XuCrossAttention (Xu et al., 2020): Tri-modal Multi-Head Cross-Attention without RUE or Nuisance Head.
  3. FireRGBTNet (2024): Lightweight spatial gated dual-stream architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.backbones.rgb_backbone import RGBBackbone
from models.backbones.thermal_backbone import ThermalBackbone
from models.backbones.audio_backbone import AudioCNN


# ─────────────────────────────────────────────────────────────────────────────
# 1. DeepFire (Khan et al., Mobile Information Systems 2022)
# ─────────────────────────────────────────────────────────────────────────────

class DeepFireModel(nn.Module):
    """
    DeepFire architecture: Dual-branch CNN for RGB and Thermal streams.
    Extracts intermediate features from each branch, concatenates them statically,
    and classifies via MLP.
    No audio modality, no dynamic reliability estimation, no nuisance head.
    """
    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        proj_dim: int = 192,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rgb_backbone = RGBBackbone(pretrained=True)         # 576-dim
        self.thermal_backbone = ThermalBackbone(pretrained=True) # 512-dim

        self.rgb_proj = nn.Sequential(
            nn.Linear(576, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.thermal_proj = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Concatenated representation: proj_dim * 2 = 384
        fusion_dim = proj_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.disaster_head = nn.Linear(128, num_disaster_classes)
        self.victim_head = nn.Linear(128, num_victim_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_rgb, _ = self.rgb_backbone(rgb)
        f_th, _ = self.thermal_backbone(thermal)

        if has_rgb is not None:
            f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        if has_thermal is not None:
            f_th = f_th * has_thermal.unsqueeze(-1)

        p_rgb = self.rgb_proj(f_rgb)
        p_th = self.thermal_proj(f_th)

        fused = torch.cat([p_rgb, p_th], dim=-1)
        feat = self.classifier(fused)

        disaster_logits = self.disaster_head(feat)
        victim_logits = self.victim_head(feat)

        # DeepFire uses fixed static weights (0.5 RGB, 0.5 Thermal, 0.0 Audio)
        B = rgb.size(0)
        fixed_rel = torch.tensor([0.5, 0.5, 0.0], device=rgb.device).unsqueeze(0).expand(B, -1)

        return disaster_logits, victim_logits, fixed_rel


# ─────────────────────────────────────────────────────────────────────────────
# 2. Xu et al. Cross-Modal Attention (IEEE Trans. Multimedia 2020)
# ─────────────────────────────────────────────────────────────────────────────

class XuCrossAttentionModel(nn.Module):
    """
    Xu et al. style tri-modal cross-attention fusion.
    Features from RGB, Thermal, and Audio are projected into a common subspace
    and fused via un-gated Multi-Head Cross-Attention.
    Lacks RUE (cannot down-weight corrupted sensors) and lacks NuisanceAwareHead.
    """
    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        proj_dim: int = 192,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rgb_backbone = RGBBackbone(pretrained=True)
        self.thermal_backbone = ThermalBackbone(pretrained=True)
        self.audio_backbone = AudioCNN()

        self.rgb_proj = nn.Sequential(nn.Linear(576, proj_dim), nn.LayerNorm(proj_dim))
        self.thermal_proj = nn.Sequential(nn.Linear(512, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(proj_dim)

        self.disaster_head = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_victim_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        audio: torch.Tensor,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_rgb, _ = self.rgb_backbone(rgb)
        f_th, _ = self.thermal_backbone(thermal)
        f_au, _ = self.audio_backbone(audio)

        if has_rgb is not None:
            f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        if has_thermal is not None:
            f_th = f_th * has_thermal.unsqueeze(-1)
        if has_audio is not None:
            f_au = f_au * has_audio.unsqueeze(-1)

        p_rgb = self.rgb_proj(f_rgb).unsqueeze(1)  # (B, 1, D)
        p_th = self.thermal_proj(f_th).unsqueeze(1)
        p_au = self.audio_proj(f_au).unsqueeze(1)

        tokens = torch.cat([p_rgb, p_th, p_au], dim=1) # (B, 3, D)

        # Standard self/cross-attention across tokens without reliability weighting
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)

        # Mean pooling across tokens
        fused = tokens.mean(dim=1)

        disaster_logits = self.disaster_head(fused)
        victim_logits = self.victim_head(fused)

        # Fixed uniform trust (0.33, 0.33, 0.33)
        B = rgb.size(0)
        rel = torch.full((B, 3), 0.333, device=rgb.device)

        return disaster_logits, victim_logits, rel


# ─────────────────────────────────────────────────────────────────────────────
# 3. FireRGBTNet (MDPI Sensors 2024)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGatedFusionBlock(nn.Module):
    """Spatial Gated Fusion (SGF) block from FireRGBTNet (2024)."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, th_feat: torch.Tensor) -> torch.Tensor:
        # Spatial gating
        cat = torch.cat([rgb_feat, th_feat], dim=1)
        gate = self.conv_gate(cat)
        gated_rgb = rgb_feat * gate
        gated_th = th_feat * (1.0 - gate)
        return self.conv_fuse(torch.cat([gated_rgb, gated_th], dim=1))


class FireRGBTNetModel(nn.Module):
    """
    FireRGBTNet (2024): Lightweight dual-stream RGB-T network with
    Spatial Gated Fusion.
    """
    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        proj_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rgb_backbone = RGBBackbone(pretrained=True)
        self.thermal_backbone = ThermalBackbone(pretrained=True)

        self.rgb_proj = nn.Sequential(
            nn.Linear(576, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.thermal_proj = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Gated fusion layer
        self.gate_fc = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.disaster_head = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_victim_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_rgb, _ = self.rgb_backbone(rgb)
        f_th, _ = self.thermal_backbone(thermal)

        if has_rgb is not None:
            f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        if has_thermal is not None:
            f_th = f_th * has_thermal.unsqueeze(-1)

        p_rgb = self.rgb_proj(f_rgb)
        p_th = self.thermal_proj(f_th)

        cat = torch.cat([p_rgb, p_th], dim=-1)
        gate = self.gate_fc(cat)
        gated = torch.cat([p_rgb * gate, p_th * (1.0 - gate)], dim=-1)
        fused = self.fuse_fc(gated)

        disaster_logits = self.disaster_head(fused)
        victim_logits = self.victim_head(fused)

        B = rgb.size(0)
        rel = torch.cat([gate.mean(dim=1, keepdim=True), (1.0 - gate).mean(dim=1, keepdim=True), torch.zeros(B, 1, device=rgb.device)], dim=1)

        return disaster_logits, victim_logits, rel


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unbiased Dynamic Routing Fusion (CVPR 2026 Style)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicRoutingFusionModel(nn.Module):
    """
    CVPR 2026 Style Unbiased Dynamic Routing:
    Uses a soft routing network across all modalities with modal balance loss.
    Lacks UAV domain-specific physics RUE tokens or nuisance modeling.
    """
    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        proj_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rgb_backbone = RGBBackbone(pretrained=True)
        self.thermal_backbone = ThermalBackbone(pretrained=True)
        self.audio_backbone = AudioCNN()

        self.rgb_proj = nn.Sequential(nn.Linear(576, proj_dim), nn.LayerNorm(proj_dim))
        self.thermal_proj = nn.Sequential(nn.Linear(512, proj_dim), nn.LayerNorm(proj_dim))
        self.audio_proj = nn.Sequential(nn.Linear(128, proj_dim), nn.LayerNorm(proj_dim))

        # Dynamic Router
        self.router = nn.Sequential(
            nn.Linear(proj_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.disaster_head = nn.Linear(proj_dim, num_disaster_classes)
        self.victim_head = nn.Linear(proj_dim, num_victim_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        audio: torch.Tensor,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_rgb, _ = self.rgb_backbone(rgb)
        f_th, _ = self.thermal_backbone(thermal)
        f_au, _ = self.audio_backbone(audio)

        if has_rgb is not None:
            f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        if has_thermal is not None:
            f_th = f_th * has_thermal.unsqueeze(-1)
        if has_audio is not None:
            f_au = f_au * has_audio.unsqueeze(-1)

        p_rgb = self.rgb_proj(f_rgb)
        p_th = self.thermal_proj(f_th)
        p_au = self.audio_proj(f_au)

        # Dynamic routing weights
        concat_all = torch.cat([p_rgb, p_th, p_au], dim=-1)
        routing_weights = self.router(concat_all) # (B, 3)

        w_rgb = routing_weights[:, 0:1]
        w_th = routing_weights[:, 1:2]
        w_au = routing_weights[:, 2:3]

        fused = w_rgb * p_rgb + w_th * p_th + w_au * p_au
        fused = self.fusion_mlp(fused)

        disaster_logits = self.disaster_head(fused)
        victim_logits = self.victim_head(fused)

        return disaster_logits, victim_logits, routing_weights
