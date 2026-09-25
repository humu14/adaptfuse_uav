"""
AdapFuse-UAV: Full Advanced Model — Incorporating both implementation plans.
Includes:
  - Quality tokens from each backbone
  - RUE (Reliability & Uncertainty Estimator) with quality signals
  - NuisanceAwareHead for false-alarm suppression
  - All 6 design alternatives (A1-F)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from models.backbones.rgb_backbone import (
    RGBBackbone, RGBOnlyModel,
    EfficientNetB0RGBBackbone, MobileViTXXSBackbone,
)
from models.backbones.thermal_backbone import (
    ThermalBackbone, ThermalOnlyModel,
    EfficientNetB0ThermalBackbone,
)
from models.backbones.audio_backbone import (
    AudioCNN, AudioOnlyModel,
    PANNsAudioBackbone, AudioCNNGD,
)


# ─────────────────────────────────────────────────────────────────────────────
# RUE: Reliability & Uncertainty Estimator (CORE NOVEL MODULE from Plan 2)
# ─────────────────────────────────────────────────────────────────────────────

class ReliabilityUncertaintyEstimator(nn.Module):
    """
    Novel RUE module: estimates per-modality reliability r_m ∈ [0,1]
    and uncertainty u_m ≥ 0 from both quality tokens AND feature statistics.

    Novelty over prior work:
    - Takes quality_tokens (backbone side heads) + feature stats (norm, var)
    - Outputs both reliability AND uncertainty separately
    - Has auxiliary degradation prediction head for supervised training
    - First applied to tri-modal UAV disaster detection (vs sentiment/generic)

    Reference: Inspired by SURE (arXiv 2025) but domain-specific.
    """

    def __init__(self, feat_dim: int = 192, quality_dim: int = 16):
        super().__init__()
        # Each modality contributes: [quality_token(16), feat_norm(1), feat_var(1)]
        in_dim = (quality_dim + 2) * 3

        self.reliability_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),    # r_rgb, r_thermal, r_audio
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),    # u_rgb, u_thermal, u_audio
            nn.Softplus(),       # ensures u >= 0
        )

        # Auxiliary: predict if modality is degraded (supervised by corruption flags)
        self.degradation_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),    # binary: is each modality degraded?
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        quality_tokens: Dict[str, torch.Tensor],
        has_modality: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: {'rgb': (B,D), 'thermal': (B,D), 'audio': (B,D)}
        quality_tokens: {'rgb': (B,Q), 'thermal': (B,Q), 'audio': (B,Q)}
        has_modality: {'rgb': (B,), 'thermal': (B,), 'audio': (B,)}

        Returns:
            reliability: (B, 3) in [0, 1]
            uncertainty: (B, 3) >= 0
            degradation_logits: (B, 3) raw logits
        """
        stats = []
        for key in ['rgb', 'thermal', 'audio']:
            f = features[key]                                    # (B, D)
            q = quality_tokens[key]                              # (B, Q)
            feat_norm = f.norm(dim=-1, keepdim=True)             # (B, 1)
            feat_var = f.var(dim=-1, keepdim=True).clamp(0, 10)  # (B, 1)
            stats.append(torch.cat([q, feat_norm, feat_var], dim=-1))

        combined = torch.cat(stats, dim=-1)  # (B, (Q+2)*3)

        reliability = self.reliability_head(combined)         # (B, 3)
        uncertainty = self.uncertainty_head(combined)         # (B, 3)
        degradation_logits = self.degradation_head(combined)  # (B, 3)

        # Zero out missing modalities — no signal → no reliability
        has = torch.stack([
            has_modality['rgb'],
            has_modality['thermal'],
            has_modality['audio'],
        ], dim=-1)  # (B, 3)

        reliability = reliability * has

        return reliability, uncertainty, degradation_logits


# ─────────────────────────────────────────────────────────────────────────────
# NuisanceAwareHead (2nd Novel Module from Plan 2)
# ─────────────────────────────────────────────────────────────────────────────

class NuisanceAwareHead(nn.Module):
    """
    Jointly predicts:
      1. disaster class (main task)
      2. victim presence (secondary task)
      3. nuisance type (auxiliary — reduces false alarms)

    Nuisance classes:
      0 = clean (no nuisance / true negative)
      1 = solar_heating (hot surface, not fire)
      2 = rgb_false_fire (red/orange object confusion)
      3 = audio_false_alarm (machinery/crowd noise)
      4 = partial_occlusion (partially visible victim)

    This head is absent in ALL prior works (DiRecNetV2, GlimmerNet, Zhang 2025, etc.)
    """

    NUISANCE_CLASSES = [
        'clean', 'solar_heating', 'rgb_false_fire',
        'audio_false_alarm', 'partial_occlusion'
    ]

    def __init__(
        self,
        feat_dim: int = 192,
        n_disaster: int = 4,
        n_victim: int = 2,
        n_nuisance: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.disaster = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, n_disaster),
        )
        self.victim = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 4, n_victim),
        )
        self.nuisance = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_nuisance),
        )

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'disaster': self.disaster(fused),
            'victim': self.victim(fused),
            'nuisance': self.nuisance(fused),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Design B — Early Fusion
# ─────────────────────────────────────────────────────────────────────────────

class EarlyFusionModel(nn.Module):
    """Design B: Concatenate modalities at input level, shared backbone."""

    def __init__(self, num_disaster_classes=4, num_victim_classes=2, dropout=0.3):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = nn.Sequential(*list(resnet.children())[:-1])
        self.disaster_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_victim_classes),
        )

    def forward(self, rgb, thermal, audio, has_rgb=None, has_thermal=None, has_audio=None):
        B = rgb.size(0)
        audio_img = F.interpolate(audio, size=(224, 224), mode="bilinear", align_corners=False)
        x = torch.cat([rgb, thermal, audio_img], dim=1)
        feat = torch.flatten(self.net(x), 1)
        d = self.disaster_head(feat)
        v = self.victim_head(feat)
        return d, v, torch.ones(B, 3, device=rgb.device) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Design C — Late Fusion
# ─────────────────────────────────────────────────────────────────────────────

class LateFusionModel(nn.Module):
    """Design C: Separate classifiers, combine probabilities with learned weights."""

    def __init__(self, num_disaster_classes=4, num_victim_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        self.rgb_backbone = RGBBackbone(pretrained=pretrained)
        self.thermal_backbone = ThermalBackbone(pretrained=pretrained)
        self.audio_backbone = AudioCNN()

        self.rgb_disaster = nn.Linear(self.rgb_backbone.out_dim, num_disaster_classes)
        self.rgb_victim = nn.Linear(self.rgb_backbone.out_dim, num_victim_classes)
        self.thermal_disaster = nn.Linear(self.thermal_backbone.out_dim, num_disaster_classes)
        self.thermal_victim = nn.Linear(self.thermal_backbone.out_dim, num_victim_classes)
        self.audio_disaster = nn.Linear(self.audio_backbone.out_dim, num_disaster_classes)
        self.audio_victim = nn.Linear(self.audio_backbone.out_dim, num_victim_classes)

        self.log_weights = nn.Parameter(torch.zeros(3))

    def forward(self, rgb, thermal, audio, has_rgb=None, has_thermal=None, has_audio=None):
        B = rgb.size(0)
        device = rgb.device
        if has_rgb is None: has_rgb = torch.ones(B, device=device)
        if has_thermal is None: has_thermal = torch.ones(B, device=device)
        if has_audio is None: has_audio = torch.ones(B, device=device)

        f_rgb = self.rgb_backbone(rgb)
        f_th = self.thermal_backbone(thermal)
        f_au = self.audio_backbone(audio)

        d_rgb = self.rgb_disaster(f_rgb)
        d_th = self.thermal_disaster(f_th)
        d_au = self.audio_disaster(f_au)
        v_rgb = self.rgb_victim(f_rgb)
        v_th = self.thermal_victim(f_th)
        v_au = self.audio_victim(f_au)

        w = F.softmax(self.log_weights, dim=0)
        w_rgb = (w[0] * has_rgb).unsqueeze(-1)
        w_th = (w[1] * has_thermal).unsqueeze(-1)
        w_au = (w[2] * has_audio).unsqueeze(-1)
        w_sum = w_rgb + w_th + w_au + 1e-8
        w_rgb, w_th, w_au = w_rgb/w_sum, w_th/w_sum, w_au/w_sum

        d = w_rgb*d_rgb + w_th*d_th + w_au*d_au
        v = w_rgb*v_rgb + w_th*v_th + w_au*v_au
        rel = torch.stack([w_rgb.squeeze(-1), w_th.squeeze(-1), w_au.squeeze(-1)], dim=1)
        return d, v, rel


# ─────────────────────────────────────────────────────────────────────────────
# Design D — Intermediate Fusion Fixed
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateFusionFixed(nn.Module):
    """Design D: Project all modalities to shared dim, concat, MLP."""

    def __init__(self, num_disaster_classes=4, num_victim_classes=2,
                 proj_dim=192, pretrained=True, dropout=0.3):
        super().__init__()
        self.rgb_enc = RGBBackbone(pretrained=pretrained)
        self.thermal_enc = ThermalBackbone(pretrained=pretrained)
        self.audio_enc = AudioCNN()
        self.rgb_proj = nn.Linear(self.rgb_enc.out_dim, proj_dim)
        self.thermal_proj = nn.Linear(self.thermal_enc.out_dim, proj_dim)
        self.audio_proj = nn.Linear(self.audio_enc.out_dim, proj_dim)
        fused_dim = proj_dim * 3
        self.disaster_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_victim_classes),
        )

    def forward(self, rgb, thermal, audio, has_rgb=None, has_thermal=None, has_audio=None):
        B = rgb.size(0)
        f_rgb = self.rgb_proj(self.rgb_enc(rgb))
        f_th = self.thermal_proj(self.thermal_enc(thermal))
        f_au = self.audio_proj(self.audio_enc(audio))
        if has_rgb is not None: f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        if has_thermal is not None: f_th = f_th * has_thermal.unsqueeze(-1)
        if has_audio is not None: f_au = f_au * has_audio.unsqueeze(-1)
        fused = torch.cat([f_rgb, f_th, f_au], dim=-1)
        return self.disaster_head(fused), self.victim_head(fused), torch.ones(B, 3, device=rgb.device)/3.0


# ─────────────────────────────────────────────────────────────────────────────
# Design E — AdapFuse v1 (FULL — with RUE, quality tokens, NuisanceHead)
# ─────────────────────────────────────────────────────────────────────────────

class AdapFuseV1(nn.Module):
    """
    Design E: AdapFuse v1 — Full proposed method incorporating BOTH plan specifications.

    Novel elements:
      1. Quality tokens from each backbone (16-d side head)
      2. RUE with quality + feature statistics → reliability + uncertainty
      3. Reliability-weighted cross-modal attention
      4. GRU for temporal stability
      5. NuisanceAwareHead with 3 task outputs

    This is the MAIN CONTRIBUTION for Pre-Thesis 2.
    """

    QUALITY_DIM = 16

    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        proj_dim: int = 192,
        num_heads: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        width_mult: float = 1.0,
        n_nuisance: int = 5,
        # Ablation flags (NOW-1)
        use_rue: bool = True,
        use_quality_tokens: bool = True,
        use_cross_attn: bool = True,
        use_gru: bool = True,
        use_nuisance: bool = True,
        # Backbone selection (Tier 4 backbone experiments)
        rgb_backbone: str = "mobilenet_v3_small",
        thermal_backbone: str = "resnet18",
        audio_backbone: str = "audiocnn",
        # EXP-22: spatial alignment attention before GAP
        use_spatial_attn: bool = False,
        # EXP-21: learnable modality domain prompts
        use_domain_prompts: bool = False,
    ):
        super().__init__()
        _proj = int(proj_dim * width_mult)
        _q = self.QUALITY_DIM

        # ── Backbone encoders ─────────────────────────────────────
        # RGB backbone selection
        if rgb_backbone == "efficientnet_b0":
            self.rgb_enc = EfficientNetB0RGBBackbone(
                pretrained=pretrained, use_spatial_attn=use_spatial_attn)
        elif rgb_backbone == "mobilevit_xxs":
            self.rgb_enc = MobileViTXXSBackbone(pretrained=pretrained)
        else:
            self.rgb_enc = RGBBackbone(
                pretrained=pretrained, use_spatial_attn=use_spatial_attn)

        # Thermal backbone selection
        if thermal_backbone == "efficientnet_b0":
            self.thermal_enc = EfficientNetB0ThermalBackbone(
                pretrained=pretrained, use_spatial_attn=use_spatial_attn)
        else:
            self.thermal_enc = ThermalBackbone(
                pretrained=pretrained, use_spatial_attn=use_spatial_attn)

        # Audio backbone selection
        if audio_backbone == "panns_cnn6":
            self.audio_enc = PANNsAudioBackbone(pretrained=pretrained)
        elif audio_backbone == "audiocnn_gd":
            self.audio_enc = AudioCNNGD(width_mult=width_mult)
        else:
            self.audio_enc = AudioCNN(width_mult=width_mult)

        # ── Feature projectors → shared dim ──────────────────────
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.rgb_enc.out_dim, _proj),
            nn.LayerNorm(_proj),
        )
        self.thermal_proj = nn.Sequential(
            nn.Linear(self.thermal_enc.out_dim, _proj),
            nn.LayerNorm(_proj),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_enc.out_dim, _proj),
            nn.LayerNorm(_proj),
        )

        # ── Quality token heads (side MLP from raw backbone features) ──
        self.rgb_quality = nn.Sequential(
            nn.Linear(self.rgb_enc.out_dim, 64), nn.ReLU(),
            nn.Linear(64, _q),
        )
        self.thermal_quality = nn.Sequential(
            nn.Linear(self.thermal_enc.out_dim, 64), nn.ReLU(),
            nn.Linear(64, _q),
        )
        self.audio_quality = nn.Sequential(
            nn.Linear(self.audio_enc.out_dim, 32), nn.ReLU(),
            nn.Linear(32, _q),
        )

        # ── RUE (core novelty from plan 2) ───────────────────────
        self.rue = ReliabilityUncertaintyEstimator(feat_dim=_proj, quality_dim=_q)

        # ── Cross-modal attention ─────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=_proj, num_heads=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.attn_norm = nn.LayerNorm(_proj)
        self.dropout = nn.Dropout(dropout)

        # ── GRU temporal fusion memory ────────────────────────────
        self.gru = nn.GRU(input_size=_proj, hidden_size=_proj, batch_first=True)

        # ── NuisanceAwareHead (2nd novelty) ──────────────────────
        self.head = NuisanceAwareHead(
            feat_dim=_proj,
            n_disaster=num_disaster_classes,
            n_victim=num_victim_classes,
            n_nuisance=n_nuisance,
            dropout=dropout,
        )

        self.proj_dim = _proj

        # Ablation flags
        self.use_rue = use_rue
        self.use_quality_tokens = use_quality_tokens
        self.use_cross_attn = use_cross_attn
        self.use_gru = use_gru
        self.use_nuisance = use_nuisance

        # EXP-21: domain prompt tokens (DPMamba-inspired)
        self.use_domain_prompts = use_domain_prompts
        if use_domain_prompts:
            self.rgb_prompt = nn.Parameter(torch.randn(1, 1, _proj) * 0.02)
            self.thermal_prompt = nn.Parameter(torch.randn(1, 1, _proj) * 0.02)
            self.audio_prompt = nn.Parameter(torch.randn(1, 1, _proj) * 0.02)

    def _encode_all(self, rgb, thermal, audio):
        """Encode all three modalities, return features + quality tokens."""
        # Raw backbone features
        f_rgb_raw = self.rgb_enc(rgb)
        f_th_raw = self.thermal_enc(thermal)
        f_au_raw = self.audio_enc(audio)

        # Projected features
        f_rgb = self.rgb_proj(f_rgb_raw)
        f_th = self.thermal_proj(f_th_raw)
        f_au = self.audio_proj(f_au_raw)

        # Quality tokens
        q_rgb = self.rgb_quality(f_rgb_raw)
        q_th = self.thermal_quality(f_th_raw)
        q_au = self.audio_quality(f_au_raw)

        return (f_rgb, f_th, f_au), (q_rgb, q_th, q_au)

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        audio: torch.Tensor,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (disaster_logits, victim_logits, reliability)
        For compatibility with Trainer which expects this 3-tuple.
        Full outputs (nuisance, uncertainty, etc.) stored in self._last_outputs.
        """
        B = rgb.size(0)
        device = rgb.device

        if has_rgb is None: has_rgb = torch.ones(B, device=device)
        if has_thermal is None: has_thermal = torch.ones(B, device=device)
        if has_audio is None: has_audio = torch.ones(B, device=device)

        # ── Encode ────────────────────────────────────────────────
        (f_rgb, f_th, f_au), (q_rgb, q_th, q_au) = self._encode_all(rgb, thermal, audio)

        # ── Zero-mask missing modalities ──────────────────────────
        f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        f_th = f_th * has_thermal.unsqueeze(-1)
        f_au = f_au * has_audio.unsqueeze(-1)

        # ── RUE: reliability + uncertainty ────────────────────────
        features = {'rgb': f_rgb, 'thermal': f_th, 'audio': f_au}
        quality_tokens = {'rgb': q_rgb, 'thermal': q_th, 'audio': q_au}
        has_modality = {'rgb': has_rgb, 'thermal': has_thermal, 'audio': has_audio}

        if self.use_rue and self.use_quality_tokens:
            reliability, uncertainty, degradation_logits = self.rue(
                features, quality_tokens, has_modality
            )
        elif self.use_rue:
            # RUE uses feature stats only — quality tokens zeroed out
            qt_zeros = {k: torch.zeros_like(v) for k, v in quality_tokens.items()}
            reliability, uncertainty, degradation_logits = self.rue(
                features, qt_zeros, has_modality
            )
        else:
            # No RUE — uniform weights per present modality
            has = torch.stack([has_rgb, has_thermal, has_audio], dim=-1)  # (B, 3)
            reliability = has / (has.sum(dim=-1, keepdim=True) + 1e-8)
            uncertainty = torch.zeros(B, 3, device=device)
            degradation_logits = torch.zeros(B, 3, device=device)

        r_rgb = reliability[:, 0:1]
        r_th = reliability[:, 1:2]
        r_au = reliability[:, 2:3]

        # ── Weight by reliability ─────────────────────────────────
        f_rgb = f_rgb * r_rgb
        f_th = f_th * r_th
        f_au = f_au * r_au

        # ── Cross-modal attention ─────────────────────────────────
        tokens = torch.stack([f_rgb, f_th, f_au], dim=1)   # (B, 3, D)
        if self.use_cross_attn:
            if self.use_domain_prompts:
                # EXP-21: prepend learnable modality domain prompts (DPMamba-inspired)
                # When a modality is missing, its prompt still carries prior signal
                prompts = torch.cat([
                    self.rgb_prompt.expand(B, -1, -1),
                    self.thermal_prompt.expand(B, -1, -1),
                    self.audio_prompt.expand(B, -1, -1),
                ], dim=1)  # (B, 3, D)
                extended = torch.cat([tokens, prompts], dim=1)  # (B, 6, D)
                attended, _ = self.cross_attn(extended, extended, extended)
                attended = attended[:, :3]  # feature token outputs only
            else:
                attended, _ = self.cross_attn(tokens, tokens, tokens)
            attended = self.attn_norm(attended + tokens)     # residual
            fused = attended.mean(dim=1)                     # (B, D)
        else:
            fused = tokens.mean(dim=1)                       # simple mean pool
        fused = self.dropout(fused)

        # ── GRU temporal memory ────────────────────────────────────
        if self.use_gru:
            fused_seq, _ = self.gru(fused.unsqueeze(1))
            fused = fused_seq.squeeze(1)  # (B, D)

        # ── NuisanceAwareHead ─────────────────────────────────────
        head_out = self.head(fused)
        nuisance_out = head_out['nuisance'] if self.use_nuisance else None

        # Cache full outputs for advanced loss computation
        self._last_outputs = {
            'fused_feat': fused,
            'reliability': reliability,
            'uncertainty': uncertainty,
            'degradation_logits': degradation_logits,
            'nuisance': nuisance_out,
        }

        return head_out['disaster'], head_out['victim'], reliability

    def get_full_outputs(self) -> Dict:
        """Return full outputs dict (call after forward)."""
        return getattr(self, '_last_outputs', {})


# ─────────────────────────────────────────────────────────────────────────────
# Design F — AdapFuse v2 Student (for KD)
# ─────────────────────────────────────────────────────────────────────────────

class AdapFuseV2Student(AdapFuseV1):
    """
    Design F: Lightweight student model for knowledge distillation.
    Smaller than teacher via:
      - proj_dim: 96 (vs 192 in teacher) — half-width projections
      - AudioCNN width_mult=0.5 — narrower audio encoder
    RGB and Thermal backbones remain full pretrained ResNet18/MobileNetV3
    (cannot arbitrarily scale their internal channels).
    """

    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        proj_dim: int = 192,
        num_heads: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        width_mult: float = 0.5,
        n_nuisance: int = 5,
        use_rue: bool = True,
        use_quality_tokens: bool = True,
        use_cross_attn: bool = True,
        use_gru: bool = True,
        use_nuisance: bool = True,
        rgb_backbone: str = "mobilenet_v3_small",
        thermal_backbone: str = "resnet18",
        audio_backbone: str = "audiocnn",
        use_spatial_attn: bool = False,
        use_domain_prompts: bool = False,
    ):
        # Student uses smaller proj_dim (e.g. 96 = 192*0.5) and narrower audio
        student_proj = max(32, int(proj_dim * width_mult))
        # num_heads must divide proj_dim
        student_heads = min(num_heads, student_proj // 16)
        student_heads = max(1, student_heads)
        super().__init__(
            num_disaster_classes=num_disaster_classes,
            num_victim_classes=num_victim_classes,
            proj_dim=student_proj,
            num_heads=student_heads,
            pretrained=pretrained,
            dropout=dropout,
            width_mult=width_mult,
            n_nuisance=n_nuisance,
            use_rue=use_rue,
            use_quality_tokens=use_quality_tokens,
            use_cross_attn=use_cross_attn,
            use_gru=use_gru,
            use_nuisance=use_nuisance,
            rgb_backbone=rgb_backbone,
            thermal_backbone=thermal_backbone,
            audio_backbone=audio_backbone,
            use_spatial_attn=use_spatial_attn,
            use_domain_prompts=use_domain_prompts,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry & Factory
# ─────────────────────────────────────────────────────────────────────────────

from models.external_baselines import (
    DeepFireModel, XuCrossAttentionModel,
    FireRGBTNetModel, DynamicRoutingFusionModel,
)

MODEL_REGISTRY = {
    "baseline_rgb": RGBOnlyModel,
    "baseline_thermal": ThermalOnlyModel,
    "baseline_audio": AudioOnlyModel,
    "early_fusion": EarlyFusionModel,
    "late_fusion": LateFusionModel,
    "intermediate_fixed": IntermediateFusionFixed,
    "adaptfuse_v1": AdapFuseV1,
    "adaptfuse_v2_kd": AdapFuseV2Student,
    "deepfire": DeepFireModel,
    "xu_cross_attention": XuCrossAttentionModel,
    "firergbtnet": FireRGBTNetModel,
    "dynamic_routing": DynamicRoutingFusionModel,
}


def build_model(config: dict) -> nn.Module:
    """Build and return model from config dict."""
    model_name = config["model"]
    if model_name == "uni_adaptfuse":
        from models.unified_model import UniAdapFuseModel
        return UniAdapFuseModel(
            num_disaster_classes=config.get("num_classes_disaster", 4),
            num_victim_classes=config.get("num_classes_victim", 2),
            num_det_classes=config.get("num_classes_det", 2),
            proj_dim=config.get("proj_dim", 192),
            det_neck_channels=config.get("det_neck_channels", 128),
            num_heads=config.get("num_heads", 4),
            pretrained=config.get("pretrained", True),
            dropout=config.get("dropout", 0.3),
            detach_det_backbone=config.get("detach_det_backbone", True),
        )

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys()) + ['uni_adaptfuse']}"
        )
    cls = MODEL_REGISTRY[model_name]
    kwargs = {
        "num_disaster_classes": config.get("num_classes_disaster", 4),
        "num_victim_classes": config.get("num_classes_victim", 2),
    }
    if model_name in ("intermediate_fixed", "deepfire", "firergbtnet", "dynamic_routing"):
        kwargs["proj_dim"] = config.get("proj_dim", 192)
    if model_name in ("xu_cross_attention",):
        kwargs["proj_dim"] = config.get("proj_dim", 192)
        kwargs["num_heads"] = config.get("num_heads", 4)
    if model_name in ("adaptfuse_v1", "adaptfuse_v2_kd"):
        kwargs["proj_dim"] = config.get("proj_dim", 192)
        kwargs["num_heads"] = config.get("num_heads", 4)
        kwargs["use_rue"] = config.get("use_rue", True)
        kwargs["use_quality_tokens"] = config.get("use_quality_tokens", True)
        kwargs["use_cross_attn"] = config.get("use_cross_attn", True)
        kwargs["use_gru"] = config.get("use_gru", True)
        kwargs["use_nuisance"] = config.get("use_nuisance", True)
        kwargs["rgb_backbone"] = config.get("rgb_backbone", "mobilenet_v3_small")
        kwargs["thermal_backbone"] = config.get("thermal_backbone", "resnet18")
        kwargs["audio_backbone"] = config.get("audio_backbone", "audiocnn")
        kwargs["use_spatial_attn"] = config.get("use_spatial_attn", False)
        kwargs["use_domain_prompts"] = config.get("use_domain_prompts", False)
    if model_name == "baseline_rgb":
        kwargs["backbone"] = config.get("rgb_backbone", "mobilenet_v3_small")
    if model_name == "baseline_thermal":
        kwargs["backbone"] = config.get("thermal_backbone", "resnet18")
    if model_name == "baseline_audio":
        kwargs["backbone"] = config.get("audio_backbone", "audiocnn")
    if model_name == "adaptfuse_v2_kd":
        kwargs["width_mult"] = config.get("width_multiplier", 0.5)
    return cls(**kwargs)
