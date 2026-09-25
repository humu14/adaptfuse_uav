"""
Dynamic Physics-Informed Reliability & Uncertainty Estimator (Dynamic-RUE)
==========================================================================
Addresses the static reliability limitation by extracting differentiable
pre-backbone signal/physics metrics alongside post-backbone feature statistics
and quality tokens.

Pre-Backbone Metrics:
  - RGB: Smoke/fog density (dark channel), sharpness (Laplacian variance), illumination mean/std.
  - Thermal: Dynamic range (max-min), hotspot contrast (p99 - median), gradient energy.
  - Audio: Spectral flatness, rotor vs high-frequency energy ratio, power dynamics.

Outputs:
  - reliability: (B, 3) in [0, 1] - dynamic per-sample weights for RGB, Thermal, Audio.
  - uncertainty: (B, 3) >= 0 - epistemic/aleatoric uncertainty estimates.
  - degradation_logits: (B, 3) - predicted corruption presence for supervised multi-task loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PreBackboneDegradationExtractor(nn.Module):
    """
    Extracts differentiable physical degradation statistics directly from raw sensor inputs.
    """

    def __init__(self):
        super().__init__()
        # 3x3 Laplacian kernel for sharpness estimation
        laplacian = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian)

    def extract_rgb_stats(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B, 3, H, W) in [0, 1] or normalized
        Returns: (B, 4) [dark_channel_mean, sharpness, lum_mean, lum_std]
        """
        B, C, H, W = rgb.shape
        # 1. Dark channel (smoke/haze indicator)
        # Rescale or clamp to positive
        rgb_pos = torch.relu(rgb)
        dark_ch, _ = torch.min(rgb_pos, dim=1, keepdim=True)  # (B, 1, H, W)
        dark_mean = dark_ch.mean(dim=[-2, -1])  # (B, 1)

        # 2. Sharpness via Laplacian filter on grayscale
        gray = rgb.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        edges = F.conv2d(gray, self.laplacian_kernel, padding=1)
        sharpness = edges.var(dim=[-2, -1]).clamp(min=1e-6)  # (B, 1)
        sharpness_log = torch.log(sharpness + 1.0)

        # 3. Illumination mean and std (differentiable without sqrt(0) instability)
        lum_mean = gray.mean(dim=[-2, -1])  # (B, 1)
        lum_std = torch.sqrt(gray.var(dim=[-2, -1]).clamp(min=1e-6))  # (B, 1)

        return torch.cat([dark_mean, sharpness_log, lum_mean, lum_std], dim=-1)

    def extract_thermal_stats(self, thermal: torch.Tensor) -> torch.Tensor:
        """
        thermal: (B, 1, H, W) or (B, 3, H, W)
        Returns: (B, 3) [dynamic_range, hotspot_prominence, th_std]
        """
        if thermal.size(1) > 1:
            th = thermal.mean(dim=1, keepdim=True)
        else:
            th = thermal
        
        # 1. Dynamic range
        th_max = th.amax(dim=[-2, -1])  # (B, 1)
        th_min = th.amin(dim=[-2, -1])  # (B, 1)
        dyn_range = (th_max - th_min).clamp(min=0.0)

        # 2. Hotspot contrast (approximate top 5% vs mean)
        th_mean = th.mean(dim=[-2, -1])
        th_std = torch.sqrt(th.var(dim=[-2, -1]).clamp(min=1e-6))
        hotspot_prominence = (th_max - th_mean).clamp(min=0.0)

        return torch.cat([dyn_range, hotspot_prominence, th_std], dim=-1)

    def extract_audio_stats(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, 1, n_mels, time) log-mel spectrogram
        Returns: (B, 3) [spectral_flatness, high_low_energy_ratio, audio_std]
        """
        audio_clamped = audio.clamp(min=-20.0, max=10.0)
        # Convert log-mel to linear-like magnitude power
        power = torch.exp(audio_clamped)  # (B, 1, F, T)
        
        # 1. Spectral Flatness = exp(mean(log(p))) / mean(p)
        log_mean = audio_clamped.mean(dim=[-2, -1])  # (B, 1)
        arith_mean = power.mean(dim=[-2, -1]).clamp(min=1e-6)  # (B, 1)
        flatness = (torch.exp(log_mean) / arith_mean).clamp(0.0, 1.0)

        # 2. High vs Low frequency energy ratio
        # Rotor noise is concentrated in lower mel bins; disaster acoustics span broader spectrum
        n_mels = audio.size(2)
        mid_bin = n_mels // 2
        low_energy = power[:, :, :mid_bin, :].mean(dim=[-2, -1]).clamp(min=1e-6)
        high_energy = power[:, :, mid_bin:, :].mean(dim=[-2, -1]).clamp(min=1e-6)
        ratio = (high_energy / low_energy).clamp(0.0, 10.0)

        audio_std = torch.sqrt(audio_clamped.var(dim=[-2, -1]).clamp(min=1e-6))

        return torch.cat([flatness, ratio, audio_std], dim=-1)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'rgb_phys': self.extract_rgb_stats(rgb),
            'thermal_phys': self.extract_thermal_stats(thermal),
            'audio_phys': self.extract_audio_stats(audio),
        }


class DynamicPhysicsRUE(nn.Module):
    """
    Novel Dual-Domain Dynamic Reliability & Uncertainty Estimator.
    Combines:
      1. Pre-backbone physical quality signals (smoke density, sharpness, SNR, contrast).
      2. Post-backbone quality tokens (learned 16-d representation from side heads).
      3. Deep feature norm and variance.
    """

    def __init__(self, feat_dim: int = 192, quality_dim: int = 16):
        super().__init__()
        self.phys_extractor = PreBackboneDegradationExtractor()
        
        # Stats per modality:
        # RGB: phys(4) + quality(16) + norm(1) + var(1) = 22
        # Thermal: phys(3) + quality(16) + norm(1) + var(1) = 21
        # Audio: phys(3) + quality(16) + norm(1) + var(1) = 21
        # Total in_dim = 22 + 21 + 21 = 64
        in_dim = (4 + quality_dim + 2) + (3 + quality_dim + 2) + (3 + quality_dim + 2)

        self.reliability_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),    # r_rgb, r_thermal, r_audio
            nn.Sigmoid(),
        )

        self.uncertainty_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),    # u_rgb, u_thermal, u_audio
            nn.Softplus(),       # ensures u >= 0
        )

        # Auxiliary degradation classifier (predicts if each modality is corrupted)
        self.degradation_net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),    # degradation logits: [deg_rgb, deg_th, deg_au]
        )

    def forward(
        self,
        rgb_raw: torch.Tensor,
        thermal_raw: torch.Tensor,
        audio_raw: torch.Tensor,
        features: Dict[str, torch.Tensor],
        quality_tokens: Dict[str, torch.Tensor],
        has_modality: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_raw: (B, 3, H, W)
            thermal_raw: (B, 1, H, W) or (B, 3, H, W)
            audio_raw: (B, 1, n_mels, T)
            features: {'rgb': (B,D), 'thermal': (B,D), 'audio': (B,D)}
            quality_tokens: {'rgb': (B,Q), 'thermal': (B,Q), 'audio': (B,Q)}
            has_modality: {'rgb': (B,), 'thermal': (B,), 'audio': (B,)}

        Returns:
            reliability: (B, 3) in [0, 1]
            uncertainty: (B, 3) >= 0
            degradation_logits: (B, 3)
        """
        # 1. Physics signals from raw inputs
        phys = self.phys_extractor(rgb_raw, thermal_raw, audio_raw)

        # 2. Assemble multi-domain statistics
        stats = []
        for key in ['rgb', 'thermal', 'audio']:
            p = phys[f"{key}_phys"]                              # (B, phys_dim)
            q = quality_tokens[key]                              # (B, Q)
            f = features[key]                                    # (B, D)
            feat_norm = f.norm(dim=-1, keepdim=True)             # (B, 1)
            feat_var = f.var(dim=-1, keepdim=True).clamp(0, 10)  # (B, 1)
            stats.append(torch.cat([p, q, feat_norm, feat_var], dim=-1))

        combined = torch.cat(stats, dim=-1)  # (B, 64)

        # 3. Predict reliability, uncertainty, and degradation
        reliability = self.reliability_net(combined)
        uncertainty = self.uncertainty_net(combined)
        degradation_logits = self.degradation_net(combined)

        # 4. Modality presence masking (missing sensor -> 0 reliability)
        has = torch.stack([
            has_modality['rgb'],
            has_modality['thermal'],
            has_modality['audio'],
        ], dim=-1)  # (B, 3)

        reliability = reliability * has

        return reliability, uncertainty, degradation_logits
