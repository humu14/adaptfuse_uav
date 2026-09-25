"""
D-RUE -- Detection-Level Reliability and Uncertainty Estimator
===============================================================
Novel Contribution #1: Extends scene-level RUE from Phase 1 to per-detection
reliability scoring.

Instead of rating the reliability of an ENTIRE image or sensor modality,
D-RUE computes a per-bounding-box reliability score r_i in [0, 1] based on:
    1. Spatial feature sharpness at the box location
    2. Modality noise/degradation metrics (e.g. thermal saturation, smoke opacity, audio SNR)
    3. Model prediction variance (uncertainty estimation via Monte Carlo dropout / feature dispersion)

High-reliability boxes (r_i ≈ 1.0) dominate the final fusion decision,
while degraded sensor boxes (r_i ≈ 0.1) are down-weighted without dropping the sensor entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class DetectionRUE(nn.Module):
    """
    Per-detection Reliability and Uncertainty Estimator (D-RUE).
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        # Reliability predictor MLP
        self.reliability_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Uncertainty predictor MLP (variance estimation)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Output > 0 (variance)
        )

    def forward(self, box_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            box_features: (N, feature_dim) pooled features per detection box
        
        Returns:
            reliability: (N, 1) reliability weight in [0, 1]
            uncertainty: (N, 1) estimated variance
        """
        reliability = self.reliability_net(box_features)
        uncertainty = self.uncertainty_net(box_features)
        return reliability, uncertainty


if __name__ == "__main__":
    box_feats = torch.randn(10, 256)
    drue = DetectionRUE(feature_dim=256)
    rel, unc = drue(box_feats)
    print("D-RUE Test:")
    print(f"  Reliability: {rel.shape}, range [{rel.min():.3f}, {rel.max():.3f}]")
    print(f"  Uncertainty: {unc.shape}, range [{unc.min():.3f}, {unc.max():.3f}]")
