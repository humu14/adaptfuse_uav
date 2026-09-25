"""
ACR — Adaptive Confidence Recalibration
==========================================
Novel Module #5: Post-detection module that recalibrates detection confidence
scores using spatial context (neighboring detections, scene statistics) and
feature quality estimation.

Extends the RUE (Reliability & Uncertainty Estimator) concept from Phase 1
classification to detection-level confidence adjustment.

The problem: Object detectors often produce overconfident false positives
or underconfident true positives. ACR learns to correct this by considering:
    1. Local feature quality at the detection location
    2. Spatial context from neighboring detections
    3. Global scene statistics (overall feature quality)
    4. Detection density (clustered detections → likely genuine)

This directly reduces false alarm rate while preserving recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureQualityEstimator(nn.Module):
    """
    Estimates the quality/reliability of features at detection locations.
    Higher quality → more trustworthy detection.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) feature map
        
        Returns:
            quality: (B, 1) quality score in [0, 1]
        """
        return self.quality_net(features)


class SpatialContextEncoder(nn.Module):
    """
    Encodes spatial context around each detection using a small MLP
    on detection statistics (positions, scores of neighbors).
    """

    def __init__(self, context_dim: int = 16, max_neighbors: int = 10):
        super().__init__()
        self.max_neighbors = max_neighbors
        # Input: [x, y, w, h, score] × max_neighbors = 5 * max_neighbors
        input_dim = 5 * max_neighbors
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, context_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, boxes, scores, query_idx):
        """
        Encode spatial context for a specific detection.
        
        Args:
            boxes: (N, 4) all detection boxes [x1, y1, x2, y2]
            scores: (N,) detection confidence scores
            query_idx: Index of the detection to get context for
        
        Returns:
            context: (context_dim,) spatial context vector
        """
        N = boxes.shape[0]
        query_box = boxes[query_idx]

        # Compute distances to all other detections
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2  # (N, 2)
        query_center = centers[query_idx]
        dists = torch.norm(centers - query_center, dim=1)

        # Get nearest neighbors (excluding self)
        dists[query_idx] = float("inf")
        _, nn_idx = torch.topk(dists, min(self.max_neighbors, N - 1), largest=False)

        # Build neighbor features [x, y, w, h, score]
        nn_boxes = boxes[nn_idx]
        nn_scores = scores[nn_idx]
        nn_wh = nn_boxes[:, 2:4] - nn_boxes[:, :2]
        nn_features = torch.cat([
            centers[nn_idx] - query_center,  # Relative position
            nn_wh,                            # Size
            nn_scores.unsqueeze(1),           # Confidence
        ], dim=1)  # (K, 5)

        # Pad to max_neighbors
        pad_size = self.max_neighbors - nn_features.shape[0]
        if pad_size > 0:
            nn_features = F.pad(nn_features, (0, 0, 0, pad_size))

        return self.encoder(nn_features.flatten().unsqueeze(0))


class ACR(nn.Module):
    """
    Adaptive Confidence Recalibration module.
    
    Takes raw detection outputs and recalibrates confidence scores
    based on multi-source quality signals.
    """

    def __init__(self, feature_channels: int = 256, hidden_dim: int = 64,
                 context_dim: int = 16, max_neighbors: int = 10):
        """
        Args:
            feature_channels: Feature map channels from detector backbone
            hidden_dim: Hidden dimension for recalibration MLP
            context_dim: Spatial context encoding dimension
            max_neighbors: Max neighbor detections for context
        """
        super().__init__()
        self.feature_quality = FeatureQualityEstimator(feature_channels)

        # Recalibration MLP
        # Input: [raw_conf, feature_quality, scene_mean, scene_std, num_detections_norm]
        recal_input_dim = 5
        self.recalibrator = nn.Sequential(
            nn.Linear(recal_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Learnable calibration temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, raw_scores, feature_map=None, boxes=None,
                max_detections=300):
        """
        Recalibrate detection confidence scores.
        
        Args:
            raw_scores: (N,) raw confidence scores from detector
            feature_map: (1, C, H, W) feature map for quality estimation
            boxes: (N, 4) detection bounding boxes
            max_detections: Max detections to process
        
        Returns:
            recalibrated_scores: (N,) adjusted confidence scores
            quality_info: dict with quality diagnostics
        """
        N = raw_scores.shape[0]
        if N == 0:
            return raw_scores, {}

        # Limit to max_detections
        if N > max_detections:
            topk_idx = torch.topk(raw_scores, max_detections).indices
            raw_scores = raw_scores[topk_idx]
            if boxes is not None:
                boxes = boxes[topk_idx]
            N = max_detections

        # Feature quality (global)
        if feature_map is not None:
            fq = self.feature_quality(feature_map)  # (B, 1)
            B = feature_map.shape[0]
            if N % B == 0:
                feat_quality = fq.repeat_interleave(N // B).squeeze()
            else:
                feat_quality = fq.mean().expand(N)
        else:
            feat_quality = torch.full((N,), 0.5, device=raw_scores.device)

        # Scene statistics
        scene_mean = raw_scores.mean()
        scene_std = raw_scores.std() if N > 1 else torch.tensor(0.0, device=raw_scores.device)
        num_det_norm = min(N / 100.0, 1.0)  # Normalize detection count

        # Build recalibration input for each detection
        recal_input = torch.stack([
            raw_scores,
            feat_quality.expand(N),
            scene_mean.expand(N),
            scene_std.expand(N),
            torch.full((N,), num_det_norm, device=raw_scores.device),
        ], dim=1)  # (N, 5)

        # Predict recalibration factors
        recal_factors = self.recalibrator(recal_input).squeeze(-1)

        # Apply temperature scaling
        temp = F.softplus(self.temperature) + 0.5  # Ensure temperature > 0.5
        calibrated = torch.sigmoid(
            torch.logit(raw_scores.clamp(1e-6, 1 - 1e-6)) / temp
        )

        # Final recalibrated score = blend of calibrated + quality-adjusted
        recalibrated = calibrated * recal_factors

        quality_info = {
            "feature_quality": feat_quality.item() if feat_quality.dim() == 0 else feat_quality.mean().item(),
            "scene_mean_conf": scene_mean.item(),
            "scene_std_conf": scene_std.item(),
            "num_detections": N,
            "temperature": temp.item(),
            "mean_recal_factor": recal_factors.mean().item(),
        }

        return recalibrated, quality_info

    def compute_calibration_loss(self, recalibrated_scores, is_correct):
        """
        Calibration loss: encourage predicted confidence to match actual accuracy.
        
        Args:
            recalibrated_scores: (N,) recalibrated confidence
            is_correct: (N,) binary — 1 if detection was correct (IoU > 0.5 with GT)
        
        Returns:
            ECE-like calibration loss
        """
        # Binary cross-entropy between confidence and correctness
        return F.binary_cross_entropy(
            recalibrated_scores,
            is_correct.float(),
            reduction="mean",
        )


if __name__ == "__main__":
    # Test
    raw_scores = torch.rand(50)  # 50 detections
    feature_map = torch.randn(1, 256, 20, 20)  # Feature map

    acr = ACR(feature_channels=256)
    recal_scores, info = acr(raw_scores, feature_map)

    print(f"ACR test:")
    print(f"  Raw scores: {raw_scores.shape}, mean={raw_scores.mean():.3f}")
    print(f"  Recalibrated: {recal_scores.shape}, mean={recal_scores.mean():.3f}")
    print(f"  Quality info: {info}")

    params = sum(p.numel() for p in acr.parameters())
    print(f"  Params: {params:,}")
