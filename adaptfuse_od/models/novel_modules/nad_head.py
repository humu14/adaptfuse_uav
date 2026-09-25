"""
NAD-Head — Nuisance-Aware Detection Head
==========================================
Novel Module #4: Extends standard detection heads with per-bounding-box
nuisance classification. Each detection carries both a class prediction
AND a nuisance flag indicating whether it's a genuine detection or a
known false alarm type.

This is a direct extension of Phase 1's NuisanceAwareHead from
classification to object detection — NO prior work has per-detection
nuisance reasoning in fire/smoke detectors.

Nuisance classes:
    0: genuine       — real fire/smoke/person
    1: solar_reflect — solar heating / reflection false alarm
    2: red_object    — red/orange object confused for fire
    3: cloud_smoke   — cloud/fog/haze confused for smoke  
    4: steam         — steam/vapor confused for smoke
    5: background    — other background artifact

During training, the nuisance branch receives auxiliary supervision.
At inference, detections with high nuisance scores can be:
    (a) Suppressed entirely (hard filtering)
    (b) Down-weighted in confidence (soft filtering)
    (c) Flagged for human review (operational mode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NuisanceClassifier(nn.Module):
    """Per-detection nuisance type classifier."""

    def __init__(self, in_features: int, num_nuisance_classes: int = 6,
                 hidden_dim: int = 64):
        """
        Args:
            in_features: Detection feature dimension (from ROI features or anchor features)
            num_nuisance_classes: Number of nuisance categories
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_nuisance_classes),
        )

    def forward(self, features):
        """
        Args:
            features: (N, in_features) — pooled features per detection
        
        Returns:
            nuisance_logits: (N, num_nuisance_classes)
        """
        return self.classifier(features)


class NADHead(nn.Module):
    """
    Nuisance-Aware Detection Head.
    
    Wraps around the detection features to add nuisance classification.
    Can be attached to any detector's output features.
    """

    def __init__(self, in_channels: int, num_classes: int, num_nuisance_classes: int = 6,
                 hidden_dim: int = 64, suppression_mode: str = "soft"):
        """
        Args:
            in_channels: Input feature channels (from FPN or detection head features)
            num_classes: Number of detection classes (fire, smoke, person)
            num_nuisance_classes: Number of nuisance types
            hidden_dim: Hidden dimension for nuisance classifier
            suppression_mode: How to handle nuisances — 'soft', 'hard', or 'flag'
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_nuisance_classes = num_nuisance_classes
        self.suppression_mode = suppression_mode

        # Feature pooling for spatial features → vector
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Nuisance classifier
        self.nuisance_head = NuisanceClassifier(
            in_features=in_channels,
            num_nuisance_classes=num_nuisance_classes,
            hidden_dim=hidden_dim,
        )

        # Confidence recalibration based on nuisance score
        self.recalibrator = nn.Sequential(
            nn.Linear(num_nuisance_classes + num_classes, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Nuisance class names for interpretability
        self.nuisance_names = [
            "genuine", "solar_reflection", "red_object",
            "cloud_smoke", "steam", "background",
        ]

    def forward(self, features, detection_logits=None):
        """
        Args:
            features: (B, C, H, W) feature map or (N, C) per-detection features
            detection_logits: (N, num_classes) existing class predictions (optional)
        
        Returns:
            dict with:
                - nuisance_logits: (N, num_nuisance_classes)
                - nuisance_probs: (N, num_nuisance_classes) softmax probabilities
                - genuineness: (N, 1) probability that detection is genuine
                - recalibrated_conf: (N, 1) adjusted confidence (if detection_logits provided)
        """
        # Handle spatial features
        if features.dim() == 4:
            B, C, H, W = features.shape
            pooled = self.pool(features).view(B, C)
        elif features.dim() == 2:
            pooled = features
        else:
            raise ValueError(f"Expected 2D or 4D features, got {features.dim()}D")

        # Predict nuisance type
        nuisance_logits = self.nuisance_head(pooled)
        nuisance_probs = F.softmax(nuisance_logits, dim=-1)

        # Genuineness = probability of class 0 (genuine)
        genuineness = nuisance_probs[:, 0:1]

        result = {
            "nuisance_logits": nuisance_logits,
            "nuisance_probs": nuisance_probs,
            "genuineness": genuineness,
        }

        # Recalibrate confidence if detection logits are available
        if detection_logits is not None:
            combined = torch.cat([nuisance_probs, detection_logits], dim=-1)
            recal_factor = self.recalibrator(combined)
            result["recalibrated_conf"] = recal_factor

        return result

    def compute_loss(self, nuisance_logits, nuisance_targets, weight=None):
        """
        Compute nuisance classification loss.
        
        Args:
            nuisance_logits: (N, num_nuisance_classes)
            nuisance_targets: (N,) integer class labels
            weight: Optional per-class weights
        
        Returns:
            Cross-entropy loss
        """
        return F.cross_entropy(nuisance_logits, nuisance_targets, weight=weight)

    def suppress_detections(self, detections, nuisance_probs, threshold=0.5):
        """
        Apply nuisance suppression to detections.
        
        Args:
            detections: Dict with 'boxes', 'scores', 'labels'
            nuisance_probs: (N, num_nuisance_classes)
            threshold: Genuineness threshold
        
        Returns:
            Filtered detections
        """
        genuineness = nuisance_probs[:, 0]  # P(genuine)

        if self.suppression_mode == "hard":
            keep = genuineness >= threshold
            return {k: v[keep] for k, v in detections.items()}

        elif self.suppression_mode == "soft":
            detections["scores"] = detections["scores"] * genuineness
            return detections

        else:  # "flag"
            detections["nuisance_flags"] = (genuineness < threshold)
            detections["genuineness"] = genuineness
            return detections


if __name__ == "__main__":
    # Test
    features = torch.randn(10, 256)  # 10 detections, 256-dim features
    det_logits = torch.randn(10, 2)  # 2 classes (fire, smoke)

    nad = NADHead(in_channels=256, num_classes=2, num_nuisance_classes=6)
    result = nad(features, det_logits)

    print(f"NADHead test:")
    print(f"  Nuisance logits: {result['nuisance_logits'].shape}")
    print(f"  Nuisance probs: {result['nuisance_probs'].shape}")
    print(f"  Genuineness: {result['genuineness'].shape}")
    print(f"  Recalibrated conf: {result['recalibrated_conf'].shape}")

    params = sum(p.numel() for p in nad.parameters())
    print(f"  Params: {params:,}")
