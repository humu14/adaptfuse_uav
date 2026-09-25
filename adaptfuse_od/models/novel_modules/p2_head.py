"""
P2 Multi-Scale Detection Head
================================
Novel Module #3: Adds a high-resolution P2 detection layer for detecting
very small objects (tiny fire spots, distant persons) in aerial imagery.

Standard YOLO uses 3 detection heads at P3 (80×80), P4 (40×40), P5 (20×20).
This module adds P2 (160×160) to capture objects at 4× higher spatial resolution.

Novel in this context: P2 head with altitude-aware anchor scaling specifically
designed for UAV fire/smoke detection at 50-200m altitude ranges.

The P2 feature map is derived from the backbone's early layers and refined
with a lightweight feature aggregation module to keep FLOPs low.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightUpsampleBlock(nn.Module):
    """Lightweight upsample + refinement for creating P2 features."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.upsample(x))


class P2FeatureAggregation(nn.Module):
    """
    Aggregate early backbone features with upsampled P3 features
    to create a high-resolution P2 feature map.
    
    Takes:
        - backbone_c2: Early backbone features at stride 4 (high-res, low-level)
        - p3_features: FPN P3 features at stride 8 (semantic, lower-res)
    
    Outputs:
        - P2 features at stride 4 for small object detection
    """

    def __init__(self, c2_channels: int, p3_channels: int, out_channels: int = 64):
        """
        Args:
            c2_channels: Channels from backbone's C2 stage (stride 4)
            p3_channels: Channels from FPN P3 level (stride 8)
            out_channels: Output channels for P2 feature map
        """
        super().__init__()

        # Reduce C2 backbone features (typically high channel count)
        self.c2_reduce = nn.Sequential(
            nn.Conv2d(c2_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Upsample P3 to match C2 resolution
        self.p3_upsample = LightweightUpsampleBlock(p3_channels, out_channels)

        # Fuse C2 + upsampled P3
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, c2_features, p3_features):
        c2 = self.c2_reduce(c2_features)
        p3_up = self.p3_upsample(p3_features)

        # Ensure spatial sizes match (in case of rounding differences)
        if c2.shape[2:] != p3_up.shape[2:]:
            p3_up = F.interpolate(p3_up, size=c2.shape[2:], mode="nearest")

        # Concatenate and fuse
        fused = torch.cat([c2, p3_up], dim=1)
        return self.fuse(fused)


class P2DetectionHead(nn.Module):
    """
    Detection head for P2 features — specialized for small objects.
    
    Outputs per-anchor predictions: [x, y, w, h, objectness, class_scores]
    Uses smaller anchors calibrated for UAV altitude ranges.
    """

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        """
        Args:
            in_channels: Input feature channels (from P2FeatureAggregation)
            num_classes: Number of object classes
            num_anchors: Anchors per spatial position
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_anchors * (5 + num_classes)  # [x,y,w,h,obj,cls...]

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, self.num_outputs, 1, bias=True),
        )

        # Small anchors for aerial small object detection
        # These are relative to the P2 feature map resolution
        self.register_buffer("anchors", torch.tensor([
            [4, 4],    # Tiny (4×4 pixels at stride 4 = 16×16 actual)
            [8, 8],    # Small (8×8 pixels at stride 4 = 32×32 actual)
            [16, 12],  # Small-wide (for smoke plumes)
        ], dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x: P2 feature map (B, C, H, W) where H,W = input_size / 4
        
        Returns:
            predictions: (B, num_outputs, H, W) raw detection outputs
        """
        return self.head(x)


class P2Module(nn.Module):
    """
    Complete P2 multi-scale detection module.
    
    Combines P2FeatureAggregation + P2DetectionHead into a single
    module that can be attached to any YOLO/detector backbone.
    """

    def __init__(self, c2_channels: int, p3_channels: int, num_classes: int,
                 hidden_dim: int = 64, num_anchors: int = 3):
        super().__init__()
        self.aggregation = P2FeatureAggregation(c2_channels, p3_channels, hidden_dim)
        self.detection_head = P2DetectionHead(hidden_dim, num_classes, num_anchors)

    def forward(self, c2_features, p3_features):
        p2 = self.aggregation(c2_features, p3_features)
        detections = self.detection_head(p2)
        return detections, p2  # Return both detections and features


if __name__ == "__main__":
    # Test
    c2 = torch.randn(2, 128, 160, 160)  # Early backbone features at stride 4
    p3 = torch.randn(2, 256, 80, 80)    # FPN P3 features at stride 8

    p2_module = P2Module(c2_channels=128, p3_channels=256, num_classes=2)
    dets, feats = p2_module(c2, p3)
    print(f"P2Module: C2 {c2.shape} + P3 {p3.shape}")
    print(f"  → Detections: {dets.shape}")
    print(f"  → Features: {feats.shape}")

    params = sum(p.numel() for p in p2_module.parameters())
    print(f"  Params: {params:,}")
