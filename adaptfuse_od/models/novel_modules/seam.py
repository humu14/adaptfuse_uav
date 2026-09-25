"""
SEAM — Suppression-Enhanced Attention Module
==============================================
Novel Module #2: Suppresses background noise patterns that resemble
smoke/fire (clouds, fog, haze, solar reflections) while enhancing
true fire/smoke features.

SEAM combines:
    1. Spatial attention: Learns WHERE to look (fire regions vs background)
    2. Channel attention: Learns WHICH features are discriminative
    3. Suppression gate: Explicitly learns to DOWN-weight confusing patterns

Novel in this context: First combination of SEAM with nuisance-aware
detection for aerial fire/smoke, where false alarms from clouds and
reflections are a major problem.

Architecture:
    Input features → Channel Attention → Spatial Attention → 
    Suppression Gate → Enhanced features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        attn = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention using channel statistics."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class SuppressionGate(nn.Module):
    """
    Learned suppression gate that identifies and down-weights
    confusing patterns (clouds, fog, solar reflections).
    
    This is the novel component — it learns a "nuisance mask" that
    suppresses features at spatial locations likely to cause false alarms.
    """

    def __init__(self, channels: int, num_suppression_patterns: int = 4):
        """
        Args:
            channels: Number of input channels
            num_suppression_patterns: How many nuisance patterns to learn
                (e.g., cloud, fog, solar reflection, haze)
        """
        super().__init__()
        self.num_patterns = num_suppression_patterns

        # Learn nuisance pattern detectors
        self.pattern_detectors = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_suppression_patterns, 1, bias=True),
        )

        # Combine pattern activations into a suppression mask
        self.gate = nn.Sequential(
            nn.Conv2d(num_suppression_patterns, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Detect nuisance patterns
        patterns = self.pattern_detectors(x)  # (B, num_patterns, H, W)

        # Generate suppression mask: high = keep, low = suppress
        suppression_mask = self.gate(patterns)  # (B, 1, H, W)

        # Apply: multiply features by (1 - suppression) to suppress nuisances
        # Or equivalently, keep_mask = 1 where genuine, 0 where nuisance
        return x * suppression_mask, patterns


class SEAM(nn.Module):
    """
    Suppression-Enhanced Attention Module.
    
    Combines channel attention, spatial attention, and a learned
    suppression gate to enhance fire/smoke features while explicitly
    suppressing confusing background patterns.
    """

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7, num_suppression_patterns: int = 4,
                 use_residual: bool = True):
        """
        Args:
            channels: Number of input/output channels
            reduction: Channel attention reduction ratio
            spatial_kernel: Spatial attention kernel size
            num_suppression_patterns: Number of nuisance patterns to learn
            use_residual: Add residual connection
        """
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)
        self.suppression_gate = SuppressionGate(channels, num_suppression_patterns)
        self.use_residual = use_residual

    def forward(self, x):
        """
        Returns:
            enhanced: Attention-enhanced features with nuisance suppression
            nuisance_patterns: (B, num_patterns, H, W) — auxiliary output
                for nuisance-aware training supervision
        """
        identity = x

        # Step 1: Channel attention — which channels matter
        out = self.channel_attn(x)

        # Step 2: Spatial attention — where to focus
        out = self.spatial_attn(out)

        # Step 3: Suppression gate — suppress nuisance regions
        out, nuisance_patterns = self.suppression_gate(out)

        # Residual connection
        if self.use_residual:
            out = out + identity

        return out, nuisance_patterns


if __name__ == "__main__":
    # Test
    x = torch.randn(2, 256, 20, 20)
    seam = SEAM(256, num_suppression_patterns=4)
    out, patterns = seam(x)
    print(f"SEAM: {x.shape} → {out.shape}")
    print(f"Nuisance patterns: {patterns.shape}")

    params = sum(p.numel() for p in seam.parameters())
    print(f"SEAM params: {params:,}")
