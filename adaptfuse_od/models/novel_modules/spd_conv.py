"""
SPD-Conv — Space-to-Depth Convolution
=======================================
Novel Module #1: Replaces strided convolutions to preserve fine-grained
spatial information for small object detection (fire spots, distant smoke).

Standard strided convolution loses spatial detail by skipping pixels.
SPD-Conv rearranges spatial dimensions into channel depth BEFORE convolution,
preserving ALL spatial information while still reducing resolution.

Reference: "No More Strided Convolutions or Pooling" (Sunkara & Luo, 2022)
Novel in this context: First application to aerial fire/smoke detection
combined with reliability-aware weighting.

Example:
    Input:  (B, C, H, W)  →  Space-to-Depth  →  (B, 4C, H/2, W/2)
                              →  Conv 1×1       →  (B, C', H/2, W/2)
"""

import torch
import torch.nn as nn


class SpaceToDepth(nn.Module):
    """Rearrange spatial blocks into channel dimension."""

    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.block_size
        # Reshape: (B, C, H, W) → (B, C, H/s, s, W/s, s) → (B, C*s*s, H/s, W/s)
        x = x.reshape(B, C, H // s, s, W // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C * s * s, H // s, W // s)
        return x


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution block.
    
    Replaces a strided convolution with:
        1. Space-to-Depth rearrangement (no information loss)
        2. 1×1 convolution to reduce channels back
        3. BatchNorm + SiLU activation
    
    This preserves spatial features that strided convolutions destroy,
    which is critical for detecting small fire spots from aerial altitude.
    """

    def __init__(self, in_channels: int, out_channels: int, block_size: int = 2):
        """
        Args:
            in_channels: Input channels before SPD
            out_channels: Output channels after 1×1 conv
            block_size: Spatial downsampling factor (2 = halve resolution)
        """
        super().__init__()
        expanded_channels = in_channels * block_size * block_size

        self.spd = SpaceToDepth(block_size)
        self.conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.spd(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SPDBlock(nn.Module):
    """
    SPD-Conv block with optional residual connection.
    Can be used as a drop-in replacement for strided conv blocks
    in YOLO backbones and necks.
    """

    def __init__(self, in_channels: int, out_channels: int, block_size: int = 2,
                 use_residual: bool = False):
        super().__init__()
        self.spd_conv = SPDConv(in_channels, out_channels, block_size)
        self.use_residual = use_residual and (in_channels == out_channels)

        if self.use_residual:
            # Downsample residual path with avg pooling
            self.residual_pool = nn.AvgPool2d(block_size)

    def forward(self, x):
        out = self.spd_conv(x)
        if self.use_residual:
            out = out + self.residual_pool(x)
        return out


if __name__ == "__main__":
    # Test
    x = torch.randn(2, 64, 80, 80)
    spd = SPDConv(64, 128, block_size=2)
    out = spd(x)
    print(f"SPDConv: {x.shape} → {out.shape}")
    # Expected: (2, 64, 80, 80) → (2, 128, 40, 40)

    spd_block = SPDBlock(64, 64, block_size=2, use_residual=True)
    out2 = spd_block(x)
    print(f"SPDBlock (residual): {x.shape} → {out2.shape}")

    params = sum(p.numel() for p in spd.parameters())
    print(f"SPDConv params: {params:,}")
