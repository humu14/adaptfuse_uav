"""
Modified Detector — Combines Baseline Detector with 5 Novel Modules
===================================================================
Phase 2 Contribution: Takes the winning benchmark model architecture
(or a lightweight CSPDarknet/ResNet/timm backbone) and integrates all 5
novel components:

    1. SPD-Conv: Space-to-Depth Downsampling (preserves small object spatial features)
    2. SEAM: Suppression-Enhanced Attention Module (suppresses clouds/glare)
    3. P2-Head: Multi-Scale Stride-4 Detection Head (small object localization)
    4. NAD-Head: Nuisance-Aware Detection Head (per-box false alarm classification)
    5. ACR: Adaptive Confidence Recalibration (feature-quality & scene-aware confidence)

Architecture Flow:
    Input (B, 3, H, W)
      ↓
    Backbone with SPD-Conv downsampling stages
      ↓ C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32)
    SEAM Modules (applied at C3, C4, C5 to suppress background nuisances)
      ↓
    FPN/PANet Neck
      ↓ P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)
    Multi-Scale Detection Heads (P2, P3, P4, P5)
      ↓
    NAD-Head (Per-box nuisance score & genuineness classification)
      ↓
    ACR Module (Recalibrates detection confidences using feature quality & context)
      ↓
    Final Detections [boxes, class_scores, recalibrated_conf, genuineness, nuisance_probs]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .spd_conv import SPDConv, SPDBlock
from .seam import SEAM
from .p2_head import P2Module
from .nad_head import NADHead
from .acr import ACR


class ModifiedBackboneWithSPD(nn.Module):
    """
    Lightweight CSP-style Backbone with SPD-Conv downsampling
    and SEAM attention modules integrated at each feature stage.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 use_spd: bool = True, use_seam: bool = True):
        super().__init__()
        self.use_spd = use_spd
        self.use_seam = use_seam

        # Stage 1: Stem (Stride 2) -> C1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        # Stage 2: C2 (Stride 4)
        c2_ch = base_channels * 2
        if use_spd:
            self.down2 = SPDConv(base_channels, c2_ch)
        else:
            self.down2 = nn.Sequential(
                nn.Conv2d(base_channels, c2_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2_ch),
                nn.SiLU(inplace=True),
            )

        # Stage 3: C3 (Stride 8)
        c3_ch = base_channels * 4
        if use_spd:
            self.down3 = SPDConv(c2_ch, c3_ch)
        else:
            self.down3 = nn.Sequential(
                nn.Conv2d(c2_ch, c3_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c3_ch),
                nn.SiLU(inplace=True),
            )
        self.seam3 = SEAM(c3_ch) if use_seam else nn.Identity()

        # Stage 4: C4 (Stride 16)
        c4_ch = base_channels * 8
        if use_spd:
            self.down4 = SPDConv(c3_ch, c4_ch)
        else:
            self.down4 = nn.Sequential(
                nn.Conv2d(c3_ch, c4_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c4_ch),
                nn.SiLU(inplace=True),
            )
        self.seam4 = SEAM(c4_ch) if use_seam else nn.Identity()

        # Stage 5: C5 (Stride 32)
        c5_ch = base_channels * 16
        if use_spd:
            self.down5 = SPDConv(c4_ch, c5_ch)
        else:
            self.down5 = nn.Sequential(
                nn.Conv2d(c4_ch, c5_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c5_ch),
                nn.SiLU(inplace=True),
            )
        self.seam5 = SEAM(c5_ch) if use_seam else nn.Identity()

        self.out_channels = {
            "c2": c2_ch,
            "c3": c3_ch,
            "c4": c4_ch,
            "c5": c5_ch,
        }

    def forward(self, x) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        c1 = self.stem(x)
        c2 = self.down2(c1)

        c3 = self.down3(c2)
        c3_out, seam3_patterns = self.seam3(c3) if self.use_seam else (c3, None)

        c4 = self.down4(c3_out)
        c4_out, seam4_patterns = self.seam4(c4) if self.use_seam else (c4, None)

        c5 = self.down5(c4_out)
        c5_out, seam5_patterns = self.seam5(c5) if self.use_seam else (c5, None)

        features = {
            "c2": c2,
            "c3": c3_out,
            "c4": c4_out,
            "c5": c5_out,
        }

        nuisance_maps = {
            "seam3": seam3_patterns,
            "seam4": seam4_patterns,
            "seam5": seam5_patterns,
        }

        return features, nuisance_maps


class AdapFuseODDetector(nn.Module):
    """
    Complete AdapFuse-OD Modified Detector for Phase 2.
    
    Modular architecture allowing toggling of any of the 5 novel features
    for ablation studies:
        - use_spd_conv (bool)
        - use_seam (bool)
        - use_p2_head (bool)
        - use_nad_head (bool)
        - use_acr (bool)
    """

    def __init__(self, num_classes: int = 2, num_nuisance_classes: int = 6,
                 base_channels: int = 32,
                 use_spd_conv: bool = True,
                 use_seam: bool = True,
                 use_p2_head: bool = True,
                 use_nad_head: bool = True,
                 use_acr: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_nuisance_classes = num_nuisance_classes
        self.use_spd_conv = use_spd_conv
        self.use_seam = use_seam
        self.use_p2_head = use_p2_head
        self.use_nad_head = use_nad_head
        self.use_acr = use_acr

        # Backbone
        self.backbone = ModifiedBackboneWithSPD(
            in_channels=3,
            base_channels=base_channels,
            use_spd=use_spd_conv,
            use_seam=use_seam,
        )

        ch = self.backbone.out_channels

        # P2 Module (stride 4)
        if use_p2_head:
            self.p2_module = P2Module(
                c2_channels=ch["c2"],
                p3_channels=ch["c3"],
                num_classes=num_classes,
                hidden_dim=ch["c2"],
            )

        # Standard Multi-scale heads (P3, P4, P5)
        num_outputs = 3 * (5 + num_classes)  # 3 anchors * (xywh + obj + classes)
        self.head_p3 = nn.Conv2d(ch["c3"], num_outputs, 1)
        self.head_p4 = nn.Conv2d(ch["c4"], num_outputs, 1)
        self.head_p5 = nn.Conv2d(ch["c5"], num_outputs, 1)

        # NAD Head (Nuisance-Aware Detection)
        if use_nad_head:
            self.nad_head = NADHead(
                in_channels=ch["c4"],
                num_classes=num_classes,
                num_nuisance_classes=num_nuisance_classes,
            )

        # ACR Module (Adaptive Confidence Recalibration)
        if use_acr:
            self.acr = ACR(feature_channels=ch["c4"])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features, nuisance_maps = self.backbone(x)

        outputs = {
            "nuisance_maps": nuisance_maps,
        }

        # Multi-scale detection heads
        dets_p3 = self.head_p3(features["c3"])
        dets_p4 = self.head_p4(features["c4"])
        dets_p5 = self.head_p5(features["c5"])

        outputs["det_p3"] = dets_p3
        outputs["det_p4"] = dets_p4
        outputs["det_p5"] = dets_p5

        # P2 head for small objects
        if self.use_p2_head:
            dets_p2, p2_feats = self.p2_module(features["c2"], features["c3"])
            outputs["det_p2"] = dets_p2
            outputs["p2_feats"] = p2_feats

        # NAD Head
        if self.use_nad_head:
            nad_res = self.nad_head(features["c4"])
            outputs["nad_res"] = nad_res

        # ACR confidence recalibration (during inference/eval or forward)
        if self.use_acr:
            # Flatten predictions to pass through ACR
            raw_scores = torch.sigmoid(dets_p4[:, 4, :, :].flatten())
            recal_scores, quality_info = self.acr(raw_scores, features["c4"])
            outputs["recal_scores"] = recal_scores
            outputs["quality_info"] = quality_info

        return outputs


if __name__ == "__main__":
    # Test forward pass of modified detector
    x = torch.randn(2, 3, 416, 416)
    model = AdapFuseODDetector(
        num_classes=2,
        use_spd_conv=True,
        use_seam=True,
        use_p2_head=True,
        use_nad_head=True,
        use_acr=True,
    )
    outputs = model(x)
    print("AdapFuseODDetector Forward Pass Success!")
    print(f"  Input: {x.shape}")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Total Params: {params/1e6:.2f}M")
