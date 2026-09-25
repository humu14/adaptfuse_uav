"""
Scene-to-Detection Bidirectional Bridge (Top-Down Context & Bottom-Up RoI Guidance)
===================================================================================
Unifies the Tri-Modal Global Classification Stream and Dense Object Detection Stream.

1. TopDownContextGate (TDCG):
   - Injects global scene context (disaster embeddings + sensor reliability) into
     multi-scale detection feature pyramids (P2, P3, P4, P5) via FiLM modulation.
   - Suppresses background false-alarms (solar heating, clouds) when global scene is clean.

2. BottomUpSpatialGuidance (BUSG):
   - Derives high-resolution spatial hazard saliency maps from detection feature pyramids.
   - Feeds RoI-weighted visual features back to the scene classification head,
     preventing small victims and tiny fire seeds from being blurred by Global Average Pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class TopDownContextGate(nn.Module):
    """
    Top-Down Contextual Conditioning:
    Modulates detection feature maps using global tri-modal scene context and reliability.
    """

    def __init__(self, scene_dim: int = 192, in_channels_list: List[int] = None):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [64, 128, 256, 512]  # P2, P3, P4, P5 channels
        
        self.modulators = nn.ModuleList()
        for ch in in_channels_list:
            # Predicts scale (gamma) and shift (beta) per channel
            mod = nn.Sequential(
                nn.Linear(scene_dim + 3, ch),  # scene_dim + 3 reliability scores
                nn.SiLU(),
                nn.Linear(ch, ch * 2),         # gamma, beta
            )
            self.modulators.append(mod)

    def forward(
        self,
        det_features: List[torch.Tensor],
        scene_embedding: torch.Tensor,
        reliability: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            det_features: List of (B, C_i, H_i, W_i) feature pyramids [P2, P3, P4, P5]
            scene_embedding: (B, scene_dim)
            reliability: (B, 3) [r_rgb, r_thermal, r_audio]
        
        Returns:
            modulated_features: List of modulated feature tensors
        """
        # Condition vector: global context + reliability
        cond = torch.cat([scene_embedding, reliability], dim=-1)  # (B, scene_dim + 3)
        
        modulated = []
        for feat, mod_net in zip(det_features, self.modulators):
            gb = mod_net(cond)  # (B, 2*C)
            gamma, beta = torch.chunk(gb, 2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
            
            # FiLM modulation: (1 + gamma) * feat + beta
            mod_feat = (1.0 + torch.tanh(gamma)) * feat + beta
            modulated.append(mod_feat)
            
        return modulated


class BottomUpSpatialGuidance(nn.Module):
    """
    Bottom-Up Spatial Attention:
    Extracts high-resolution hazard/victim saliency from detection features
    to enrich global scene classification.
    """

    def __init__(self, in_channels_list: List[int] = None, out_dim: int = 192):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [64, 128, 256, 512]
        
        # Spatial saliency conv per pyramid level
        self.saliency_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size=1),
                nn.Sigmoid()
            ) for ch in in_channels_list
        ])
        
        # RoI token projection
        total_ch = sum(in_channels_list)
        self.roi_proj = nn.Sequential(
            nn.Linear(total_ch, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, det_features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            det_features: List of (B, C_i, H_i, W_i) detection pyramids [P2, P3, P4, P5]
        
        Returns:
            salient_hazard_token: (B, out_dim) localized hazard representation
            saliency_maps: List of (B, 1, H_i, W_i) spatial hazard masks
        """
        level_tokens = []
        saliency_maps = []
        
        for feat, conv in zip(det_features, self.saliency_convs):
            sal_map = conv(feat)  # (B, 1, H_i, W_i)
            saliency_maps.append(sal_map)
            
            # Weighted RoI spatial pooling
            weight_sum = sal_map.sum(dim=[-2, -1], keepdim=True).clamp(min=1e-5)
            weighted_feat = (feat * sal_map).sum(dim=[-2, -1], keepdim=True) / weight_sum
            level_tokens.append(weighted_feat.squeeze(-1).squeeze(-1))  # (B, C_i)
            
        combined_roi = torch.cat(level_tokens, dim=-1)  # (B, sum(C_i))
        salient_hazard_token = self.roi_proj(combined_roi)  # (B, out_dim)
        
        return salient_hazard_token, saliency_maps
