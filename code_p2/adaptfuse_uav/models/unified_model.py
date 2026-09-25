"""
UniAdapFuse-UAV: Unified Hierarchical Multimodal Disaster Perception Framework
=============================================================================
Jointly performs:
  1. Tri-Modal Global Scene Classification (Disaster 4-cls, Victim 2-cls, Nuisance 5-cls)
  2. Physics-Informed Dynamic Reliability & Uncertainty Estimation (Dynamic-RUE)
  3. Multi-Scale 2D Bounding Box Object Detection (Fire, Smoke, Victim localization)
  4. Bidirectional Cross-Scale Bridging:
     - Top-Down Context Gate (TDCG): Scene context + reliability modulates detection neck.
     - Bottom-Up Spatial Guidance (BUSG): Detection RoI saliency enriches scene victim head.

Author: AdapFuse-UAV Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from models.backbones.rgb_backbone import RGBBackbone
from models.backbones.thermal_backbone import ThermalBackbone
from models.backbones.audio_backbone import AudioCNN, AudioCNNGD
from models.novel_modules.dynamic_rue import DynamicPhysicsRUE
from models.novel_modules.scene_det_bridge import TopDownContextGate, BottomUpSpatialGuidance
from models.full_model import NuisanceAwareHead


class MultiScaleVisualEncoder(nn.Module):
    """
    Extracts multi-scale feature pyramids (C2, C3, C4, C5) and pooled global tokens
    from RGB (MobileNetV3) and Thermal (ResNet18).
    """

    def __init__(self, in_channels: int = 3, is_thermal: bool = False, pretrained: bool = True):
        super().__init__()
        self.is_thermal = is_thermal
        
        if is_thermal:
            import torchvision.models as models
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            if in_channels == 1:
                # 1-channel thermal
                old_conv = resnet.conv1
                resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    if pretrained:
                        resnet.conv1.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
            
            self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # Stride 4 -> C2
            self.stage2 = resnet.layer1  # 64 ch, Stride 4 -> C2
            self.stage3 = resnet.layer2  # 128 ch, Stride 8 -> C3
            self.stage4 = resnet.layer3  # 256 ch, Stride 16 -> C4
            self.stage5 = resnet.layer4  # 512 ch, Stride 32 -> C5
            self.out_channels = [64, 128, 256, 512]
            self.pooled_dim = 512
        else:
            import torchvision.models as models
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mobilenet = models.mobilenet_v3_small(weights=weights)
            features = list(mobilenet.features.children())
            # C2: Stride 4 (features 0..1), C3: Stride 8 (features 2..3), C4: Stride 16 (features 4..8), C5: Stride 32 (features 9..12)
            self.stage2 = nn.Sequential(*features[0:2])    # 16 ch -> C2 (projected to 64)
            self.stage3 = nn.Sequential(*features[2:4])    # 24 ch -> C3 (projected to 128)
            self.stage4 = nn.Sequential(*features[4:9])    # 48 ch -> C4 (projected to 256)
            self.stage5 = nn.Sequential(*features[9:])     # 576 ch -> C5 (projected to 512)
            
            self.proj_c2 = nn.Conv2d(16, 64, 1)
            self.proj_c3 = nn.Conv2d(24, 128, 1)
            self.proj_c4 = nn.Conv2d(48, 256, 1)
            self.proj_c5 = nn.Conv2d(576, 512, 1)
            self.out_channels = [64, 128, 256, 512]
            self.pooled_dim = 512

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            pyramid: [C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32)]
            pooled: (B, pooled_dim)
        """
        if self.is_thermal:
            c1 = self.stage1(x)
            c2 = self.stage2(c1)
            c3 = self.stage3(c2)
            c4 = self.stage4(c3)
            c5 = self.stage5(c4)
            pooled = F.adaptive_avg_pool2d(c5, (1, 1)).flatten(1)
            return [c2, c3, c4, c5], pooled
        else:
            x2 = self.stage2(x)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            x5 = self.stage5(x4)
            c2 = self.proj_c2(x2)
            c3 = self.proj_c3(x3)
            c4 = self.proj_c4(x4)
            c5 = self.proj_c5(x5)
            pooled = F.adaptive_avg_pool2d(c5, (1, 1)).flatten(1)
            return [c2, c3, c4, c5], pooled


class LightweightFPNPANNeck(nn.Module):
    """
    Multi-Scale Feature Pyramid & Path Aggregation Network (FPN + PANet)
    with P2 Stride-4 head for small victim/hazard localization.
    """

    def __init__(self, in_channels: List[int] = None, out_channels: int = 128):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]  # C2, C3, C4, C5
        
        self.lateral_c5 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral_c2 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c2, c3, c4, c5) -> List[torch.Tensor]:
        # Top-down FPN pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p2 = self.smooth_p2(p2)
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        return [p2, p3, p4, p5]  # Strides 4, 8, 16, 32


class DetectionHead(nn.Module):
    """
    Multi-Scale Object Detection Head:
    Predicts [reg (4), obj (1), cls (num_classes)] per pyramid level.
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        # Box regression + objectness + class scores
        out_dim = 4 + 1 + num_classes
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_dim, 1),
            ) for _ in range(4)  # 4 levels: P2, P3, P4, P5
        ])

    def forward(self, pyramids: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Returns list of (B, 5 + num_classes, H_i, W_i) predictions per scale
        """
        return [head(feat) for head, feat in zip(self.heads, pyramids)]


class UniAdapFuseModel(nn.Module):
    """
    Unified Dual-Scale Multimodal Disaster Perception Model.
    """

    QUALITY_DIM = 16

    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        num_det_classes: int = 2,       # Fire, Smoke
        n_nuisance: int = 5,
        proj_dim: int = 192,
        det_neck_channels: int = 128,
        num_heads: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        detach_det_backbone: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.detach_det_backbone = detach_det_backbone
        self._last_outputs: Dict[str, torch.Tensor] = {}
        _q = self.QUALITY_DIM

        # ── 1. Encoders ────────────────────────────────────────────────────────
        self.rgb_encoder = MultiScaleVisualEncoder(in_channels=3, is_thermal=False, pretrained=pretrained)
        self.thermal_encoder = MultiScaleVisualEncoder(in_channels=1, is_thermal=True, pretrained=pretrained)
        self.audio_encoder = AudioCNN()

        # ── 2. Projectors to shared latent space ──────────────────────────────
        self.rgb_proj = nn.Sequential(
            nn.Linear(self.rgb_encoder.pooled_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.thermal_proj = nn.Sequential(
            nn.Linear(self.thermal_encoder.pooled_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_encoder.out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ── 3. Quality Token Heads ─────────────────────────────────────────────
        self.rgb_quality = nn.Sequential(
            nn.Linear(self.rgb_encoder.pooled_dim, 64), nn.ReLU(),
            nn.Linear(64, _q),
        )
        self.thermal_quality = nn.Sequential(
            nn.Linear(self.thermal_encoder.pooled_dim, 64), nn.ReLU(),
            nn.Linear(64, _q),
        )
        self.audio_quality = nn.Sequential(
            nn.Linear(self.audio_encoder.out_dim, 32), nn.ReLU(),
            nn.Linear(32, _q),
        )

        # ── 4. Novel Dynamic-RUE ───────────────────────────────────────────────
        self.dynamic_rue = DynamicPhysicsRUE(feat_dim=proj_dim, quality_dim=_q)

        # ── 5. Cross-Modal Scene Attention & Temporal Memory ───────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(proj_dim)
        self.gru = nn.GRU(input_size=proj_dim, hidden_size=proj_dim, batch_first=True)

        # ── 6. Detection Stream: FPN Neck & Modulators ─────────────────────────
        # Combined RGB + Thermal multi-scale fusion convs
        self.pyramid_fusers = nn.ModuleList([
            nn.Conv2d(64 * 2, 64, 1),
            nn.Conv2d(128 * 2, 128, 1),
            nn.Conv2d(256 * 2, 256, 1),
            nn.Conv2d(512 * 2, 512, 1),
        ])
        self.det_neck = LightweightFPNPANNeck(in_channels=[64, 128, 256, 512], out_channels=det_neck_channels)
        self.det_head = DetectionHead(in_channels=det_neck_channels, num_classes=num_det_classes)

        # ── 7. Bidirectional Bridge ────────────────────────────────────────────
        self.top_down_gate = TopDownContextGate(
            scene_dim=proj_dim, in_channels_list=[det_neck_channels] * 4
        )
        self.bottom_up_guidance = BottomUpSpatialGuidance(
            in_channels_list=[det_neck_channels] * 4, out_dim=proj_dim
        )

        # ── 8. Classification Heads ────────────────────────────────────────────
        self.nuisance_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_nuisance),
        )
        # Fuses global scene context + bottom-up RoI hazard token
        self.disaster_head = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_victim_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        audio: torch.Tensor,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified Forward Pass.
        
        Args:
            rgb: (B, 3, H, W)
            thermal: (B, 1, H, W) or (B, 3, H, W)
            audio: (B, 1, n_mels, T)
            has_*: (B,) availability flags
        
        Returns:
            Dict with keys:
              - 'disaster': (B, 4)
              - 'victim': (B, 2)
              - 'nuisance': (B, 5)
              - 'reliability': (B, 3)
              - 'uncertainty': (B, 3)
              - 'degradation_logits': (B, 3)
              - 'det_pyramids': List of (B, 5+num_det, H_i, W_i) dense detection maps
              - 'saliency_maps': List of (B, 1, H_i, W_i) spatial hazard focus maps
        """
        B = rgb.size(0)
        device = rgb.device

        if has_rgb is None: has_rgb = torch.ones(B, device=device)
        if has_thermal is None: has_thermal = torch.ones(B, device=device)
        if has_audio is None: has_audio = torch.ones(B, device=device)

        if thermal.size(1) == 3:
            th_in = thermal.mean(dim=1, keepdim=True)
        else:
            th_in = thermal

        # ── A. Multi-Scale Encodings ───────────────────────────────────────────
        rgb_pyr, f_rgb_pool = self.rgb_encoder(rgb)
        th_pyr, f_th_pool = self.thermal_encoder(th_in)
        f_au_pool = self.audio_encoder(audio)

        # Projections
        f_rgb = self.rgb_proj(f_rgb_pool) * has_rgb.unsqueeze(-1)
        f_th = self.thermal_proj(f_th_pool) * has_thermal.unsqueeze(-1)
        f_au = self.audio_proj(f_au_pool) * has_audio.unsqueeze(-1)

        # Quality tokens
        q_rgb = self.rgb_quality(f_rgb_pool)
        q_th = self.thermal_quality(f_th_pool)
        q_au = self.audio_quality(f_au_pool)

        # ── B. Dynamic-RUE ─────────────────────────────────────────────────────
        features = {'rgb': f_rgb, 'thermal': f_th, 'audio': f_au}
        quality_tokens = {'rgb': q_rgb, 'thermal': q_th, 'audio': q_au}
        has_modality = {'rgb': has_rgb, 'thermal': has_thermal, 'audio': has_audio}

        reliability, uncertainty, degradation_logits = self.dynamic_rue(
            rgb_raw=rgb, thermal_raw=th_in, audio_raw=audio,
            features=features, quality_tokens=quality_tokens, has_modality=has_modality
        )

        r_rgb = reliability[:, 0:1]
        r_th = reliability[:, 1:2]
        r_au = reliability[:, 2:3]

        # ── C. Tri-Modal Scene Fusion ──────────────────────────────────────────
        f_rgb_w = f_rgb * r_rgb
        f_th_w = f_th * r_th
        f_au_w = f_au * r_au

        tokens = torch.stack([f_rgb_w, f_th_w, f_au_w], dim=1)  # (B, 3, D)
        attended, _ = self.cross_attn(tokens, tokens, tokens)
        attended = self.attn_norm(attended + tokens)
        fused_scene = attended.mean(dim=1)  # (B, D)
        
        # Temporal memory
        fused_seq, _ = self.gru(fused_scene.unsqueeze(1))
        h_scene = fused_seq.squeeze(1)  # (B, D)

        # ── D. Detection Stream with Dual-Scale Fusion ─────────────────────────
        # Fuse RGB & Thermal feature maps at each pyramid level
        fused_c2 = self.pyramid_fusers[0](torch.cat([rgb_pyr[0] * r_rgb.view(B, 1, 1, 1), th_pyr[0] * r_th.view(B, 1, 1, 1)], dim=1))
        fused_c3 = self.pyramid_fusers[1](torch.cat([rgb_pyr[1] * r_rgb.view(B, 1, 1, 1), th_pyr[1] * r_th.view(B, 1, 1, 1)], dim=1))
        fused_c4 = self.pyramid_fusers[2](torch.cat([rgb_pyr[2] * r_rgb.view(B, 1, 1, 1), th_pyr[2] * r_th.view(B, 1, 1, 1)], dim=1))
        fused_c5 = self.pyramid_fusers[3](torch.cat([rgb_pyr[3] * r_rgb.view(B, 1, 1, 1), th_pyr[3] * r_th.view(B, 1, 1, 1)], dim=1))

        # Gradient isolation: detach pyramid features before detection neck
        # so detection loss gradients do NOT flow back into the shared backbone.
        # This prevents the detection task from overwriting classification features.
        if self.detach_det_backbone:
            det_c2 = fused_c2.detach()
            det_c3 = fused_c3.detach()
            det_c4 = fused_c4.detach()
            det_c5 = fused_c5.detach()
        else:
            det_c2, det_c3, det_c4, det_c5 = fused_c2, fused_c3, fused_c4, fused_c5

        # FPN/PANet multi-scale neck
        raw_neck_pyramids = self.det_neck(det_c2, det_c3, det_c4, det_c5)  # [P2, P3, P4, P5]

        # ── E. Bidirectional Bridge Interaction ────────────────────────────────
        # 1. Top-Down Context Modulation (Scene -> Det Neck)
        modulated_neck_pyramids = self.top_down_gate(raw_neck_pyramids, h_scene, reliability)

        # 2. Bottom-Up Spatial Guidance (Det Neck -> Scene Head)
        salient_hazard_token, saliency_maps = self.bottom_up_guidance(modulated_neck_pyramids)

        # ── F. Final Prediction Heads ──────────────────────────────────────────
        # Dense bounding box predictions
        det_pyramids = self.det_head(modulated_neck_pyramids)

        # Joint scene classification (Conditioned on both Global Scene + Local RoI token)
        joint_cls_embedding = torch.cat([h_scene, salient_hazard_token], dim=-1)  # (B, 2*D)

        disaster_logits = self.disaster_head(joint_cls_embedding)
        victim_logits = self.victim_head(joint_cls_embedding)
        nuisance_logits = self.nuisance_head(h_scene)

        outputs = {
            'disaster': disaster_logits,
            'victim': victim_logits,
            'nuisance': nuisance_logits,
            'reliability': reliability,
            'uncertainty': uncertainty,
            'degradation_logits': degradation_logits,
            'det_pyramids': det_pyramids,
            'saliency_maps': saliency_maps,
            'fused_scene': h_scene,
            'fused_feat': h_scene,  # alias for KD compatibility
        }
        self._last_outputs = outputs
        return outputs

    def get_full_outputs(self) -> Dict[str, torch.Tensor]:
        """Return full outputs dict (call after forward). Fixes trainer integration."""
        return self._last_outputs

    def set_training_phase(self, phase: str):
        """
        Switch between curriculum training phases.
        
        Args:
            phase: 'classification_only' - freeze detection stream, train classification.
                   'joint' - unfreeze everything for joint multi-task training.
        """
        if phase == 'classification_only':
            # Freeze detection-specific components
            for module in [self.det_neck, self.det_head, self.pyramid_fusers]:
                for p in module.parameters():
                    p.requires_grad = False
            # Keep classification stream trainable
            for module in [self.rgb_encoder, self.thermal_encoder, self.audio_encoder,
                           self.rgb_proj, self.thermal_proj, self.audio_proj,
                           self.rgb_quality, self.thermal_quality, self.audio_quality,
                           self.dynamic_rue, self.cross_attn, self.attn_norm, self.gru,
                           self.top_down_gate, self.bottom_up_guidance,
                           self.disaster_head, self.victim_head, self.nuisance_head]:
                for p in module.parameters():
                    p.requires_grad = True
            print("[PHASE] Classification-only warmup: detection stream frozen.")
        elif phase == 'joint':
            # Unfreeze everything
            for p in self.parameters():
                p.requires_grad = True
            print("[PHASE] Joint multi-task training: all parameters unfrozen.")
        else:
            raise ValueError(f"Unknown training phase: {phase}")
