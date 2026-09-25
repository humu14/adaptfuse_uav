"""
Audio backbone: 2D-CNN on log-mel spectrograms.
Design A3 unimodal baseline + shared encoder for fusion models.
Architecture: 3 conv blocks (32→64→128 filters) + GAP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Conv2d + BN + ReLU + MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AudioCNN(nn.Module):
    """
    2D-CNN on log-mel spectrogram.
    Input: (B, 1, 64, 63)  — (batch, 1ch, n_mels, time_frames)
    Output: (B, 128)
    """

    def __init__(self, in_channels: int = 1, width_mult: float = 1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        self.out_dim = c3

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c1, pool=True),   # (B, 32, 32, 31)
            ConvBlock(c1, c2, pool=True),             # (B, 64, 16, 15)
            ConvBlock(c2, c3, pool=True),             # (B, 128, 8, 7)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 64, T)
        returns: (B, 128)
        """
        x = self.encoder(x)    # (B, 128, H', W')
        x = self.gap(x)        # (B, 128, 1, 1)
        x = torch.flatten(x, 1)  # (B, 128)
        return x


class AudioOnlyModel(nn.Module):
    """
    Design A3: Audio-only classification model.
    2D-CNN on log-mel → GAP → Linear(128,64) → ReLU → Linear(64, num_classes)
    """

    def __init__(
        self,
        num_disaster_classes: int = 4,
        num_victim_classes: int = 2,
        dropout: float = 0.3,
        backbone: str = "audiocnn",
    ):
        super().__init__()
        if backbone == "panns_cnn6":
            self.backbone = PANNsAudioBackbone(pretrained=True)
        elif backbone == "audiocnn_gd":
            self.backbone = AudioCNNGD()
        else:
            self.backbone = AudioCNN()
        feat_dim = self.backbone.out_dim  # 128 / 512 / 128

        self.disaster_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_disaster_classes),
        )
        self.victim_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_victim_classes),
        )

    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        has_rgb: Optional[torch.Tensor] = None,
        has_thermal: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ):
        feat = self.backbone(audio)
        disaster_out = self.disaster_head(feat)
        victim_out = self.victim_head(feat)
        reliability = torch.ones(audio.size(0), 3, device=audio.device) / 3.0
        return disaster_out, victim_out, reliability


# ─────────────────────────────────────────────────────────────────────────────
# EXP-14: PANNS-CNN6 pretrained audio backbone
# Kong et al. 2020 — pretrained on 527-class AudioSet (2M+ clips)
# ─────────────────────────────────────────────────────────────────────────────

class PANNsAudioBackbone(nn.Module):
    """
    CNN6 architecture pretrained on AudioSet (PANNS, Kong et al. 2020).
    Replaces random-init AudioCNN. AudioSet classes include fire, crowd,
    sirens, machinery — directly relevant to disaster audio.
    out_dim: 512
    """

    def __init__(self, pretrained: bool = True, freeze_early: bool = False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d((2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

        if pretrained:
            self._load_panns_weights()

        if freeze_early:
            for p in list(self.conv1.parameters()) + list(self.conv2.parameters()):
                p.requires_grad = False

    def _load_panns_weights(self):
        import urllib.request
        import os
        cache = os.path.expanduser('~/.cache/panns/Cnn6_mAP=0.343.pth')
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if not os.path.exists(cache):
            print('[PANNS] Downloading CNN6 weights (~65MB)...')
            url = 'https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth'
            try:
                urllib.request.urlretrieve(url, cache)
            except Exception as e:
                print(f'[PANNS] Download failed: {e}. Training from random init.')
                return
        try:
            ckpt = torch.load(cache, map_location='cpu')
            sd = ckpt.get('model', ckpt)
            own_sd = self.state_dict()
            # Map PANNS CNN6 key names → our simplified layer names
            panns_to_ours = {
                'conv_block1.conv1.weight': 'conv1.0.weight',
                'conv_block1.bn1.weight':   'conv1.1.weight',
                'conv_block1.bn1.bias':     'conv1.1.bias',
                'conv_block2.conv1.weight': 'conv2.0.weight',
                'conv_block2.bn1.weight':   'conv2.1.weight',
                'conv_block2.bn1.bias':     'conv2.1.bias',
                'conv_block3.conv1.weight': 'conv3.0.weight',
                'conv_block3.bn1.weight':   'conv3.1.weight',
                'conv_block3.bn1.bias':     'conv3.1.bias',
                'conv_block4.conv1.weight': 'conv4.0.weight',
                'conv_block4.bn1.weight':   'conv4.1.weight',
                'conv_block4.bn1.bias':     'conv4.1.bias',
            }
            mapped = {}
            for panns_k, our_k in panns_to_ours.items():
                if panns_k in sd and our_k in own_sd:
                    if sd[panns_k].shape == own_sd[our_k].shape:
                        mapped[our_k] = sd[panns_k]
            missing = self.load_state_dict(mapped, strict=False)
            print(f'[PANNS] Loaded {len(mapped)} pretrained tensors. '
                  f'Missing: {len(missing.missing_keys)}')
        except Exception as e:
            print(f'[PANNS] Weight load failed: {e}. Training from random init.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 64, T) → (B, 512)"""
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.gap(x)
        return torch.flatten(x, 1)


# ─────────────────────────────────────────────────────────────────────────────
# EXP-18: GDBlock audio encoder (GlimmerNet-inspired multi-scale dilated CNN)
# ─────────────────────────────────────────────────────────────────────────────

class GDBlock(nn.Module):
    """
    Multi-scale Grouped Dilated conv block inspired by GlimmerNet (2025).
    3 parallel dilated convs (d=1,3,5) + pointwise fusion.
    Captures multi-scale temporal context in spectrograms.
    """

    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 3, 5)):
        super().__init__()
        branch_ch = max(1, in_ch // len(dilations))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        total_ch = branch_ch * len(dilations)
        self.pointwise = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(torch.cat([b(x) for b in self.branches], dim=1))


class AudioCNNGD(nn.Module):
    """
    AudioCNN with GDBlocks replacing standard ConvBlocks.
    Same channel progression (32→64→128), multi-scale dilated receptive field.
    out_dim: 128 (compatible with AudioCNN projectors)
    """

    def __init__(self, in_channels: int = 1, width_mult: float = 1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        self.out_dim = c3
        self.encoder = nn.Sequential(
            GDBlock(in_channels, c1), nn.MaxPool2d(2),
            GDBlock(c1, c2),          nn.MaxPool2d(2),
            GDBlock(c2, c3),          nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.gap(x)
        return torch.flatten(x, 1)
