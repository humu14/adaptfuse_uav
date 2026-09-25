"""
AudioYOLO — Sound Event Bounding Box Detection on Spectrograms
===============================================================
Novel Contribution #3: Treats log-mel spectrograms (128 mel-bins × 256 time-frames)
as grayscale images and predicts time-frequency (TF) bounding boxes around
sound events (e.g. fire crackling, human cries, sirens, machinery).

Includes a learned Spectral Subtraction Layer that estimates and removes
the drone's rotor noise profile from DroneAudioset inputs.

TF Bounding Box: [t_center, f_center, t_width, f_width, confidence, class_prob]
    - t_center, t_width: Time onset/offset and duration
    - f_center, f_width: Frequency band range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DroneNoiseFilterLayer(nn.Module):
    """
    Learned spectral subtraction layer for suppressing UAV rotor noise
    from raw log-mel spectrograms.
    
    Drone noise is concentrated in specific low-frequency harmonic bands
    (typically 100 Hz - 2 kHz depending on motor RPM). This module learns
    an adaptive frequency mask to attenuate motor harmonics.
    """

    def __init__(self, num_mel_bins: int = 128):
        super().__init__()
        # Learnable per-frequency suppression mask
        self.freq_weight = nn.Parameter(torch.ones(1, 1, num_mel_bins, 1))
        # Adaptive noise estimator network
        self.noise_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((num_mel_bins, 1)),  # Average over time
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=(3, 1), padding=(1, 0)),
            nn.Sigmoid(),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: (B, 1, F, T) log-mel spectrogram (F=128, T=256)
        
        Returns:
            clean_spectrogram: (B, 1, F, T) noise-attenuated spectrogram
        """
        noise_profile = self.noise_estimator(spectrogram)  # (B, 1, F, 1)
        # Spectral subtraction in log domain = division in linear domain
        filtered = spectrogram * (1.0 - 0.7 * noise_profile) * self.freq_weight
        return filtered


class AudioYOLOBackbone(nn.Module):
    """
    Lightweight 2D CNN backbone tailored for non-square spectrograms (128 × 256).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.noise_filter = DroneNoiseFilterLayer(num_mel_bins=128)

        # Stage 1: (128, 256) -> (64, 128)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        # Stage 2: (64, 128) -> (32, 64)
        c2 = base_channels * 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # Stage 3: (32, 64) -> (16, 32)
        c3 = base_channels * 4
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )

        # Stage 4: (16, 32) -> (8, 16)
        c4 = base_channels * 8
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )

        self.out_channels = c4

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        clean_spec = self.noise_filter(spec)
        s1 = self.stage1(clean_spec)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s4


class AudioYOLODetector(nn.Module):
    """
    AudioYOLO Model for detecting sound events as time-frequency bounding boxes.
    """

    def __init__(self, num_audio_classes: int = 5, base_channels: int = 32):
        """
        Classes (example):
            0: fire_crackling
            1: human_cry / voice
            2: siren
            3: explosion
            4: machinery / vehicle
        """
        super().__init__()
        self.num_classes = num_audio_classes
        self.backbone = AudioYOLOBackbone(in_channels=1, base_channels=base_channels)

        out_ch = self.backbone.out_channels
        num_anchors = 3
        outputs_per_anchor = 5 + num_audio_classes  # [t, f, w, h, conf, class_probs]

        self.head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, num_anchors * outputs_per_anchor, 1),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: (B, 1, 128, 256) log-mel spectrogram
        
        Returns:
            tf_boxes: (B, num_anchors * outputs, 8, 16) time-frequency detections
        """
        feats = self.backbone(spectrogram)
        tf_boxes = self.head(feats)
        return tf_boxes


if __name__ == "__main__":
    spec = torch.randn(2, 1, 128, 256)  # 2 samples, log-mel spectrogram
    audio_net = AudioYOLODetector(num_audio_classes=5)
    out = audio_net(spec)
    print("AudioYOLO Test:")
    print(f"  Spectrogram input: {spec.shape}")
    print(f"  TF-Box output:     {out.shape}")
    params = sum(p.numel() for p in audio_net.parameters())
    print(f"  Total Params:      {params:,}")
