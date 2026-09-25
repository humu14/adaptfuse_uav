"""Lightweight hazard-aware input gate for fire and smoke detection.

The module exposes three differentiable, task-specific cues to a very small
learned residual branch:

* fire chroma: red excess over the green/blue channels;
* smoke chroma: local achromaticity (low channel spread);
* boundary detail: luminance gradient magnitude.

It does not require masks or auxiliary labels.  The detector's ordinary box,
class, and distribution-focal losses train the gate end-to-end.  The residual
path is initialized close to the identity so a pretrained detector is not
destroyed at the beginning of fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardPriorGate(nn.Module):
    """Spectral-spatial residual gate for normalized RGB images."""

    def __init__(
        self,
        hidden_channels: int = 16,
        use_fire_prior: bool = True,
        use_smoke_prior: bool = True,
        use_detail_prior: bool = True,
        max_residual: float = 0.25,
        gate_stride: int = 4,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        if not any((use_fire_prior, use_smoke_prior, use_detail_prior)):
            raise ValueError("At least one hazard prior must be enabled")

        self.use_fire_prior = use_fire_prior
        self.use_smoke_prior = use_smoke_prior
        self.use_detail_prior = use_detail_prior
        self.max_residual = float(max_residual)
        self.gate_stride = int(gate_stride)
        if self.gate_stride < 1:
            raise ValueError("gate_stride must be at least 1")

        cue_channels = 3 + sum(
            (use_fire_prior, use_smoke_prior, use_detail_prior)
        )
        self.cue_encoder = nn.Sequential(
            nn.Conv2d(cue_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        # Three RGB residuals plus one shared spatial gate.
        self.residual_head = nn.Conv2d(hidden_channels, 4, 1)
        self.residual_strength = nn.Parameter(torch.tensor(-2.0))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # Near-identity initialization preserves the pretrained input domain.
        nn.init.normal_(self.residual_head.weight[:3], mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head.bias[:3])
        nn.init.zeros_(self.residual_head.weight[3:])
        nn.init.constant_(self.residual_head.bias[3:], -1.5)

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"HazardPriorGate expects BCHW RGB input, received {tuple(x.shape)}"
            )

    def _hazard_cues(self, x: torch.Tensor) -> list[torch.Tensor]:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        cues: list[torch.Tensor] = []

        if self.use_fire_prior:
            cues.append(F.relu(r - 0.5 * (g + b)))

        if self.use_smoke_prior:
            channel_max = x.amax(dim=1, keepdim=True)
            channel_min = x.amin(dim=1, keepdim=True)
            chroma = (channel_max - channel_min) / channel_max.clamp_min(1e-4)
            cues.append(1.0 - chroma.clamp(0.0, 1.0))

        if self.use_detail_prior:
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            grad_x = F.conv2d(luminance, self.sobel_x, padding=1)
            grad_y = F.conv2d(luminance, self.sobel_y, padding=1)
            magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
            cues.append(torch.tanh(4.0 * magnitude))

        return cues

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        cue_parts = [x, *self._hazard_cues(x)]
        if self.gate_stride > 1:
            cue_parts = [
                F.avg_pool2d(part, self.gate_stride, self.gate_stride, ceil_mode=True)
                for part in cue_parts
            ]
        features = self.cue_encoder(torch.cat(cue_parts, dim=1))
        correction = self.residual_head(features)
        if correction.shape[-2:] != x.shape[-2:]:
            correction = F.interpolate(
                correction, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        rgb_residual = torch.tanh(correction[:, :3])
        spatial_gate = torch.sigmoid(correction[:, 3:4])
        strength = self.max_residual * torch.sigmoid(self.residual_strength)
        return x + strength * spatial_gate * rgb_residual


class HazardAwareStem(nn.Module):
    """Attach :class:`HazardPriorGate` before an Ultralytics stem layer.

    Ultralytics stores graph metadata on each top-level layer.  Copying those
    attributes makes this wrapper a drop-in replacement for layer zero.
    """

    def __init__(self, stem: nn.Module, gate: HazardPriorGate) -> None:
        super().__init__()
        self.gate = gate
        self.stem = stem
        for attribute in ("i", "f", "type", "np"):
            if hasattr(stem, attribute):
                setattr(self, attribute, getattr(stem, attribute))
        self.type = f"{type(self).__module__}.{type(self).__name__}"
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(self.gate(x))


def attach_hazard_prior_gate(model: nn.Module, **gate_kwargs) -> nn.Module:
    """Insert the gate before layer zero of an Ultralytics detection model."""
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential) or len(layers) == 0:
        raise TypeError("Expected an Ultralytics model with a Sequential 'model'")
    if isinstance(layers[0], HazardAwareStem):
        return model
    layers[0] = HazardAwareStem(layers[0], HazardPriorGate(**gate_kwargs))
    return model
