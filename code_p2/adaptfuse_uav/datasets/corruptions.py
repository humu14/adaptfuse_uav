"""
AdapFuse-UAV: Corruption augmentation functions.
Implements: smoke overlay, motion blur, low light, thermal drift, rotor noise.
"""

import cv2
import numpy as np
import random
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# RGB / Visual corruptions
# ─────────────────────────────────────────────────────────────────────────────

def apply_smoke_overlay(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    Add a gray-white smoke haze via alpha blending.
    severity: 0.0 (no smoke) to 1.0 (fully gray)
    img: (H, W, 3) uint8 or float32
    """
    smoke_color = np.array([200, 200, 200], dtype=np.float32)
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32)
        blended = (1 - severity) * img_f + severity * smoke_color
        return np.clip(blended, 0, 255).astype(np.uint8)
    else:
        smoke_norm = smoke_color / 255.0
        blended = (1 - severity) * img + severity * smoke_norm
        return np.clip(blended, 0, 1).astype(np.float32)


def apply_motion_blur(img: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Apply horizontal motion blur.
    kernel_size: odd integer, e.g. 3, 5, 7, 11, 15
    """
    kernel_size = max(3, kernel_size | 1)  # ensure odd
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def apply_low_light(img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """
    Simulate low-light by applying gamma correction (darkening).
    gamma > 1 darkens the image.
    img: (H, W, 3) uint8
    """
    if img.dtype == np.uint8:
        table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(img, table)
    else:
        return np.clip(img**gamma, 0, 1).astype(np.float32)


def apply_gaussian_noise(img: np.ndarray, std: float = 20.0) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.randn(*img.shape).astype(np.float32) * std
    if img.dtype == np.uint8:
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        return np.clip(img + noise / 255.0, 0, 1).astype(np.float32)


def apply_rain_streaks(img: np.ndarray, density: int = 500) -> np.ndarray:
    """Simulate rain streaks on image."""
    h, w = img.shape[:2]
    rain = img.copy().astype(np.float32)
    for _ in range(density):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 20)
        x2 = x1 + random.randint(-2, 2)
        y2 = y1 + random.randint(10, 20)
        cv2.line(rain, (x1, y1), (x2, y2), (200, 200, 200), 1)
    if img.dtype == np.uint8:
        return np.clip(rain, 0, 255).astype(np.uint8)
    return np.clip(rain / 255.0, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Thermal corruptions
# ─────────────────────────────────────────────────────────────────────────────

def apply_thermal_drift(thermal: np.ndarray, offset: float = 0.3) -> np.ndarray:
    """
    Apply uniform thermal drift (DC offset shift).
    thermal: (1, H, W) float32, z-score normalized
    """
    return np.clip(thermal + offset, -4.0, 4.0)


def apply_thermal_saturation_noise(thermal: np.ndarray, noise_std: float = 0.5) -> np.ndarray:
    """Add saturation noise to thermal image."""
    noise = np.random.randn(*thermal.shape).astype(np.float32) * noise_std
    return np.clip(thermal + noise, -4.0, 4.0)


def apply_thermal_dead_pixels(thermal: np.ndarray, prob: float = 0.01) -> np.ndarray:
    """Simulate dead pixels in thermal sensor."""
    mask = np.random.random(thermal.shape) < prob
    thermal = thermal.copy()
    thermal[mask] = 0.0
    return thermal


# ─────────────────────────────────────────────────────────────────────────────
# Audio corruptions
# ─────────────────────────────────────────────────────────────────────────────

def add_rotor_noise(log_mel: np.ndarray, snr_db: float = 5.0) -> np.ndarray:
    """
    Add synthetic UAV rotor noise to log-mel spectrogram.
    snr_db: signal-to-noise ratio in dB. Lower = more noise.
    """
    signal_power = np.mean(log_mel ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*log_mel.shape).astype(np.float32) * np.sqrt(noise_power)
    # Emphasize low-frequency noise (rotor is mostly low-freq)
    low_freq_mask = np.ones_like(log_mel)
    low_freq_mask[0, :log_mel.shape[1] // 4, :] = 2.0  # boost low mels
    return log_mel + noise * low_freq_mask


def add_wind_noise(log_mel: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add broadband wind noise to log-mel."""
    noise = np.random.randn(*log_mel.shape).astype(np.float32) * intensity
    return log_mel + noise


def apply_audio_dropout(log_mel: np.ndarray, prob: float = 0.1) -> np.ndarray:
    """Randomly zero out time frames (simulates packet loss or dropout)."""
    T = log_mel.shape[-1]
    mask = np.random.random(T) > prob
    log_mel = log_mel.copy()
    log_mel[:, :, ~mask] = log_mel.min()
    return log_mel


# ─────────────────────────────────────────────────────────────────────────────
# Corruption dispatch
# ─────────────────────────────────────────────────────────────────────────────

CORRUPTION_PARAMS = {
    "smoke_overlay": [0.1, 0.3, 0.5, 0.7, 0.9],
    "motion_blur": [3, 5, 7, 11, 15],
    "low_light": [2.0, 2.5, 3.0, 3.5, 4.0],
    "thermal_drift": [0.1, 0.3, 0.5, 0.7, 1.0],
    "rotor_noise_snr": [20, 15, 10, 5, 0],
}


def apply_rgb_corruption(img: np.ndarray, corruption: str, severity: int = 3) -> np.ndarray:
    """
    severity: 1-5 index into CORRUPTION_PARAMS.
    """
    sev_idx = max(0, min(4, severity - 1))
    if corruption == "smoke_overlay":
        return apply_smoke_overlay(img, CORRUPTION_PARAMS["smoke_overlay"][sev_idx])
    elif corruption == "motion_blur":
        return apply_motion_blur(img, CORRUPTION_PARAMS["motion_blur"][sev_idx])
    elif corruption == "low_light":
        return apply_low_light(img, CORRUPTION_PARAMS["low_light"][sev_idx])
    elif corruption == "gaussian_noise":
        stds = [5, 10, 20, 35, 50]
        return apply_gaussian_noise(img, stds[sev_idx])
    else:
        return img


def apply_thermal_corruption(thermal: np.ndarray, corruption: str, severity: int = 3) -> np.ndarray:
    sev_idx = max(0, min(4, severity - 1))
    if corruption == "thermal_drift":
        return apply_thermal_drift(thermal, CORRUPTION_PARAMS["thermal_drift"][sev_idx])
    elif corruption == "saturation_noise":
        stds = [0.1, 0.3, 0.5, 0.8, 1.2]
        return apply_thermal_saturation_noise(thermal, stds[sev_idx])
    else:
        return thermal


def apply_audio_corruption(log_mel: np.ndarray, corruption: str, severity: int = 3) -> np.ndarray:
    sev_idx = max(0, min(4, severity - 1))
    if corruption == "rotor_noise_snr":
        return add_rotor_noise(log_mel, CORRUPTION_PARAMS["rotor_noise_snr"][sev_idx])
    elif corruption == "wind_noise":
        intensities = [0.05, 0.1, 0.2, 0.35, 0.5]
        return add_wind_noise(log_mel, intensities[sev_idx])
    else:
        return log_mel
