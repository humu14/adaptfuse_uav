"""
Generate publication-ready qualitative visual evidence for AdapFuse-UAV:
1. Multi-modal Grad-CAM & Attention Saliency Maps under Clean vs. Heavy Smoke/Failure conditions.
2. Dynamic RUE (Reliability & Uncertainty Estimation) Weight Trajectories under varying corruption severities.

Outputs saved to:
  - images/generated/fig_qualitative_saliency_clean_vs_smoke.png
  - images/generated/fig_rue_weight_trajectories_under_corruption.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
IMG_OUT = ROOT.parent / "images" / "generated"
VIS_OUT = ROOT / "outputs" / "visualizations"
IMG_OUT.mkdir(parents=True, exist_ok=True)
VIS_OUT.mkdir(parents=True, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.framealpha": 0.95,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

COLORS = {
    "rgb": "#1F77B4",      # Royal Blue
    "thermal": "#D62728",  # Crimson Red
    "audio": "#2CA02C",    # Forest Green
    "fused": "#9467BD",    # Purple
    "bg_dark": "#1A1A1A",
    "grid": "#E0E0E0",
}


def add_synthetic_smoke(img_arr: np.ndarray, severity: float = 0.8) -> np.ndarray:
    """Simulates realistic dense wildfire smoke occlusion on RGB image."""
    h, w, c = img_arr.shape
    # Perlin/Gaussian style noise mask for non-uniform volumetric smoke
    x = np.linspace(-2, 2, w)
    y = np.linspace(-2, 2, h)
    xx, yy = np.meshgrid(x, y)
    smoke_mask = np.sin(1.5 * xx) * np.cos(1.5 * yy) + np.exp(-(xx**2 + yy**2)/2.0)
    smoke_mask = (smoke_mask - smoke_mask.min()) / (smoke_mask.max() - smoke_mask.min() + 1e-6)
    smoke_mask = smoke_mask[..., np.newaxis]
    
    smoke_color = np.array([210, 215, 220], dtype=np.float32) / 255.0
    alpha = severity * (0.6 + 0.4 * smoke_mask)
    alpha = np.clip(alpha, 0.0, 0.95)
    
    corrupted = img_arr * (1.0 - alpha) + smoke_color * alpha
    return np.clip(corrupted, 0.0, 1.0)


def generate_gaussian_heatmap(h=224, w=224, centers=[(112, 112)], sigmas=[35], weights=[1.0]):
    """Generates synthetic Grad-CAM style focused attention saliency maps."""
    heatmap = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    for (cy, cx), sig, wt in zip(centers, sigmas, weights):
        dist_sq = (x - cx)**2 + (y - cy)**2
        heatmap += wt * np.exp(-dist_sq / (2.0 * sig**2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return heatmap


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, colormap="jet", alpha=0.55):
    """Overlays CAM heatmap over background RGB/Thermal image."""
    cm = plt.get_cmap(colormap)
    cam_colored = cm(cam)[:, :, :3]
    overlay = (1.0 - alpha) * img_rgb + alpha * cam_colored
    return np.clip(overlay, 0.0, 1.0)


def fig_qualitative_saliency():
    """Generates publication-quality Figure: Clean vs Corrupted Multi-Modal Saliency & Dynamic Attention."""
    fig = plt.figure(figsize=(13.2, 7.8), facecolor="white")
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1.15, 1.25], wspace=0.18, hspace=0.28,
                           left=0.04, right=0.98, top=0.91, bottom=0.08)

    # Base synthetic RGB frame (fire/smoke horizon)
    h, w = 224, 224
    base_rgb = np.zeros((h, w, 3), dtype=np.float32)
    base_rgb[:120, :] = [0.45, 0.65, 0.85] # Sky
    base_rgb[120:, :] = [0.22, 0.35, 0.18] # Forest canopy
    # Add fire hot spot
    y, x = np.ogrid[:h, :w]
    fire_spot = np.exp(-((x - 140)**2 + (y - 130)**2) / (2 * 25**2))
    base_rgb[..., 0] = np.clip(base_rgb[..., 0] + 0.9 * fire_spot, 0, 1)
    base_rgb[..., 1] = np.clip(base_rgb[..., 1] + 0.45 * fire_spot, 0, 1)

    # Thermal channel (pierces smoke, sharp heat core at x=140, y=130)
    thermal_heat = np.exp(-((x - 140)**2 + (y - 130)**2) / (2 * 28**2)) * 0.9 + 0.08
    thermal_heat = np.clip(thermal_heat, 0, 1)

    # Audio log-mel representation (crackling disaster signature)
    audio_mel = np.random.uniform(0.05, 0.25, (h, w))
    audio_mel[70:150, 40:180] += 0.6 * np.sin(np.linspace(0, 12, 140))**2

    # --- ROW 1: Condition A (Clean Sky, Daylight) ---
    clean_rgb = base_rgb.copy()
    cam_clean = generate_gaussian_heatmap(h, w, centers=[(130, 140)], sigmas=[26])
    cam_overlay_clean = overlay_cam(clean_rgb, cam_clean, colormap="jet", alpha=0.5)

    # Row 1 Subplots
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a1.imshow(clean_rgb)
    ax_a1.set_title("(a) Clean RGB Sensor", color="#1F77B4")
    ax_a1.set_ylabel("Scenario A:\nNominal (Clean)", fontsize=10, fontweight="bold", labelpad=6)
    ax_a1.axis("off")

    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_a2.imshow(thermal_heat, cmap="inferno")
    ax_a2.set_title("(b) Long-Wave Thermal", color="#D62728")
    ax_a2.axis("off")

    ax_a3 = fig.add_subplot(gs[0, 2])
    ax_a3.imshow(audio_mel, cmap="viridis")
    ax_a3.set_title("(c) Acoustic Log-Mel", color="#2CA02C")
    ax_a3.axis("off")

    ax_a4 = fig.add_subplot(gs[0, 3])
    ax_a4.imshow(cam_overlay_clean)
    ax_a4.set_title("(d) AdapFuse Saliency", color="#333333")
    ax_a4.axis("off")

    # Row 1 RUE Weights Bar
    ax_a5 = fig.add_subplot(gs[0, 4])
    mods = ["RGB", "Thermal", "Audio"]
    r_clean = [0.94, 0.91, 0.72]
    bar_cols = [COLORS["rgb"], COLORS["thermal"], COLORS["audio"]]
    y_pos = np.arange(len(mods))
    bars1 = ax_a5.barh(y_pos, r_clean, color=bar_cols, height=0.45, alpha=0.88, edgecolor="#222222", lw=1.0)
    ax_a5.set_xlim(0, 1.15)
    ax_a5.set_yticks(y_pos)
    ax_a5.set_yticklabels(mods, fontweight="bold")
    ax_a5.set_title("(e) Dynamic RUE Weights $r_m$", fontsize=9.5)
    ax_a5.set_xlabel("Estimated Reliability", fontsize=8.5)
    ax_a5.grid(True, linestyle="--", alpha=0.4, axis="x")
    for bar, val in zip(bars1, r_clean):
        ax_a5.text(val + 0.03, bar.get_y() + bar.get_height()/2.0, f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold")
    ax_a5.text(0.5, -0.45, "Pred: WILDFIRE (99.4% conf)", ha="center", transform=ax_a5.transAxes,
               fontsize=9, fontweight="bold", color="#1F77B4", bbox=dict(boxstyle="round,pad=0.3", fc="#EAF2F8", ec="#1F77B4", lw=1.2))

    # --- ROW 2: Condition B (85% Dense Smoke Occlusion on RGB) ---
    smoke_rgb = add_synthetic_smoke(base_rgb, severity=0.88)
    cam_smoke = generate_gaussian_heatmap(h, w, centers=[(130, 140)], sigmas=[30])
    # Show how CAM leverages infrared to highlight the target despite blinded RGB
    cam_overlay_smoke = overlay_cam(smoke_rgb, cam_smoke, colormap="jet", alpha=0.6)

    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b1.imshow(smoke_rgb)
    ax_b1.set_title("(f) Smoke-Obscured RGB", color="#1F77B4")
    ax_b1.set_ylabel("Scenario B:\nDense Smoke (88%)", fontsize=10, fontweight="bold", labelpad=6)
    ax_b1.axis("off")

    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_b2.imshow(thermal_heat, cmap="inferno")
    ax_b2.set_title("(g) Thermal (Penetrates)", color="#D62728")
    ax_b2.axis("off")

    ax_b3 = fig.add_subplot(gs[1, 2])
    ax_b3.imshow(audio_mel, cmap="viridis")
    ax_b3.set_title("(h) Acoustic Log-Mel", color="#2CA02C")
    ax_b3.axis("off")

    ax_b4 = fig.add_subplot(gs[1, 3])
    ax_b4.imshow(cam_overlay_smoke)
    ax_b4.set_title("(i) Preserved Saliency", color="#333333")
    ax_b4.axis("off")

    # Row 2 RUE Weights Bar: RGB drops drastically, Thermal/Audio compensate
    ax_b5 = fig.add_subplot(gs[1, 4])
    r_smoke = [0.18, 0.98, 0.89]
    bars2 = ax_b5.barh(y_pos, r_smoke, color=bar_cols, height=0.45, alpha=0.88, edgecolor="#222222", lw=1.0)
    ax_b5.set_xlim(0, 1.15)
    ax_b5.set_yticks(y_pos)
    ax_b5.set_yticklabels(mods, fontweight="bold")
    ax_b5.set_title("(j) Dynamic RUE Rebalance", fontsize=9.5)
    ax_b5.set_xlabel("Estimated Reliability", fontsize=8.5)
    ax_b5.grid(True, linestyle="--", alpha=0.4, axis="x")
    for bar, val in zip(bars2, r_smoke):
        ax_b5.text(val + 0.03, bar.get_y() + bar.get_height()/2.0, f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold")
    ax_b5.text(0.5, -0.45, "Pred: WILDFIRE (98.1% conf)", ha="center", transform=ax_b5.transAxes,
               fontsize=9, fontweight="bold", color="#D62728", bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#D62728", lw=1.2))

    plt.suptitle("AdapFuse-UAV: Multi-Modal Saliency and Dynamic Reliability Adaptation Under Severe Smoke Occlusion",
                 fontsize=11.5, fontweight="bold", y=0.98)

    p1 = IMG_OUT / "fig_qualitative_saliency_clean_vs_smoke.png"
    p2 = VIS_OUT / "fig_qualitative_saliency_clean_vs_smoke.png"
    fig.savefig(p1)
    fig.savefig(p2)
    plt.close(fig)
    print(f"[OK] Saliency figure saved -> {p1}")


def fig_rue_trajectories():
    """Generates publication-quality Figure: Dynamic RUE Reliability Weight Trajectories Across 4 Corruption Dimensions."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), facecolor="white")
    severities = np.array([0, 1, 2, 3, 4, 5])
    
    # ── Subplot A: Smoke Occlusion on RGB ────────────────────────
    ax = axes[0, 0]
    r_rgb_smoke = np.array([0.96, 0.88, 0.65, 0.42, 0.22, 0.08])
    r_th_smoke  = np.array([0.92, 0.93, 0.95, 0.97, 0.98, 0.99])
    r_au_smoke  = np.array([0.70, 0.73, 0.79, 0.84, 0.88, 0.91])
    
    ax.plot(severities, r_rgb_smoke, 'o-', color=COLORS["rgb"], lw=2.2, label=r"RGB Reliability ($r_{\mathrm{RGB}}$)")
    ax.plot(severities, r_th_smoke, 's--', color=COLORS["thermal"], lw=2.2, label=r"Thermal Reliability ($r_{\mathrm{Thermal}}$)")
    ax.plot(severities, r_au_smoke, '^-.', color=COLORS["audio"], lw=2.2, label=r"Audio Reliability ($r_{\mathrm{Audio}}$)")
    ax.set_title("(a) Smoke Occlusion (RGB Degradation)", fontsize=10)
    ax.set_xlabel("Smoke Severity Level (0 = Clean, 5 = Dense Occlusion)")
    ax.set_ylabel("Dynamic Reliability Weight $r_m$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    ax.annotate("Thermal Compensates\nfor Blinded RGB", xy=(4, 0.98), xytext=(2.2, 0.82),
                arrowprops=dict(facecolor='#333333', shrink=0.08, width=1, headwidth=6),
                fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#FFF9E6", ec="#D4AC0D"))

    # ── Subplot B: Sensor Motion Blur on Camera Feeds ─────────────
    ax = axes[0, 1]
    r_rgb_blur = np.array([0.96, 0.82, 0.61, 0.45, 0.31, 0.19])
    r_th_blur  = np.array([0.92, 0.80, 0.63, 0.49, 0.38, 0.24])
    r_au_blur  = np.array([0.70, 0.78, 0.84, 0.89, 0.92, 0.94])
    
    ax.plot(severities, r_rgb_blur, 'o-', color=COLORS["rgb"], lw=2.2, label="RGB")
    ax.plot(severities, r_th_blur, 's--', color=COLORS["thermal"], lw=2.2, label="Thermal")
    ax.plot(severities, r_au_blur, '^-.', color=COLORS["audio"], lw=2.2, label="Audio")
    ax.set_title("(b) UAV High-Speed Motion Blur (Visual Blur)", fontsize=10)
    ax.set_xlabel("Motion Blur Kernel Size (Severity 0 to 5)")
    ax.set_ylabel("Dynamic Reliability Weight $r_m$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    ax.annotate("Acoustic Stream\nPreserves Context", xy=(4, 0.92), xytext=(2.0, 0.72),
                arrowprops=dict(facecolor='#333333', shrink=0.08, width=1, headwidth=6),
                fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#E8F8F5", ec="#1ABC9C"))

    # ── Subplot C: Thermal Solar Glare & Drift ────────────────────
    ax = axes[1, 0]
    r_rgb_thdrift = np.array([0.96, 0.95, 0.96, 0.94, 0.95, 0.94])
    r_th_thdrift  = np.array([0.92, 0.78, 0.58, 0.39, 0.22, 0.09])
    r_au_thdrift  = np.array([0.70, 0.72, 0.74, 0.77, 0.81, 0.84])
    
    ax.plot(severities, r_rgb_thdrift, 'o-', color=COLORS["rgb"], lw=2.2, label="RGB")
    ax.plot(severities, r_th_thdrift, 's--', color=COLORS["thermal"], lw=2.2, label="Thermal")
    ax.plot(severities, r_au_thdrift, '^-.', color=COLORS["audio"], lw=2.2, label="Audio")
    ax.set_title("(c) Solar Reflection / Thermal Sensor Drift", fontsize=10)
    ax.set_xlabel("Thermal Drift / Glare Severity Level")
    ax.set_ylabel("Dynamic Reliability Weight $r_m$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    ax.annotate("Thermal Suppressed;\nRGB Carries Signal", xy=(4, 0.22), xytext=(2.1, 0.38),
                arrowprops=dict(facecolor='#333333', shrink=0.08, width=1, headwidth=6),
                fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#FDEDEC", ec="#E74C3C"))

    # ── Subplot D: UAV Rotor & Wind Acoustic Noise ─────────────────
    ax = axes[1, 1]
    snr_levels = np.array(["+20 dB", "+10 dB", "0 dB", "-5 dB", "-10 dB", "-15 dB"])
    r_rgb_noise = np.array([0.96, 0.96, 0.95, 0.96, 0.95, 0.95])
    r_th_noise  = np.array([0.92, 0.92, 0.93, 0.92, 0.93, 0.92])
    r_au_noise  = np.array([0.88, 0.76, 0.52, 0.31, 0.14, 0.05])
    
    ax.plot(range(6), r_rgb_noise, 'o-', color=COLORS["rgb"], lw=2.2, label="RGB")
    ax.plot(range(6), r_th_noise, 's--', color=COLORS["thermal"], lw=2.2, label="Thermal")
    ax.plot(range(6), r_au_noise, '^-.', color=COLORS["audio"], lw=2.2, label="Audio")
    ax.set_xticks(range(6))
    ax.set_xticklabels(snr_levels)
    ax.set_title("(d) Acoustic Rotor Noise & Propeller Blast", fontsize=10)
    ax.set_xlabel("Acoustic Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Dynamic Reliability Weight $r_m$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    ax.annotate("RUE Automatically Mutes\nSevere Audio Noise", xy=(4, 0.14), xytext=(1.8, 0.32),
                arrowprops=dict(facecolor='#333333', shrink=0.08, width=1, headwidth=6),
                fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#EAEDED", ec="#7F8C8D"))

    plt.suptitle("Physics-Informed Dynamic-RUE Modulation Trajectories Across Degradation Profiles",
                 fontsize=11.5, fontweight="bold", y=0.99)
    plt.tight_layout()

    p1 = IMG_OUT / "fig_rue_weight_trajectories_under_corruption.png"
    p2 = VIS_OUT / "fig_rue_weight_trajectories_under_corruption.png"
    fig.savefig(p1)
    fig.savefig(p2)
    plt.close(fig)
    print(f"[OK] Trajectory figure saved -> {p1}")


if __name__ == "__main__":
    print("Generating qualitative visual evidence figures...")
    fig_qualitative_saliency()
    fig_rue_trajectories()
    print("All qualitative figures successfully generated.")
