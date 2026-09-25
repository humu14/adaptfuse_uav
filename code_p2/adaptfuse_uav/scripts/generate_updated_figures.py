"""
Generate completely modernized, publication-grade figures for UniAdapFuse-UAV thesis/manuscript.
Reflects the unified Object Detection + Dynamic Scene-Prior Gating methodology.
Output -> e:\\Research Projects\\reserch writing\\Co-Sup\\images\\generated\\ and images\\
"""

import os
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np

# ── Real-data assets used as insets in schematic diagrams ──────────────────
IMG_ROOT = Path(r"e:\Research Projects\reserch writing\Co-Sup\images")
ASSET_MEL_SPEC   = IMG_ROOT / "sample_mel_spectrogram.png"
ASSET_RGB_FIRE   = IMG_ROOT / "sample_rgb_fire.jpg"
ASSET_THERMAL_FIRE = IMG_ROOT / "sample_thermal_fire.png"
ASSET_DFIRE_CROP = IMG_ROOT / "generated" / "inset_fire_detection.png"


# ══════════════════════════════════════════════════════════════════════════════
# Conference-style diagram primitives (shared by the three schematic figures)
# ══════════════════════════════════════════════════════════════════════════════

def draw_stream_panel(ax, x, y, w, h, title, number, color, subtitle=None):
    """Rounded, dashed, color-coded section panel with a numbered badge, in the
    style of top-venue system diagrams (e.g. a stage/stream grouping box)."""
    panel = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
        facecolor=color, edgecolor=color, alpha=0.08, linewidth=0, zorder=0,
    )
    ax.add_patch(panel)
    border = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
        facecolor="none", edgecolor=color, linewidth=1.6,
        linestyle=(0, (5, 3)), zorder=1,
    )
    ax.add_patch(border)
    badge_r = 0.24
    badge = Circle((x + 0.32, y + h - 0.32), badge_r, facecolor=color, edgecolor="white", linewidth=1.2, zorder=5)
    ax.add_patch(badge)
    ax.text(x + 0.32, y + h - 0.32, str(number), ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white", zorder=6)
    ax.text(x + 0.68, y + h - 0.32, title, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=color, zorder=6)
    if subtitle:
        ax.text(x + 0.68, y + h - 0.62, subtitle, ha="left", va="center",
                fontsize=7.3, style="italic", color="#666666", zorder=6)


def draw_node(ax, x, y, w, h, title, subtitle="", fc="#FFFFFF", ec="#333333",
              lw=1.8, fs=9.0, tc="#111122", sc="#555577", shadow=True, zorder=3):
    """Rounded solid node with an optional soft drop shadow, matching the
    clean card style of the reference architecture figure."""
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.045, y - 0.045), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#000000", edgecolor="none", alpha=0.10, zorder=zorder - 1,
        )
        ax.add_patch(sh)
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(rect)
    cy = y + h / 2
    if subtitle:
        ax.text(x + w / 2, cy + h * 0.16, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)
        ax.text(x + w / 2, cy - h * 0.20, subtitle, ha="center", va="center",
                fontsize=fs - 1.1, color=sc, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)
    return rect


def draw_arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.6, style="-|>", ls="solid", zorder=2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, mutation_scale=11),
        zorder=zorder,
    )


def add_real_inset(ax, img_path, xy, zoom=0.16, border_color="#333333", zorder=6):
    """Places a small real-data thumbnail (spectrogram/RGB/thermal/detection
    crop) at a data coordinate with a clean bordered frame, matching the
    reference figure's density-map/heatmap inset style."""
    try:
        img = mpimg.imread(str(img_path))
    except Exception:
        return None
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox, xy, frameon=True, pad=0.22,
        bboxprops=dict(edgecolor=border_color, linewidth=1.5, boxstyle="round,pad=0.15"),
        zorder=zorder,
    )
    ax.add_artist(ab)
    return ab


def draw_reliability_mini_bar(ax, x, y, w, h, values, labels=("RGB", "Th", "Au"),
                                colors=("#2271B5", "#C0392B", "#0D9E73")):
    """Small real-valued horizontal bar chart used as an inset, grounded in
    the thesis's measured RUE reliability averages (Table 5.4)."""
    n = len(values)
    bar_h = h / n * 0.6
    gap = h / n
    for i, (v, lbl, c) in enumerate(zip(values, labels, colors)):
        by = y + h - (i + 1) * gap + (gap - bar_h) / 2
        ax.add_patch(Rectangle((x, by), w * 0.16, bar_h, facecolor="#E8E8E8", edgecolor="none"))
        ax.add_patch(Rectangle((x, by), w * 0.16 * v, bar_h, facecolor=c, edgecolor="none"))
        ax.text(x - 0.05, by + bar_h / 2, lbl, ha="right", va="center", fontsize=6.3, color="#333333")
        ax.text(x + w * 0.16 + 0.04, by + bar_h / 2, f"{v:.3f}", ha="left", va="center", fontsize=6.0, color="#333333")

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
THESIS_IMG = Path(r"e:\Research Projects\reserch writing\Co-Sup\images")
OUT        = THESIS_IMG / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# ── Publication-quality styling ───────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.95,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
})

def save(fig, name, tight=True):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  [OK] Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 -- UniAdapFuse-UAV Unified Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════════
def generate_fig_01_architecture():
    fig, ax = plt.subplots(figsize=(16.2, 7.6))
    ax.set_xlim(0, 23.0); ax.set_ylim(0, 10.2)
    ax.axis("off")
    ax.set_aspect("equal")

    bx = lambda *a, **k: draw_node(ax, *a, **k)

    def ar(x1, y1, x2, y2, c="#444444", lw=1.6, ls="solid", z=2):
        draw_arrow(ax, x1, y1, x2, y2, color=c, lw=lw, ls=ls, zorder=z)

    # ── Numbered, dashed, color-coded stream panels ──
    draw_stream_panel(ax, 0.2, 5.35, 22.5, 4.55, "DENSE AERIAL OBJECT DETECTION STREAM",
                       1, "#C0392B", subtitle="60% task weightage — spatial hazard localization")
    draw_stream_panel(ax, 0.2, 0.3, 22.5, 4.6, "DYNAMIC SCENE-PRIOR & RELIABILITY GATING STREAM",
                       2, "#2271B5", subtitle="40% task weightage — global scene trust + false-alarm control")

    # ── Sensor inputs ──
    bx(0.5, 7.8, 2.0, 0.9, "RGB Camera", "224×224×3 (High-Res)", fc="#D6E9FF", ec="#2271B5")
    bx(0.5, 6.2, 2.0, 0.9, "Thermal IR", "224×224×1 (LWIR)", fc="#FFDCD4", ec="#C0392B")
    bx(0.5, 1.2, 2.0, 0.9, "Microphone", "64×63 Log-Mel Spec", fc="#D4F4E5", ec="#0D9E73")
    add_real_inset(ax, ASSET_MEL_SPEC, (1.5, 2.55), zoom=0.10, border_color="#0D9E73")

    # ── Hazard Prior Gate (HPG) ──
    bx(3.1, 7.8, 2.2, 1.2, "Hazard Prior Gate (HPG)", "2R-G-B & Sat. Residuals", fc="#FFF0CC", ec="#E69F00", lw=2.0)
    ar(2.5, 8.25, 3.1, 8.25, c="#2271B5", lw=1.5)
    ax.text(4.2, 7.25, "Stride-4 Physics Induction\n(2.3×–4.0× speedup)", ha="center", fontsize=7.2, color="#B35400", style="italic")

    # ── Visual / audio backbones ──
    bx(3.1, 5.8, 2.2, 1.2, "Visual Backbones", "MobileNetV3 / EfficientNet-B0", fc="#EAEAEA", ec="#555555")
    ar(2.5, 6.65, 3.1, 6.65, c="#C0392B", lw=1.5)
    ar(2.5, 7.9, 3.1, 6.7, c="#2271B5", lw=1.0)

    bx(3.1, 1.2, 2.2, 0.9, "Audio Encoder", "2D-CNN / GDBlock", fc="#D4F4E5", ec="#0D9E73")
    ar(2.5, 1.65, 3.1, 1.65, c="#0D9E73", lw=1.5)

    # ── Multi-scale feature pyramids ──
    bx(6.0, 7.4, 2.0, 1.7, "Feature Pyramids", "P3, P4, P5 (Multi-Scale)", fc="#EEF2FF", ec="#3C5488")
    ar(5.3, 8.4, 6.0, 8.4, c="#E69F00", lw=1.5)
    ar(5.3, 6.4, 6.0, 7.8, c="#555555", lw=1.5)

    # Stop-Gradient Isolation
    bx(8.5, 7.6, 2.1, 1.3, "Stop-Gradient\nIsolation (.detach())", "Shields Perceptual Reps", fc="#FFEAEA", ec="#D62728", lw=2.0)
    ar(8.0, 8.25, 8.5, 8.25, c="#D62728", lw=2.0)
    ax.text(9.55, 7.02, "Eliminates Multi-Task\nNegative Transfer (+45.5pp)", ha="center", fontsize=7.2, color="#A80000", fontweight="bold")

    # Top-Down Context Gate (TDCG)
    bx(11.2, 7.6, 2.2, 1.3, "Top-Down Context\nGate (TDCG)", "FiLM: γ·F + β", fc="#FFF3E0", ec="#F57C00", lw=1.8)
    ar(10.6, 8.25, 11.2, 8.25, c="#D62728", lw=1.5)

    # Detection neck & head
    bx(14.0, 7.6, 2.3, 1.3, "FPN + PAN Neck &\nDetection Head", "Anchor-Free Regression", fc="#FFE4E4", ec="#D62728", lw=2.0)
    ar(13.4, 8.25, 14.0, 8.25, c="#F57C00", lw=1.5)

    # Detection outputs (with a real detection-crop inset, grounded in the
    # actual trained YOLO26s(CLAHE) checkpoint rather than a mock box)
    bx(16.9, 8.4, 3.1, 0.7, "Flame Boxes (2D)", "77.1% AP$_{50}$ (fire class)", fc="#FFD0D0", ec="#D62728", fs=8.3)
    bx(16.9, 7.4, 3.1, 0.7, "Smoke Plumes (2D)", "84.85% AP$_{50}$ (smoke)", fc="#FFD0D0", ec="#D62728", fs=8.3)
    bx(16.9, 6.4, 3.1, 0.7, "Survivor RoIs (2D)", "Spatial Triage (Future Work)", fc="#FFD0D0", ec="#D62728", fs=8.3)
    ar(16.3, 8.4, 16.9, 8.75, c="#D62728", lw=1.2)
    ar(16.3, 8.25, 16.9, 7.75, c="#D62728", lw=1.2)
    ar(16.3, 8.1, 16.9, 6.75, c="#D62728", lw=1.2)
    add_real_inset(ax, ASSET_DFIRE_CROP, (21.4, 7.55), zoom=0.14, border_color="#D62728")

    # ── Quality tokens ──
    bx(6.0, 2.8, 1.8, 0.9, "Quality Tokens", "16-dim Per-Sensor", fc="#FFF8E0", ec="#E69F00")
    ar(5.3, 6.0, 6.0, 3.4, c="#555555", lw=1.0, ls=":")
    ar(5.3, 1.65, 6.0, 3.0, c="#0D9E73", lw=1.0, ls=":")

    # RUE box + real measured reliability values as an inset mini-bar
    bx(8.5, 2.2, 2.1, 1.8, "Reliability Estimator\n(RUE)", "$r_{rgb}, r_{th}, r_{audio} \\in [0,1]$", fc="#FFF0CC", ec="#E69F00", lw=2.2)
    ar(7.8, 3.25, 8.5, 3.25, c="#E69F00", lw=1.5)
    draw_reliability_mini_bar(ax, 8.75, 0.55, 1.55, 1.35, values=[0.972, 0.972, 0.006])
    ax.text(9.52, 2.02, "Measured test-set average", ha="center", fontsize=5.8, color="#888888", style="italic")

    # Bridge: RUE → TDCG
    ar(9.55, 4.0, 11.8, 7.6, c="#E69F00", lw=1.8, ls="--")
    ax.text(11.1, 5.6, "RUE Trust Conditioning\n(Suppresses Corrupted Boxes)", ha="center", fontsize=7.6, color="#D35400", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF9E6", ec="#E69F00", lw=0.9))

    # Cross-modal attention
    bx(11.2, 2.2, 2.2, 1.8, "Cross-Modal\nAttention", "4-Head Token Fusion\n(192-dim)", fc="#E4F8EF", ec="#0D9E73", lw=1.8)
    ar(10.6, 3.1, 11.2, 3.1, c="#E69F00", lw=1.5)
    ar(5.3, 5.6, 11.2, 3.5, c="#3C5488", lw=0.9, ls=":")

    # Temporal GRU + Nuisance Head
    bx(14.0, 2.2, 2.3, 1.8, "GRU & Nuisance\nAware Head", "5-Class False Alarm\nDisentanglement", fc="#FFE4E4", ec="#D62728", lw=2.0)
    ar(13.4, 3.1, 14.0, 3.1, c="#0D9E73", lw=1.5)

    # Classification outputs
    bx(16.9, 3.6, 3.1, 0.7, "Disaster Class (4-Class)", "99.44% Accuracy", fc="#EBF3FB", ec="#2271B5", fs=8.3)
    bx(16.9, 2.6, 3.1, 0.7, "Victim Presence (Binary)", "98.36% Macro F1", fc="#EBF3FB", ec="#2271B5", fs=8.3)
    bx(16.9, 1.6, 3.1, 0.7, "Nuisance Type (5-Class)", "0.73% False Positive Rate", fc="#FFF8E0", ec="#E69F00", fs=8.3)
    ar(16.3, 3.3, 16.9, 3.95, c="#2271B5", lw=1.2)
    ar(16.3, 3.1, 16.9, 2.95, c="#2271B5", lw=1.2)
    ar(16.3, 2.9, 16.9, 1.95, c="#E69F00", lw=1.2)

    # BUSG bridge
    ar(15.15, 7.6, 15.15, 4.0, c="#9467BD", lw=1.8, ls="--")
    ax.text(15.8, 5.6, "Bottom-Up Spatial\nGuidance (BUSG)", ha="center", fontsize=7.6, color="#6C3483", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F4ECF7", ec="#9467BD", lw=0.9))

    ax.set_title("UniAdapFuse-UAV: Unified Multimodal Fusion & Scene-Prior Gating Framework",
                 fontsize=13, fontweight="bold", pad=14, color="#111122")
    save(fig, "fig_01_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 -- Dual Benchmark Performance: OD SOTA Suite + 20-Model Master Table
# ══════════════════════════════════════════════════════════════════════════════
def generate_fig_03_performance():
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.30, left=0.07, right=0.96, top=0.92, bottom=0.12)

    # ── Left: D-Fire object detection (7 models, Table 6.1) ──
    ax1 = fig.add_subplot(gs[0])
    od_models = [
        "YOLO26s (CLAHE)",
        "RT-DETRv2-R18",
        "YOLO26n (CLAHE FT)",
        "YOLO11n",
        "YOLO26n",
        "Hazard-YOLO26n",
        "YOLOv10n",
    ]
    map50 = [77.10, 73.80, 72.58, 72.25, 71.47, 70.53, 67.75]
    # Hazard-YOLO26n: sum of three resumed segments in results.csv
    # (2.57 h + 4.13 h + 1.94 h); the summary JSON's 1.96 h is the last segment only.
    # YOLO26n (CLAHE FT) is fine-tuned from the YOLO26n run, so its time is additional.
    train_hrs = [42.6, 7.85, 4.28, 4.60, 4.45, 8.65, 5.07]
    smoke_ap = [84.85, 81.20, 79.64, 79.10, 78.33, 78.12, 75.28]
    fire_ap  = [69.36, 66.40, 65.51, 65.40, 64.60, 62.94, 60.23]

    y_pos = np.arange(len(od_models))
    w = 0.36
    ax1.barh(y_pos - w/2, smoke_ap, height=w, color="#E64B35", alpha=0.85, label="Smoke")
    ax1.barh(y_pos + w/2, fire_ap,  height=w, color="#F48024", alpha=0.85, label="Fire")
    # right of bars: mAP50 and wall-clock training time
    for i in range(len(od_models)):
        ax1.text(max(smoke_ap[i], fire_ap[i]) + 1.2, y_pos[i], f"{map50[i]:.2f} | {train_hrs[i]} h",
                 va="center", fontsize=7.5, color="#333333")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(od_models, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(50, 100)
    ax1.set_xlabel(r"AP$_{50}$ (%)")
    ax1.set_title("(A) D-Fire detection", fontweight="bold", fontsize=9.5)
    ax1.legend(loc="lower right", fontsize=8.0)

    # ── Right: scene classification incl. joint-prototype recovery ──
    ax2 = fig.add_subplot(gs[1])
    categories = [
        "EfficientNet-B0 (RGB)",
        "RGB-only (A1)",
        "AdapFuse-v1",
        "UniAdapFuse v2",
        "AdapFuse-v2 (KD)",
        "Fixed (D)",
        "Early (B)",
        "UniAdapFuse v1 (naive)",
    ]
    comb_f1 = [99.24, 98.85, 98.74, 98.57, 98.61, 98.69, 96.93, 73.25]
    dis_acc = [99.40, 99.51, 99.44, 99.26, 99.44, 99.47, 98.90, 53.71]

    y_pos2 = np.arange(len(categories))
    w2 = 0.36
    ax2.barh(y_pos2 - w2/2, comb_f1, height=w2, color="#2271B5", alpha=0.85, label="Combined F1")
    ax2.barh(y_pos2 + w2/2, dis_acc, height=w2, color="#00A087", alpha=0.85, label="Disaster acc.")

    # label only the collapsed naive-joint row; others sit near 99%
    for yy, v in ((y_pos2[-1] - w2/2, comb_f1[-1]), (y_pos2[-1] + w2/2, dis_acc[-1])):
        ax2.text(v + 0.8, yy, f"{v:.2f}", va="center", fontsize=7.5, color="#333333")

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(categories, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(45, 105)
    ax2.set_xlabel("Score (%)")
    ax2.set_title("(B) Scene classification", fontweight="bold", fontsize=9.5)
    ax2.legend(loc="lower right", fontsize=8.0)

    save(fig, "fig_03_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 -- Environmental Degradation and Robustness Stress Testing
# ══════════════════════════════════════════════════════════════════════════════
def generate_fig_05_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    plt.subplots_adjust(wspace=0.28, left=0.08, right=0.96, top=0.88, bottom=0.14)

    # ── Left: Multi-Axis Corruption Curves ──
    ax1 = axes[0]
    severities = [1, 2, 3, 4, 5]
    
    # AdapFuse-v1 & UniAdapFuse v2 under Smoke
    f1_adapt_smoke   = [98.7, 97.5, 95.2, 92.4, 89.7]
    f1_rgb_smoke     = [98.8, 88.2, 68.4, 48.1, 34.2]
    f1_static_smoke  = [98.6, 89.4, 72.1, 54.3, 41.5]
    f1_adapt_noise   = [98.7, 98.7, 98.6, 98.6, 98.5] # Protected by RUE
    f1_ungated_noise = [98.6, 96.2, 94.1, 92.5, 91.8]

    ax1.plot(severities, f1_adapt_smoke, "o-", color="#D62728", lw=2.4, label="AdapFuse-v1 (Smoke Occlusion)")
    ax1.plot(severities, f1_adapt_noise, "s-", color="#0D9E73", lw=2.0, label="AdapFuse-v1 (Rotor Noise: r_au->0)")
    ax1.plot(severities, f1_static_smoke, "^--", color="#3C5488", lw=1.6, label="Intermediate Fixed (Smoke)")
    ax1.plot(severities, f1_rgb_smoke, "x--", color="#2271B5", lw=1.6, label="RGB-only (Smoke Collapse)")
    ax1.plot(severities, f1_ungated_noise, "v:", color="#7B7B7B", lw=1.4, label="Ungated Audio Attention (Noise)")

    ax1.set_xlabel("Degradation Severity (1 = Mild -> 5 = Catastrophic / 88% Smoke)")
    ax1.set_ylabel("Macro Combined F1 (%)")
    ax1.set_ylim(25, 102)
    ax1.set_title("(A) Environmental Corruption Stress Trajectories\nDynamic RUE Gating vs Static Baselines", fontweight="bold", fontsize=9.5)
    ax1.legend(loc="lower left", fontsize=7.8)

    # ── Right: Missing-Modality Sensor Dropout ──
    ax2 = axes[1]
    dropout_conditions = ["Full Tri-Modal", "RGB Dropped", "Thermal Dropped", "Audio Dropped", "Visual Only (No Mic)"]
    adapt_f1  = [98.74, 97.43, 98.49, 98.50, 98.50]
    static_f1 = [98.69, 52.10, 61.40, 98.20, 98.10]
    rgb_f1    = [98.85,  0.00, 98.85, 98.85, 98.85]

    x = np.arange(len(dropout_conditions))
    w = 0.26

    ax2.bar(x - w, adapt_f1,  width=w, color="#D62728", alpha=0.9, label="AdapFuse-v1 (Dynamic Gating)")
    ax2.bar(x,     static_f1, width=w, color="#3C5488", alpha=0.8, label="Intermediate Fixed D")
    ax2.bar(x + w, rgb_f1,    width=w, color="#2271B5", alpha=0.7, label="RGB-only Baseline A1")

    ax2.set_xticks(x)
    ax2.set_xticklabels(dropout_conditions, rotation=25, ha="right", fontsize=8.0)
    ax2.set_ylabel("Macro Combined F1 (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("(B) Sensor Dropout Resilience\nGraceful Degradation Under Complete Camera Loss", fontweight="bold", fontsize=9.5)
    ax2.legend(loc="lower left", fontsize=7.8)

    fig.suptitle("Environmental Robustness and Fault-Tolerant Sensor Gating", fontsize=11, fontweight="bold", y=0.98)
    save(fig, "fig_05_robustness.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 -- Design Comparison: Design D vs Design E vs UniAdapFuse v2
# ══════════════════════════════════════════════════════════════════════════════
def generate_fig_09_design_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": [1, 1.2, 1.3]})
    plt.subplots_adjust(wspace=0.16, left=0.02, right=0.99, top=0.86, bottom=0.05)

    def bx(ax, *a, **k):
        return draw_node(ax, *a, **k)

    def ar(ax, x1, y1, x2, y2, c="#555555", lw=1.4, ls="solid"):
        draw_arrow(ax, x1, y1, x2, y2, color=c, lw=lw, ls=ls)

    panel_specs = [
        ("Design D: Intermediate Fixed Fusion", "#7F8C8D"),
        ("Design E: AdapFuse-v1 (Teacher)", "#E69F00"),
        ("Proposed: UniAdapFuse v2 (Unified)", "#0D9E73"),
    ]

    for ax_i, ax in enumerate(axes):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_aspect("equal")
        title, color = panel_specs[ax_i]
        draw_stream_panel(ax, 0.05, 0.05, 9.9, 9.5, title, ax_i + 1, color)

        # Draw 3 sensor inputs (shared across all three designs)
        bx(ax, 0.3, 7.3, 2.2, 1.2, "RGB Sensor", "MobileNetV3", fc="#D6E9FF", ec="#2271B5")
        bx(ax, 0.3, 4.4, 2.2, 1.2, "Thermal IR", "ResNet-18", fc="#FFDCD4", ec="#C0392B")
        bx(ax, 0.3, 1.5, 2.2, 1.2, "Audio Stream", "2D-CNN", fc="#D4F4E5", ec="#0D9E73")

        if ax_i == 0:  # Design D
            add_real_inset(ax, ASSET_MEL_SPEC, (0.75, 0.85), zoom=0.045, border_color="#0D9E73")
            bx(ax, 3.8, 4.0, 2.5, 1.8, "Fixed Concat\n(Equal Weights)", "192-dim", fc="#EEF2FF", ec="#3C5488")
            ar(ax, 2.5, 7.9, 3.8, 5.3, "#2271B5")
            ar(ax, 2.5, 5.0, 3.8, 4.9, "#C0392B")
            ar(ax, 2.5, 2.1, 3.8, 4.5, "#0D9E73")

            bx(ax, 7.1, 4.3, 2.5, 1.2, "MLP Classifier", "Disaster + Victim", fc="#FFE4E4", ec="#D62728")
            ar(ax, 6.3, 4.9, 7.1, 4.9, "#3C5488")
            ax.text(5.0, 0.4, "Static scalar weights — blind to sensor failure",
                    ha="center", fontsize=7.6, color="#888888", style="italic",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#F9F9F9", ec="#CCCCCC"))

        elif ax_i == 1:  # Design E
            bx(ax, 3.2, 2.6, 2.0, 4.6, "RUE Gating\n+ Cross-Attn", "", fc="#FFF0CC", ec="#E69F00", lw=1.8)
            draw_reliability_mini_bar(ax, 3.5, 2.75, 1.5, 1.35, values=[0.972, 0.972, 0.006])
            ar(ax, 2.5, 7.9, 3.2, 6.1, "#2271B5")
            ar(ax, 2.5, 5.0, 3.2, 4.9, "#C0392B")
            ar(ax, 2.5, 2.1, 3.2, 3.7, "#0D9E73")

            bx(ax, 5.9, 3.6, 2.2, 2.4, "GRU &\nNuisanceHead", "3 Tasks", fc="#FFE4E4", ec="#D62728", lw=1.8)
            ar(ax, 5.2, 4.9, 5.9, 4.8, "#E69F00")

            bx(ax, 8.4, 5.8, 1.5, 0.9, "Disaster", "0.0020 Loss", fc="#EBF3FB", ec="#2271B5", fs=7.2)
            bx(ax, 8.4, 4.3, 1.5, 0.9, "Victim", "98.36% F1", fc="#EBF3FB", ec="#2271B5", fs=7.2)
            bx(ax, 8.4, 2.8, 1.5, 0.9, "Nuisance", "0.73% FPR", fc="#FFF8E0", ec="#E69F00", fs=7.2)
            ar(ax, 8.1, 5.2, 8.4, 6.25, "#2271B5")
            ar(ax, 8.1, 4.8, 8.4, 4.75, "#2271B5")
            ar(ax, 8.1, 4.3, 8.4, 3.25, "#E69F00")

            ax.text(5.0, 0.4, "Dynamic reliability weighting — auxiliary false-alarm modeling",
                    ha="center", fontsize=7.6, color="#C0392B", style="italic",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#FFF5F0", ec="#E8BBBB"))

        else:  # UniAdapFuse v2
            bx(ax, 2.8, 7.3, 1.6, 1.2, "HPG", "Physics", fc="#FFF0CC", ec="#E69F00", fs=7.4)
            ar(ax, 2.5, 7.9, 2.8, 7.9, "#2271B5")

            bx(ax, 4.9, 6.6, 2.3, 2.0, "FPN Neck &\nDetection Head", "Stop-Grad .detach()", fc="#FFE4E4", ec="#D62728", lw=1.8)
            ar(ax, 4.4, 7.9, 4.9, 7.9, "#E69F00")
            ar(ax, 2.5, 5.0, 4.9, 7.1, "#C0392B", ls=(0, (5, 3)))

            bx(ax, 7.9, 6.8, 1.9, 1.6, "Dense 2D Boxes\n(Fire, Smoke)", "77.10% mAP$_{50}$", fc="#FFD0D0", ec="#D62728", fs=7.4)
            ar(ax, 7.2, 7.6, 7.9, 7.6, "#D62728")
            add_real_inset(ax, ASSET_DFIRE_CROP, (9.1, 9.15), zoom=0.07, border_color="#D62728")

            bx(ax, 4.9, 1.9, 2.3, 2.2, "RUE + Attn &\nNuisanceHead", "Scene Prior", fc="#EBF3FB", ec="#2271B5", lw=1.8)
            ar(ax, 2.5, 2.1, 4.9, 2.7, "#0D9E73")
            ar(ax, 2.5, 4.7, 4.9, 3.4, "#C0392B", ls=(0, (2, 2)))

            bx(ax, 7.9, 2.1, 1.9, 1.6, "Scene Triage\n(98.57% F1)", "0.0337 ECE", fc="#EBF3FB", ec="#2271B5", fs=7.4)
            ar(ax, 7.2, 3.0, 7.9, 3.0, "#2271B5")

            ar(ax, 6.0, 4.1, 6.0, 6.6, "#F57C00", lw=1.5)
            ax.text(6.5, 5.3, "TDCG", fontsize=7.6, fontweight="bold", color="#D35400")

            ax.text(5.0, 0.4, "Joint dense localization + scene prior — zero negative transfer (+45.5pp)",
                    ha="center", fontsize=7.6, color="#0D9E73", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#EAFAF1", ec="#0D9E73"))

    fig.suptitle("Evolution of Fusion Paradigms: From Fixed Concatenation to UniAdapFuse v2",
                 fontsize=13, fontweight="bold", y=0.98)
    save(fig, "fig_09_design_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1.1 -- End-to-End UAV Disaster Framework Pipeline (images/uav_multimodal_framework.png)
# ══════════════════════════════════════════════════════════════════════════════
def generate_fig_uav_framework():
    fig, ax = plt.subplots(figsize=(16.0, 6.6))
    ax.set_xlim(0, 21.4); ax.set_ylim(0, 8.6)
    ax.axis("off")
    ax.set_aspect("equal")

    bx = lambda *a, **k: draw_node(ax, *a, **k)

    def ar(x1, y1, x2, y2, c="#444444", lw=1.5, ls="solid"):
        draw_arrow(ax, x1, y1, x2, y2, color=c, lw=lw, ls=ls)

    stages = [
        ("SENSING & INGESTION", "#2271B5", 0.3, 3.9),
        ("PREPROCESSING & HPG", "#E69F00", 4.5, 3.9),
        ("MULTI-SCALE PERCEPTION", "#D62728", 8.7, 3.9),
        ("DUAL TRIAGE & DEPLOYMENT", "#0D9E73", 12.9, 4.0),
    ]
    for i, (title, color, sx, sw) in enumerate(stages, start=1):
        draw_stream_panel(ax, sx, 0.4, sw, 7.6, title, i, color)

    # ── Stage 1: Sensing ──
    bx(0.6, 5.2, 3.4, 1.2, "UAV Aerial RGB Camera", "4K / 1080p Optical Stream", fc="#FFFFFF", ec="#2271B5")
    bx(0.6, 3.4, 3.4, 1.2, "LWIR Thermal Camera", "320×240 Radiometric IR", fc="#FFFFFF", ec="#C0392B")
    bx(0.6, 1.6, 3.4, 1.2, "Acoustic Array / Mic", "16 kHz Denoised Audio", fc="#FFFFFF", ec="#0D9E73")
    ax.text(2.3, 1.05, "Real capture samples:", ha="center", fontsize=6.6, style="italic", color="#666666")
    add_real_inset(ax, ASSET_RGB_FIRE, (1.55, 0.68), zoom=0.052, border_color="#2271B5")
    add_real_inset(ax, ASSET_THERMAL_FIRE, (2.3, 0.68), zoom=0.052, border_color="#C0392B")
    add_real_inset(ax, ASSET_MEL_SPEC, (3.05, 0.68), zoom=0.038, border_color="#0D9E73")

    ar(4.0, 5.8, 4.6, 5.8, "#2271B5")
    ar(4.0, 4.0, 4.6, 4.0, "#C0392B")
    ar(4.0, 2.2, 4.6, 2.2, "#0D9E73")

    # ── Stage 2: Preprocessing ──
    bx(4.6, 5.2, 3.4, 1.2, "Hazard Prior Gate (HPG)", "2R-G-B Chromaticity Residuals", fc="#FFF9E6", ec="#E69F00", lw=1.8)
    bx(4.6, 3.4, 3.4, 1.2, "CLAHE & Normalization", "Histogram Thermal Equalization", fc="#FFFFFF", ec="#E64B35")
    bx(4.6, 1.6, 3.4, 1.2, "Log-Mel Spectrogram", "Spectral Subtraction (Rotor Denoising)", fc="#FFFFFF", ec="#0D9E73")

    ar(8.0, 5.8, 8.8, 5.8, "#E69F00")
    ar(8.0, 4.0, 8.8, 4.0, "#C0392B")
    ar(8.0, 2.2, 8.8, 2.2, "#0D9E73")

    # ── Stage 3: Multi-scale perception ──
    bx(8.8, 4.8, 3.4, 1.8, "UniAdapFuse Backbone &\nFPN Detection Neck", "Stop-Gradient (.detach())\nCurriculum Warmup", fc="#FFEAEA", ec="#D62728", lw=2.0)
    bx(8.8, 1.6, 3.4, 2.4, "RUE Reliability Gating &\nCross-Modal Attention", "Dynamic Reliability Prior\nNuisance Disentanglement", fc="#FFF0CC", ec="#E69F00", lw=2.0)

    ar(12.2, 5.7, 13.0, 5.7, "#D62728")
    ar(12.2, 2.8, 13.0, 2.8, "#E69F00")

    ar(10.5, 4.0, 10.5, 4.8, "#F57C00", lw=1.6)
    ax.text(11.15, 4.4, "TDCG", fontsize=7.6, fontweight="bold", color="#D35400")

    # ── Stage 4: Dual triage & deployment ──
    bx(13.0, 4.8, 3.1, 1.8, "Dense Object Localization", "2D Smoke & Fire Boxes\n(77.10% mAP$_{50}$, 4.0× speedup)", fc="#FFFFFF", ec="#D62728")
    bx(13.0, 1.6, 3.1, 2.4, "Scene & Survivor Triage", "Disaster Class (99.26% Acc)\nVictim Presence (97.94% F1)\nFalse Alarm Rate: 0.73%", fc="#FFFFFF", ec="#2271B5")
    add_real_inset(ax, ASSET_DFIRE_CROP, (15.3, 7.05), zoom=0.095, border_color="#D62728")

    # ── Edge deployment card ──
    bx(17.0, 2.4, 3.6, 3.2, "Target Edge UAV\nArchitecture",
       "AdapFuse-v1 (teacher): 12.94M params\n14.26 ms / 70.1 FPS (workstation)\nAdapFuse-v2 (student): 12.30M params\n(edge-GPU profiling: future work)",
       fc="#2E7D32", ec="#1B5E20", lw=2.0, tc="#FFFFFF", sc="#E8F5E9", fs=8.6)
    ar(16.1, 5.7, 17.0, 4.6, "#0D9E73", lw=1.5)
    ar(16.1, 2.8, 17.0, 3.5, "#0D9E73", lw=1.5)

    ax.set_title("End-to-End UAV Multimodal Disaster Perception & Dense Aerial Localization Framework",
                 fontsize=12.5, fontweight="bold", pad=14)

    out_path = THESIS_IMG / "uav_multimodal_framework.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved -> {out_path}")


# ── Execution ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating updated, publication-grade figures...")
    generate_fig_01_architecture()
    generate_fig_03_performance()
    generate_fig_05_robustness()
    generate_fig_09_design_comparison()
    generate_fig_uav_framework()
    print("All figures successfully regenerated and saved!")
