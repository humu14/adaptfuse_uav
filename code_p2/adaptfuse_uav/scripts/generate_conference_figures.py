"""Generate evidence-grounded, conference-style thesis figures.

The figures in this script intentionally distinguish sensor measurements from
derived or label-matched modalities.  All image panels come from local dataset
files or real experiment outputs; no synthetic image generator is used.

Outputs
-------
images/uav_multimodal_framework.png        (Figure 1.1)
images/generated/fig_01_architecture.png   (Figure 4.1)
images/generated/fig_02_qualitative.png    (Figure 4.2)
images/generated/fig_09_design_comparison.png (Figure 4.5)
images/generated/fig_14_data_provenance.png   (new Figure 4.6)
images/generated/fig_05_robustness.png        (Figure 5.3 integrity fix)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent.parent
OUT = THESIS / "images" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
META = ROOT / "data" / "metadata" / "test.csv"
CKPT = ROOT / "outputs" / "checkpoints" / "adaptfuse_v1_best.pth"

sys.path.insert(0, str(ROOT))
from datasets.multimodal_dataset import MultimodalDisasterDataset  # noqa: E402
from models.full_model import build_model  # noqa: E402


# Colorblind-safe palette close to common CVPR/NeurIPS figure styling.
INK = "#172033"
MUTED = "#667085"
GRID = "#D0D5DD"
BG = "#F8FAFC"
BLUE = "#2F6B9A"
BLUE_BG = "#E8F1F8"
RED = "#C43C39"
RED_BG = "#FBECEB"
TEAL = "#1B8A78"
TEAL_BG = "#E7F5F1"
GOLD = "#B7791F"
GOLD_BG = "#FFF6DF"
PURPLE = "#6D5AA6"
PURPLE_BG = "#F0ECFA"
GREEN = "#2F7D4A"

DISASTER = ["Clean", "Fire / smoke", "Collapse / flood", "Other hazard"]
SAMPLE_IDS = ["flame_030057", "flame_040754", "c2a_000444", "c2a_005602"]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[OK] {path}")


def rounded_box(
    ax,
    xy,
    width,
    height,
    title,
    detail="",
    *,
    edge=BLUE,
    face=BLUE_BG,
    lw=1.2,
    dashed=False,
    title_size=8,
    detail_size=6.8,
    zorder=3,
):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(rect)
    x, y = xy
    if detail:
        ax.text(
            x + width / 2,
            y + height * 0.61,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            zorder=zorder + 1,
        )
        ax.text(
            x + width / 2,
            y + height * 0.30,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=MUTED,
            linespacing=1.15,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x + width / 2,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return rect


def arrow(ax, start, end, *, color=MUTED, lw=1.2, dashed=False, zorder=2):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle="--" if dashed else "-",
            mutation_scale=9,
            shrinkA=1,
            shrinkB=1,
        ),
        zorder=zorder,
    )


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def read_gray(path: str | Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("L"))


def inset_image(ax, arr, extent, *, cmap=None, edge=GRID, label=None):
    x0, x1, y0, y1 = extent
    ax.imshow(arr, extent=extent, cmap=cmap, aspect="auto", zorder=4)
    ax.add_patch(
        patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=edge, linewidth=1, zorder=5
        )
    )
    if label:
        ax.text(
            x0 + 0.006,
            y0 + 0.012,
            label,
            ha="left",
            va="bottom",
            fontsize=5.8,
            color="white",
            fontweight="bold",
            bbox=dict(facecolor=INK, edgecolor="none", alpha=0.78, pad=1.5),
            zorder=6,
        )


def load_evidence():
    df = pd.read_csv(META)
    missing = [s for s in SAMPLE_IDS if s not in set(df.sample_id)]
    if missing:
        raise RuntimeError(f"Missing fixed qualitative samples: {missing}")
    rows = [df[df.sample_id == sample_id].iloc[0] for sample_id in SAMPLE_IDS]
    indices = [int(df.index[df.sample_id == sample_id][0]) for sample_id in SAMPLE_IDS]

    dataset = MultimodalDisasterDataset(str(META), split="test", corruption_prob=0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running qualitative forward passes on {device}", flush=True)
    model = build_model({"model": "adaptfuse_v1"})
    state = torch.load(CKPT, map_location="cpu")
    state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    items = [dataset[idx] for idx in indices]
    with torch.no_grad():
        dlog, vlog, reliability = model(
            torch.stack([item["rgb"] for item in items]).to(device),
            torch.stack([item["thermal"] for item in items]).to(device),
            torch.stack([item["audio"] for item in items]).to(device),
            torch.stack([item["has_rgb"] for item in items]).to(device),
            torch.stack([item["has_thermal"] for item in items]).to(device),
            torch.stack([item["has_audio"] for item in items]).to(device),
        )
    dprob = torch.softmax(dlog, dim=-1).cpu()
    vprob = torch.softmax(vlog, dim=-1).cpu()
    reliability = reliability.cpu()

    evidence = []
    for i, (row, item) in enumerate(zip(rows, items)):
        evidence.append(
            {
                "row": row,
                "item": item,
                "rgb": read_rgb(row.rgb_path),
                "thermal": read_gray(row.thermal_path),
                "audio": item["audio"].squeeze(0).numpy(),
                "pred": int(dprob[i].argmax()),
                "confidence": float(dprob[i].max()),
                "victim_pred": int(vprob[i].argmax()),
                "victim_confidence": float(vprob[i].max()),
                "reliability": reliability[i].numpy(),
            }
        )
    return evidence


def figure_1_overview(evidence):
    """Figure 1.1: separates the classification and D-Fire detection evidence tracks."""
    fire = evidence[1]
    collapse = evidence[2]
    dfire_path = (
        THESIS
        / "adaptfuse_od"
        / "data"
        / "datasets"
        / "dfire"
        / "images"
        / "test"
        / "WEB10547.jpg"
    )
    dfire = read_rgb(dfire_path)

    mel_preview = read_rgb(THESIS / "images" / "sample_mel_spectrogram.png")

    fig, ax = plt.subplots(figsize=(11.2, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.955, "A  Scene-level multimodal study", fontsize=9, fontweight="bold")
    ax.text(0.015, 0.905, "72,997 heterogeneous samples", fontsize=7, color=MUTED)
    inset_image(ax, fire["rgb"], (0.018, 0.108, 0.57, 0.86), label="FLAME RGB")
    inset_image(ax, fire["thermal"], (0.116, 0.206, 0.57, 0.86), cmap="inferno", label="derived thermal")
    ax.imshow(mel_preview, extent=(0.214, 0.304, 0.57, 0.86), aspect="auto", zorder=4)
    ax.add_patch(patches.Rectangle((0.214, 0.57), 0.09, 0.29, fill=False, edgecolor=GRID, lw=1, zorder=5))
    ax.text(0.219, 0.582, "ESC-50\nlabel match", color="white", fontsize=5.4, fontweight="bold", va="bottom", bbox=dict(fc=INK, ec="none", alpha=.78, pad=1.2), zorder=6)

    rounded_box(ax, (0.34, 0.57), 0.145, 0.29, "AdapFuse-v1", "three encoders\nreliability-weighted tokens", edge=PURPLE, face=PURPLE_BG)
    rounded_box(ax, (0.515, 0.57), 0.145, 0.29, "Scene outputs", "4-class disaster\nvictim presence", edge=TEAL, face=TEAL_BG)
    arrow(ax, (0.305, 0.715), (0.338, 0.715), color=BLUE)
    arrow(ax, (0.486, 0.715), (0.513, 0.715), color=PURPLE)
    ax.text(0.588, 0.535, "Measured: 98.74% combined macro-F1", ha="center", fontsize=6.6, color=GREEN, fontweight="bold")

    ax.text(0.015, 0.435, "B  Dense hazard-localization benchmark", fontsize=9, fontweight="bold")
    ax.text(0.015, 0.385, "D-Fire: 21,527 RGB images; smoke/fire boxes", fontsize=7, color=MUTED)
    inset_image(ax, dfire, (0.018, 0.205, 0.06, 0.34), label="D-Fire RGB")
    rounded_box(ax, (0.245, 0.06), 0.155, 0.28, "YOLO26s + CLAHE", "independent detector\n640 px input", edge=RED, face=RED_BG)
    rounded_box(ax, (0.435, 0.06), 0.225, 0.28, "Localized smoke and flame", "77.10% mAP50\n44.80% mAP50:95", edge=GOLD, face=GOLD_BG)
    arrow(ax, (0.208, 0.20), (0.242, 0.20), color=RED)
    arrow(ax, (0.402, 0.20), (0.432, 0.20), color=RED)

    # Explicit evidence boundary and integration status.
    ax.plot([0.70, 0.70], [0.04, 0.90], color=GRID, lw=1)
    ax.text(0.725, 0.865, "Integration status", fontsize=9, fontweight="bold")
    rounded_box(ax, (0.725, 0.58), 0.245, 0.22, "UniAdapFuse v2 prototype", "shared scene/detection graph\nclassification recovered", edge=PURPLE, face=PURPLE_BG)
    rounded_box(ax, (0.725, 0.31), 0.245, 0.17, "Detection validation incomplete", "recorded multi-task mAP50 = 0.0", edge=RED, face="white", dashed=True, detail_size=6.5)
    rounded_box(ax, (0.725, 0.08), 0.245, 0.13, "Embedded UAV deployment", "future validation", edge=MUTED, face=BG, dashed=True, title_size=7.6)
    arrow(ax, (0.848, 0.58), (0.848, 0.49), color=PURPLE)
    arrow(ax, (0.848, 0.31), (0.848, 0.22), color=MUTED, dashed=True)
    ax.text(0.71, 0.02, "Solid = measured in this project     Dashed = unvalidated / future", fontsize=6.3, color=MUTED)

    save(fig, THESIS / "images" / "uav_multimodal_framework.png")


def figure_4_1_architecture(evidence):
    """Figure 4.1: accurate AdapFuse-v1 computational graph with real inputs."""
    fire = evidence[1]
    mel_preview = read_rgb(THESIS / "images" / "sample_mel_spectrogram.png")
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headings.
    for x, text in [(0.035, "Inputs"), (0.18, "Encoders"), (0.35, "Shared tokens"), (0.55, "Adaptive fusion"), (0.79, "Prediction")]:
        ax.text(x, 0.96, text, fontsize=8.5, fontweight="bold", color=INK)
    ax.plot([0.02, 0.98], [0.925, 0.925], color=GRID, lw=0.8)

    rows = [0.67, 0.40, 0.13]
    specs = [
        ("RGB", fire["rgb"], None, BLUE, BLUE_BG, "MobileNetV3-S", "576d"),
        ("Pseudo-thermal", fire["thermal"], "inferno", RED, RED_BG, "ResNet-18", "512d"),
        ("Label-matched audio", mel_preview, None, TEAL, TEAL_BG, "2D-CNN", "128d"),
    ]
    for y, (label, arr, cmap, edge, face, enc, enc_dim) in zip(rows, specs):
        if label == "Label-matched audio":
            ax.imshow(arr, extent=(0.025, 0.125, y, y + 0.20), aspect="auto", cmap=cmap, zorder=4)
        else:
            ax.imshow(arr, extent=(0.025, 0.125, y, y + 0.20), cmap=cmap, aspect="auto", zorder=4)
        ax.add_patch(patches.Rectangle((0.025, y), 0.10, 0.20, fill=False, ec=edge, lw=1.2, zorder=5))
        ax.text(0.075, y - 0.022, label, ha="center", va="top", fontsize=6.2, color=edge, fontweight="bold")
        rounded_box(ax, (0.17, y + 0.025), 0.125, 0.15, enc, enc_dim, edge=edge, face=face, title_size=7.6, detail_size=6.5)
        rounded_box(ax, (0.335, y + 0.025), 0.115, 0.15, "Projection", "LayerNorm, 192d", edge=PURPLE, face=PURPLE_BG, title_size=7.3, detail_size=6.2)
        arrow(ax, (0.127, y + 0.10), (0.168, y + 0.10), color=edge)
        arrow(ax, (0.297, y + 0.10), (0.333, y + 0.10), color=PURPLE)

    rounded_box(ax, (0.49, 0.57), 0.13, 0.26, "RUE gate", "quality MLPs +\nfeature statistics\n3 sigmoid weights", edge=GOLD, face=GOLD_BG, title_size=8.3)
    for y in [0.77, 0.50, 0.23]:
        arrow(ax, (0.452, y), (0.488, 0.70), color=GOLD, lw=0.9)

    rounded_box(ax, (0.655, 0.54), 0.12, 0.32, "Reliability-weighted\n3-token attention", "4 heads; mean pool", edge=TEAL, face=TEAL_BG, title_size=7.5, detail_size=6.5)
    for y, edge in zip([0.77, 0.50, 0.23], [BLUE, RED, TEAL]):
        arrow(ax, (0.452, y), (0.653, 0.70), color=edge, lw=0.9)
    arrow(ax, (0.622, 0.70), (0.653, 0.70), color=GOLD, lw=1.3)

    rounded_box(ax, (0.655, 0.20), 0.12, 0.18, "GRU refinement", "single step (T=1)", edge=PURPLE, face=PURPLE_BG, title_size=7.6, detail_size=6.3)
    arrow(ax, (0.715, 0.54), (0.715, 0.39), color=PURPLE)

    rounded_box(ax, (0.82, 0.62), 0.15, 0.16, "Disaster head", "4 classes", edge=BLUE, face=BLUE_BG)
    rounded_box(ax, (0.82, 0.40), 0.15, 0.16, "Victim head", "presence / absence", edge=RED, face=RED_BG)
    rounded_box(ax, (0.82, 0.14), 0.15, 0.16, "Nuisance head*", "5 logits", edge=MUTED, face=BG, dashed=True)
    for yy, cc in [(0.70, BLUE), (0.48, RED), (0.22, MUTED)]:
        arrow(ax, (0.777, 0.29), (0.818, yy), color=cc, dashed=(cc == MUTED), lw=1.0)

    ax.text(0.49, 0.075, "Weights modulate projected features before attention.", fontsize=6.5, color=MUTED)
    ax.text(0.82, 0.075, "*Defined in the model, but nuisance labels are not supplied by the current trainer.", fontsize=6.0, color=MUTED)
    ax.text(0.025, 0.005, "Displayed inputs are representative project files: held-out FLAME RGB/pseudo-thermal and a label-matched ESC-50 clip.", fontsize=6.3, color=MUTED)

    save(fig, OUT / "fig_01_architecture.png")


def figure_4_2_qualitative(evidence):
    """Figure 4.2: four held-out examples with actual model forward-pass outputs."""
    fig = plt.figure(figsize=(10.9, 8.1))
    gs = fig.add_gridspec(4, 4, width_ratios=[1.15, 1.15, 1.20, 1.55], hspace=0.22, wspace=0.10)
    headers = ["RGB", "Pseudo-thermal", "Audio", "AdapFuse-v1 output"]

    for row_idx, ev in enumerate(evidence):
        row = ev["row"]
        ax_rgb = fig.add_subplot(gs[row_idx, 0])
        ax_rgb.imshow(ev["rgb"])
        ax_rgb.set_axis_off()
        ax_rgb.text(
            0.02,
            0.03,
            f"{row.sample_id}",
            transform=ax_rgb.transAxes,
            fontsize=5.7,
            color="white",
            bbox=dict(fc=INK, ec="none", alpha=0.76, pad=1.5),
        )

        ax_th = fig.add_subplot(gs[row_idx, 1])
        ax_th.imshow(ev["thermal"], cmap="inferno")
        ax_th.set_axis_off()

        ax_audio = fig.add_subplot(gs[row_idx, 2])
        if int(row.has_audio) == 1:
            ax_audio.imshow(ev["audio"], origin="lower", aspect="auto", cmap="magma")
            ax_audio.text(0.02, 0.04, "ESC-50; label matched", transform=ax_audio.transAxes, fontsize=5.7, color="white", bbox=dict(fc=INK, ec="none", alpha=0.76, pad=1.5))
            ax_audio.set_xticks([])
            ax_audio.set_yticks([])
        else:
            ax_audio.set_facecolor(BG)
            ax_audio.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=8, color=MUTED, fontweight="bold")
            ax_audio.set_xticks([])
            ax_audio.set_yticks([])
            for spine in ax_audio.spines.values():
                spine.set_color(GRID)

        ax_out = fig.add_subplot(gs[row_idx, 3])
        ax_out.set_xlim(-0.20, 1)
        ax_out.set_ylim(-0.7, 3.6)
        true_label = DISASTER[int(row.disaster_label)]
        pred_label = DISASTER[ev["pred"]]
        correct = ev["pred"] == int(row.disaster_label)
        pred_text = pred_label if correct else f"{pred_label} (GT: {true_label})"
        ax_out.text(0.00, 2.70, f"{pred_text}  ({ev['confidence']:.3f})", fontsize=8.1, color=GREEN if correct else RED, fontweight="bold")
        ax_out.text(0.00, 2.16, f"Victim: {'present' if ev['victim_pred'] else 'absent'}  ({ev['victim_confidence']:.3f})", fontsize=6.8, color=INK)
        labels = ["RGB", "Pseudo-T", "Audio"]
        colors = [BLUE, RED, TEAL]
        rel = ev["reliability"]
        for i, (lab, color, value) in enumerate(zip(labels, colors, rel)):
            y = 1.35 - i * 0.68
            ax_out.barh(y, value, height=0.34, color=color, alpha=0.9)
            label_x = 0.01 if value > 0.22 else -0.18
            ax_out.text(label_x, y, lab, va="center", ha="left", fontsize=6.5, color="white" if value > 0.22 else color, fontweight="bold")
            value_x = min(float(value) + 0.03, 0.88) if value > 0.05 else 0.03
            ax_out.text(value_x, y, f"{value:.3f}", va="center", fontsize=6.5, color=INK)
        ax_out.set_xticks([0, 0.5, 1.0])
        if row_idx == len(evidence) - 1:
            ax_out.set_xticklabels(["0", ".5", "1"], fontsize=5.8)
            ax_out.set_xlabel("Gate value", fontsize=6.2, labelpad=1)
        else:
            ax_out.set_xticklabels([])
        ax_out.set_yticks([])
        ax_out.spines[["top", "right", "left"]].set_visible(False)
        ax_out.spines["bottom"].set_color(GRID)

        if row_idx == 0:
            for ax, header in zip([ax_rgb, ax_th, ax_audio, ax_out], headers):
                ax.set_title(header, fontsize=8, fontweight="bold", pad=5)

        # Left-side class marker.
        ax_rgb.text(-0.10, 0.5, f"{chr(97 + row_idx)})", transform=ax_rgb.transAxes, fontsize=8, fontweight="bold", ha="right", va="center")

    save(fig, OUT / "fig_02_qualitative.png")


def figure_4_5_comparison(evidence):
    """Figure 4.5: architecture delta plus measured clean-test comparison."""
    thumb = evidence[2]["rgb"]
    fixed = json.loads((ROOT / "outputs" / "logs" / "intermediate_fixed_test_results.json").read_text())["test"]
    adaptive = json.loads((ROOT / "outputs" / "logs" / "adaptfuse_v1_test_results.json").read_text())["test"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 4.45), gridspec_kw={"wspace": 0.10})
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Design D.
    ax_l.text(0.02, 0.94, "Design D — intermediate fixed fusion", fontsize=9.2, fontweight="bold")
    ax_l.text(0.02, 0.89, "Same three encoders; equal treatment after projection", fontsize=6.8, color=MUTED)
    ax_l.imshow(thumb, extent=(0.03, 0.20, 0.49, 0.76), aspect="auto", zorder=3)
    ax_l.add_patch(patches.Rectangle((0.03, 0.49), 0.17, 0.27, fill=False, ec=GRID, lw=1, zorder=4))
    rounded_box(ax_l, (0.27, 0.52), 0.18, 0.20, "3 encoders", "RGB / Pseudo-T / Audio", edge=BLUE, face=BLUE_BG)
    rounded_box(ax_l, (0.53, 0.52), 0.18, 0.20, "Concatenate", "576d fixed vector", edge=PURPLE, face=PURPLE_BG)
    rounded_box(ax_l, (0.79, 0.52), 0.17, 0.20, "Shared MLP", "disaster + victim", edge=RED, face=RED_BG)
    arrow(ax_l, (0.205, 0.625), (0.267, 0.625), color=BLUE)
    arrow(ax_l, (0.452, 0.625), (0.527, 0.625), color=PURPLE)
    arrow(ax_l, (0.712, 0.625), (0.787, 0.625), color=RED)
    ax_l.text(0.03, 0.37, "Held-out clean test", fontsize=6.6, color=MUTED, fontweight="bold")
    ax_l.text(0.03, 0.29, f"Combined macro-F1   {fixed['combined_f1']*100:.2f}%", fontsize=8.1, color=INK)
    ax_l.text(0.03, 0.21, f"Focal test loss          {fixed['loss']:.4f}", fontsize=8.1, color=INK)
    ax_l.text(0.03, 0.08, "No reliability estimate; no missing-sensor-aware weights.", fontsize=6.4, color=MUTED)

    # Design E.
    ax_r.text(0.02, 0.94, "Design E — AdapFuse-v1", fontsize=9.2, fontweight="bold", color=RED)
    ax_r.text(0.02, 0.89, "Adds reliability weighting and token interaction", fontsize=6.8, color=MUTED)
    ax_r.imshow(thumb, extent=(0.03, 0.20, 0.49, 0.76), aspect="auto", zorder=3)
    ax_r.add_patch(patches.Rectangle((0.03, 0.49), 0.17, 0.27, fill=False, ec=GRID, lw=1, zorder=4))
    rounded_box(ax_r, (0.24, 0.52), 0.13, 0.20, "3 encoders", "192d tokens", edge=BLUE, face=BLUE_BG, title_size=7.3)
    rounded_box(ax_r, (0.41, 0.52), 0.13, 0.20, "RUE", "3 weights", edge=GOLD, face=GOLD_BG, title_size=7.6)
    rounded_box(ax_r, (0.58, 0.52), 0.15, 0.20, "4-head\nattention", "mean pool", edge=TEAL, face=TEAL_BG, title_size=7.1)
    rounded_box(ax_r, (0.77, 0.52), 0.19, 0.20, "Single-step GRU\n+ heads", "nuisance branch*", edge=RED, face=RED_BG, title_size=7.1)
    for a, b, c in [(0.205, 0.238, BLUE), (0.372, 0.408, GOLD), (0.542, 0.578, TEAL), (0.732, 0.768, RED)]:
        arrow(ax_r, (a, 0.625), (b, 0.625), color=c)
    ax_r.text(0.03, 0.37, "Held-out clean test", fontsize=6.6, color=MUTED, fontweight="bold")
    delta_f1 = (adaptive["combined_f1"] - fixed["combined_f1"]) * 100
    loss_ratio = fixed["loss"] / adaptive["loss"]
    ax_r.text(0.03, 0.29, f"Combined macro-F1   {adaptive['combined_f1']*100:.2f}%   (+{delta_f1:.2f} pp)", fontsize=8.1, color=INK)
    ax_r.text(0.03, 0.21, f"Focal test loss          {adaptive['loss']:.4f}   ({loss_ratio:.1f}× lower)", fontsize=8.1, color=INK)
    ax_r.text(0.03, 0.08, "*Nuisance branch exists, but its loss is zero without nuisance_gt.", fontsize=6.4, color=MUTED)

    fig.add_artist(plt.Line2D([0.505, 0.505], [0.06, 0.94], transform=fig.transFigure, color=GRID, lw=1))
    save(fig, OUT / "fig_09_design_comparison.png")


def figure_4_6_provenance(evidence):
    """New figure: data provenance and supervision boundary."""
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.94, "What the 72,997-row unified metadata actually contains", fontsize=9.3, fontweight="bold")
    ax.text(0.02, 0.89, "Counts are recomputed from train.csv, val.csv, and test.csv", fontsize=6.7, color=MUTED)

    # Representative real thumbnails.
    for i, ev in enumerate(evidence):
        x0 = 0.02 + i * 0.105
        ax.imshow(ev["rgb"], extent=(x0, x0 + 0.092, 0.58, 0.82), aspect="auto", zorder=3)
        ax.add_patch(patches.Rectangle((x0, 0.58), 0.092, 0.24, fill=False, ec=GRID, lw=.9, zorder=4))
        ax.text(x0 + 0.046, 0.555, DISASTER[i].split(" /")[0], ha="center", fontsize=5.8, color=MUTED)

    cards = [
        (0.47, "RGB", "70,997 measured images\n70,997 unique paths", BLUE, BLUE_BG),
        (0.64, "Pseudo-thermal", "70,997 derived maps\n0 sensor-IR paths", RED, RED_BG),
        (0.81, "Audio", "39,003 assigned rows\n2,000 unique ESC-50 clips", TEAL, TEAL_BG),
    ]
    for x, title, detail, edge, face in cards:
        rounded_box(ax, (x, 0.58), 0.15, 0.24, title, detail, edge=edge, face=face, title_size=8.1, detail_size=6.5)

    ax.text(0.02, 0.44, "Supervision and evaluation boundary", fontsize=8.6, fontweight="bold")
    rounded_box(ax, (0.02, 0.13), 0.20, 0.23, "Scene labels", "4-class disaster\n72,997 rows", edge=BLUE, face=BLUE_BG)
    rounded_box(ax, (0.26, 0.13), 0.20, 0.23, "Victim labels", "binary presence\ndataset-dependent", edge=RED, face=RED_BG)
    rounded_box(ax, (0.50, 0.13), 0.20, 0.23, "Bounding boxes", "15,970 rows\nC2A + SARD", edge=GOLD, face=GOLD_BG)
    rounded_box(ax, (0.74, 0.13), 0.24, 0.23, "Not supervised", "nuisance labels; corruption flags;\nsynchronized tri-modal sequences", edge=MUTED, face=BG, dashed=True, title_size=7.8, detail_size=6.2)
    for x in [0.22, 0.46, 0.70]:
        arrow(ax, (x, 0.245), (x + 0.038, 0.245), color=MUTED, dashed=True, lw=.8)

    ax.text(0.02, 0.045, "Implication: results demonstrate heterogeneous-data fusion and missing-modality handling, not synchronized RGB–thermal–audio fusion in operational flight.", fontsize=6.8, color=INK, fontweight="bold")
    save(fig, OUT / "fig_14_data_provenance.png")


def figure_5_3_robustness():
    """Replace the old panel that mixed logged data with a hypothetical baseline."""
    robustness = json.loads((ROOT / "outputs" / "logs" / "adaptfuse_v1_robustness.json").read_text())["corruption"]
    behavior = json.loads((ROOT / "outputs" / "logs" / "adaptfuse_v1_reliability_behavior.json").read_text())
    severity = np.arange(0, 6)
    clean = 0.9912683395845754
    curve_spec = [
        ("smoke_overlay", "Smoke overlay", RED, "o"),
        ("motion_blur", "Motion blur", BLUE, "s"),
        ("low_light", "Low light", PURPLE, "^"),
        ("thermal_drift", "Thermal drift", GOLD, "D"),
        ("rotor_noise_snr", "Audio noise", TEAL, "v"),
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.25), gridspec_kw={"wspace": 0.30})
    for key, label, color, marker in curve_spec:
        vals = [clean] + [float(robustness[key][str(i)]) for i in range(1, 6)]
        ax0.plot(severity, vals, color=color, marker=marker, lw=1.8, ms=4.2, label=label)
    ax0.set_title("(a) Macro-F1", loc="left", fontweight="bold")
    ax0.set_xlabel("Severity")
    ax0.set_ylabel("Disaster macro-F1")
    ax0.set_xticks(severity)
    ax0.set_ylim(0.88, 1.00)
    ax0.grid(True, color=GRID, lw=.6, alpha=.8)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.legend(frameon=False, fontsize=6.6, loc="lower left")
    ax0.annotate("0.897", xy=(5, robustness["smoke_overlay"]["5"]), xytext=(3.0, .913), fontsize=6.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=.8))

    # Only severities saved by the reliability audit: 0, 1, 3, 5.
    mapping = [
        ("thermal", "r_thermal", "Pseudo-T (drift)", RED, "s"),
        ("rgb", "r_rgb", "RGB (smoke)", BLUE, "o"),
        ("audio", "r_audio", "Audio (noise)", TEAL, "^"),
    ]
    for group, field, label, color, marker in mapping:
        xs = [int(x["severity"]) for x in behavior[group]]
        ys = [float(x[field]) for x in behavior[group]]
        # RGB and pseudo-T gates coincide; draw RGB dashed on top so both stay visible
        style = dict(ls="--", mfc="white") if group == "rgb" else {}
        ax1.plot(xs, ys, color=color, marker=marker, lw=1.8, ms=4.2, label=label, **style)
    ax1.set_title("(b) Gate value", loc="left", fontweight="bold")
    ax1.set_xlabel("Severity")
    ax1.set_ylabel("Mean gate")
    ax1.set_xticks([0, 1, 3, 5])
    ax1.set_ylim(-0.03, 1.03)
    ax1.grid(True, color=GRID, lw=.6, alpha=.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=6.6, loc="center right")
    save(fig, OUT / "fig_05_robustness.png")


def main():
    evidence = load_evidence()
    for ev in evidence:
        r = ev["row"]
        print(
            f"{r.sample_id}: gt={DISASTER[int(r.disaster_label)]}, "
            f"pred={DISASTER[ev['pred']]}, conf={ev['confidence']:.4f}, "
            f"rel={np.round(ev['reliability'], 4).tolist()}"
        )
    figure_1_overview(evidence)
    figure_4_1_architecture(evidence)
    figure_4_2_qualitative(evidence)
    figure_4_5_comparison(evidence)
    figure_4_6_provenance(evidence)
    figure_5_3_robustness()


if __name__ == "__main__":
    main()
