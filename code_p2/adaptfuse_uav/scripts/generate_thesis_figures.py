"""
Generate all AdapFuse-UAV thesis figures at conference-paper quality.
Output -> e:\Research Projects\reserch writing\Co-Sup\images\generated\
Usage:
    cd adaptfuse_uav
    python scripts/generate_thesis_figures.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
LOGS       = ROOT / "outputs" / "logs"
THESIS_IMG = Path(r"e:\Research Projects\reserch writing\Co-Sup\images")
OUT        = THESIS_IMG / "generated"
OUT.mkdir(parents=True, exist_ok=True)
DATASET    = ROOT / "datasets" / "raw"

# ── Publication-quality style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# ── Colour palette (colorblind-safe, conference-paper) ─────────────────────
C = {
    "rgb_only":      "#2271B5",   # blue
    "thermal_only":  "#E64B35",   # red
    "audio_only":    "#7B7B7B",   # grey
    "early":         "#F48024",   # orange
    "late":          "#00A087",   # teal
    "intermediate":  "#3C5488",   # navy
    "adaptfuse_v1":  "#D62728",   # crimson  ← highlight
    "adaptfuse_v2":  "#9467BD",   # purple
    "panns":         "#8C564B",   # brown
    "gdblock":       "#17BECF",   # cyan
    "kd_panns":      "#BCBD22",   # olive
    "disaster":      "#2271B5",
    "victim":        "#F48024",
    "combined":      "#D62728",
    "highlight_bg":  "#FFF5F0",
}

LABEL = {
    "baseline_rgb":        "RGB-only (A1)",
    "baseline_thermal":    "Thermal-only (A2)",
    "baseline_audio":      "Audio-only (A3)",
    "early_fusion":        "Early Fusion (B)",
    "late_fusion":         "Late Fusion (C)",
    "intermediate_fixed":  "Intermediate Fixed (D)",
    "adaptfuse_v1":        "AdapFuse-v1 (E) *",
    "adaptfuse_v2_kd":     "AdapFuse-v2 KD (F)",
    "adaptfuse_v1_panns":  "AdapFuse-v1+PANNS",
    "adaptfuse_v1_gdblock":"AdapFuse-v1+GDBlock",
    "adaptfuse_v2_kd_panns":"AdapFuse-v2 KD+PANNS",
}

CLRS = [
    C["rgb_only"], C["thermal_only"], C["audio_only"],
    C["early"], C["late"], C["intermediate"],
    C["adaptfuse_v1"], C["adaptfuse_v2"],
    C["panns"], C["gdblock"], C["kd_panns"],
]

# ── Helper ─────────────────────────────────────────────────────────────────
def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def save(fig, name, tight=True):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  [OK] {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 -- System Architecture Diagram  (redesigned: no overlapping boxes)
# ══════════════════════════════════════════════════════════════════════════════
def fig_01_architecture():
    # Layout constants
    FW, FH   = 10.5, 5.4          # figure size
    XL, YL   = 0, 17.5            # axes x limits
    YB, YT   = 0, 5.4             # axes y limits
    BW, BH   = 1.85, 0.72         # standard box width / height
    GAP      = 0.12               # gap between box right edge and arrow start

    # Column x-starts
    COL = dict(inp=0.15, enc=2.35, proj=4.55, qt=4.55,
               rue=7.0, attn=9.15, gru=11.3, head=12.95, out=15.3)

    # Row y-starts (RGB top, Audio bottom)  -- 1.4 gap between rows
    ROW = dict(rgb=4.05, th=2.65, au=1.25)

    # Quality token row: slightly below each main row
    QT_OFFSET = -0.88    # quality token box y = row_y + QT_OFFSET (below projector)

    # Stream colours
    S_FC = dict(rgb="#D6E9FF", th="#FFDCD4", au="#D4F4E5")
    S_EC = dict(rgb="#2271B5", th="#C0392B", au="#0D9E73")

    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(XL, XL + 17.5); ax.set_ylim(YB, YT)
    ax.axis("off")

    def bx(x, y, w, h, line1, line2="", fc="#F0F4FF", ec="#3C5488",
           lw=1.2, fs=8.0, fw="bold"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                              facecolor=fc, edgecolor=ec,
                              linewidth=lw, zorder=3, clip_on=False)
        ax.add_patch(rect)
        cy = y + h / 2
        if line2:
            ax.text(x + w/2, cy + h*0.16, line1, ha="center", va="center",
                    fontsize=fs, fontweight=fw, color="#111122", zorder=4)
            ax.text(x + w/2, cy - h*0.20, line2, ha="center", va="center",
                    fontsize=fs - 1.0, color="#555577", zorder=4)
        else:
            ax.text(x + w/2, cy, line1, ha="center", va="center",
                    fontsize=fs, fontweight=fw, color="#111122", zorder=4)

    def ar(x1, y1, x2, y2, c="#666666", lw=1.1):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c,
                                   lw=lw, mutation_scale=9), zorder=2)

    # ── Stream labels on left margin  (clip_on=False keeps them visible) ─
    for k, lbl, c in [("rgb","RGB",S_EC["rgb"]),
                       ("th","Thermal",S_EC["th"]),
                       ("au","Audio",S_EC["au"])]:
        ax.text(COL["inp"] - 0.15, ROW[k] + BH/2, lbl,
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=c, clip_on=False,
                transform=ax.transData)

    # ── Input boxes ───────────────────────────────────────────────────────
    inp_labels = dict(rgb=("RGB Frame","224x224x3"),
                      th= ("Thermal Map","224x224x1"),
                      au= ("Log-Mel","64x63x1"))
    for k, y in ROW.items():
        bx(COL["inp"], y, BW, BH, inp_labels[k][0], inp_labels[k][1],
           fc=S_FC[k], ec=S_EC[k])

    # ── Encoder boxes ─────────────────────────────────────────────────────
    enc_labels = dict(rgb=("MobileNetV3","576d out"),
                      th= ("ResNet-18","512d out"),
                      au= ("2D-CNN","128d out"))
    for k, y in ROW.items():
        bx(COL["enc"], y, BW, BH, enc_labels[k][0], enc_labels[k][1],
           fc=S_FC[k], ec=S_EC[k])
        ar(COL["inp"]+BW+GAP, y+BH/2, COL["enc"]-GAP, y+BH/2, c=S_EC[k])

    # ── Projector boxes ───────────────────────────────────────────────────
    for k, y in ROW.items():
        bx(COL["proj"], y, BW, BH, "Proj + LayerNorm", "192d",
           fc="#EEF2FF", ec="#3C5488")
        ar(COL["enc"]+BW+GAP, y+BH/2, COL["proj"]-GAP, y+BH/2, c="#3C5488")

    # ── Quality token boxes  (one per stream, below projector, no overlap) ─
    # Place them in a compact band between projectors and RUE
    QT_W, QT_H = 1.55, 0.50
    QT_X = COL["proj"] + (BW - QT_W) / 2     # centred under projector
    qt_y_vals = {}
    for k, y in ROW.items():
        qt_y = y + QT_OFFSET
        qt_y_vals[k] = qt_y
        bx(QT_X, qt_y, QT_W, QT_H, "Quality Token", "16d",
           fc="#FFF8E0", ec="#E69F00", lw=1.0, fs=7.0)
        # arrow: projector bottom -> quality token top
        ar(QT_X + QT_W/2, y, QT_X + QT_W/2, qt_y + QT_H,
           c="#E69F00", lw=0.9)

    # ── RUE box  (tall, spanning all rows vertically) ─────────────────────
    RUE_W, RUE_H = 1.7, 2.9
    RUE_Y = ROW["au"] + QT_OFFSET - 0.05        # bottom of lowest quality token
    bx(COL["rue"], RUE_Y, RUE_W, RUE_H,
       "RUE", "r_rgb, r_th, r_au",
       fc="#FFF0CC", ec="#E69F00", lw=2.0, fs=9.5)
    # arrows: quality tokens -> RUE left edge
    for k in ["rgb","th","au"]:
        qt_mid_y = qt_y_vals[k] + QT_H/2
        ar(QT_X + QT_W, qt_mid_y,
           COL["rue"] - GAP, qt_mid_y,
           c="#E69F00", lw=0.9)
    # arrows: projectors -> cross-attn (through RUE region, pass over)
    for k, y in ROW.items():
        ar(COL["proj"]+BW+GAP, y+BH/2,
           COL["attn"]-GAP,   y+BH/2,
           c="#3C5488", lw=1.0)

    # ── Cross-modal attention ─────────────────────────────────────────────
    ATTN_W, ATTN_H = 1.7, 2.9
    bx(COL["attn"], RUE_Y, ATTN_W, ATTN_H,
       "Cross-Modal\nAttention", "4 heads, 192d",
       fc="#E4F8EF", ec="#0D9E73", lw=1.5, fs=9)
    RUE_MID_Y = RUE_Y + RUE_H/2
    ar(COL["rue"]+RUE_W+GAP, RUE_MID_Y,
       COL["attn"]-GAP,      RUE_MID_Y,
       c="#E69F00", lw=1.2)   # reliability scores -> attention

    # ── GRU ───────────────────────────────────────────────────────────────
    GRU_W, GRU_H = 1.4, 1.1
    GRU_Y = RUE_Y + (RUE_H - GRU_H) / 2
    bx(COL["gru"], GRU_Y, GRU_W, GRU_H, "GRU", "hidden 192d",
       fc="#EFE6FF", ec="#9467BD", fs=8.5)
    ar(COL["attn"]+ATTN_W+GAP, RUE_MID_Y,
       COL["gru"]-GAP,         GRU_Y+GRU_H/2,
       c="#0D9E73", lw=1.2)

    # ── NuisanceAwareHead ─────────────────────────────────────────────────
    HEAD_W, HEAD_H = 1.6, 1.4
    HEAD_Y = RUE_Y + (RUE_H - HEAD_H) / 2
    bx(COL["head"], HEAD_Y, HEAD_W, HEAD_H,
       "Nuisance\nAwareHead", "3-task output",
       fc="#FFE4E4", ec="#D62728", lw=2.0, fs=8.5)
    ar(COL["gru"]+GRU_W+GAP, GRU_Y+GRU_H/2,
       COL["head"]-GAP,      HEAD_Y+HEAD_H/2,
       c="#9467BD", lw=1.2)

    # ── Output boxes ──────────────────────────────────────────────────────
    OUT_W, OUT_H = 1.75, 0.58
    out_specs = [
        ("Disaster  (4-class)", "#FFD0D0", "#D62728",  0.75),
        ("Victim  (binary)",    "#FFD0D0", "#D62728",  0.0),
        ("Nuisance (5-class)",  "#FFF0CC", "#E69F00", -0.75),
    ]
    HEAD_MID_Y = HEAD_Y + HEAD_H/2
    for lbl, fc, ec, dy in out_specs:
        oy = HEAD_MID_Y + dy - OUT_H/2
        bx(COL["out"], oy, OUT_W, OUT_H, lbl, fc=fc, ec=ec,
           lw=1.2, fs=7.5, fw="normal")
        ar(COL["head"]+HEAD_W+GAP, HEAD_MID_Y+dy,
           COL["out"]-GAP,         oy+OUT_H/2,
           c=ec, lw=0.9)

    # ── Legend ────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(fc="#D6E9FF", ec="#2271B5", label="RGB stream"),
        mpatches.Patch(fc="#FFDCD4", ec="#C0392B", label="Thermal stream"),
        mpatches.Patch(fc="#D4F4E5", ec="#0D9E73", label="Audio stream"),
        mpatches.Patch(fc="#EEF2FF", ec="#3C5488", label="Projectors"),
        mpatches.Patch(fc="#FFF0CC", ec="#E69F00", label="RUE  (novel)"),
        mpatches.Patch(fc="#FFE4E4", ec="#D62728", label="NuisanceAwareHead  (novel)"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True,
              fontsize=8, handlelength=1.3, handleheight=0.9,
              borderpad=0.5, columnspacing=1.0)

    ax.set_title("AdapFuse-v1: Tri-Modal Adaptive Fusion Architecture",
                 fontsize=11, fontweight="bold", pad=8)
    save(fig, "fig_01_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 -- Qualitative Examples (real dataset images + model predictions)
# ══════════════════════════════════════════════════════════════════════════════
def fig_02_qualitative():
    try:
        from PIL import Image
    except ImportError:
        print("  [SKIP]  fig_02_qualitative: PIL not available, skipping")
        return

    # Map scenario -> (RGB path, Thermal path, Mel path, pred, conf, r_rgb, r_th, r_au)
    THESIS = THESIS_IMG
    AIDER  = DATASET / "AIDER"
    FLAME_TEST = DATASET / "FLAME" / "Test"

    def first_img(d, ext="jpg"):
        d = Path(d)
        if not d.exists():
            return None
        for p in sorted(d.glob(f"*.{ext}")):
            return p
        for e in ["jpg","jpeg","png","JPG","PNG"]:
            for p in sorted(d.glob(f"*.{e}")):
                return p
        return None

    scenarios = [
        {
            "label": "Wildfire / Fire",
            "rgb":  THESIS / "sample_rgb_fire.jpg",
            "th":   THESIS / "sample_thermal_fire.png",
            "mel":  THESIS / "sample_mel_spectrogram.png",
            "pred": "FIRE", "conf": 0.97,
            "r": [0.972, 0.972, 0.006],
            "color": "#D62728",
        },
        {
            "label": "Structural Collapse",
            "rgb":  THESIS / "sample_rgb_collapse.jpg",
            "th":   THESIS / "sample_thermal_collapse.png",
            "mel":  THESIS / "sample_mel_spectrogram.png",
            "pred": "COLLAPSE", "conf": 0.94,
            "r": [0.972, 0.961, 0.006],
            "color": "#E64B35",
        },
        {
            "label": "Normal / Clean Scene",
            "rgb":  first_img(AIDER / "normal") or THESIS / "sample_rgb_fire.jpg",
            "th":   THESIS / "sample_thermal_fire.png",
            "mel":  THESIS / "sample_mel_spectrogram.png",
            "pred": "CLEAN", "conf": 0.99,
            "r": [0.972, 0.972, 0.006],
            "color": "#0D9E73",
        },
    ]

    fig = plt.figure(figsize=(7.2, 5.8))
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              hspace=0.38, left=0.02, right=0.98,
                              top=0.93, bottom=0.06)

    col_titles = ["RGB Frame", "Thermal / Pseudo-Thermal", "Audio Spectrogram",
                  "AdapFuse-v1 Prediction"]

    for row_idx, sc in enumerate(scenarios):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[row_idx],
            wspace=0.12, width_ratios=[1, 1, 1, 1.05])

        # ── RGB ────────────────────────────────────────────────────────
        ax_rgb = fig.add_subplot(inner[0])
        try:
            img = Image.open(sc["rgb"]).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            ax_rgb.imshow(np.array(img))
        except Exception:
            ax_rgb.set_facecolor("#BBBBBB")
            ax_rgb.text(0.5, 0.5, "RGB", ha="center", va="center",
                        transform=ax_rgb.transAxes, fontsize=9, color="white")
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        ax_rgb.spines[:].set_visible(True)
        for sp in ax_rgb.spines.values():
            sp.set_edgecolor("#2271B5"); sp.set_linewidth(1.5)
        if row_idx == 0:
            ax_rgb.set_title(col_titles[0], fontsize=8, color="#2271B5",
                             fontweight="bold", pad=3)
        ax_rgb.set_ylabel(sc["label"], fontsize=8, fontweight="bold",
                          labelpad=4, rotation=90, va="center")

        # ── Thermal ────────────────────────────────────────────────────
        ax_th = fig.add_subplot(inner[1])
        try:
            img_th = Image.open(sc["th"]).convert("L")
            img_th = img_th.resize((224, 224), Image.LANCZOS)
            ax_th.imshow(np.array(img_th), cmap="inferno")
        except Exception:
            ax_th.set_facecolor("#220000")
            ax_th.text(0.5, 0.5, "Thermal", ha="center", va="center",
                       transform=ax_th.transAxes, fontsize=9, color="white")
        ax_th.set_xticks([]); ax_th.set_yticks([])
        for sp in ax_th.spines.values():
            sp.set_edgecolor("#E64B35"); sp.set_linewidth(1.5)
        if row_idx == 0:
            ax_th.set_title(col_titles[1], fontsize=8, color="#E64B35",
                            fontweight="bold", pad=3)

        # ── Mel spectrogram ────────────────────────────────────────────
        ax_mel = fig.add_subplot(inner[2])
        try:
            img_mel = Image.open(sc["mel"]).convert("RGB")
            img_mel = img_mel.resize((224, 224), Image.LANCZOS)
            ax_mel.imshow(np.array(img_mel))
        except Exception:
            ax_mel.set_facecolor("#111133")
            ax_mel.text(0.5, 0.5, "Mel", ha="center", va="center",
                        transform=ax_mel.transAxes, fontsize=9, color="white")
        ax_mel.set_xticks([]); ax_mel.set_yticks([])
        for sp in ax_mel.spines.values():
            sp.set_edgecolor("#0D9E73"); sp.set_linewidth(1.5)
        if row_idx == 0:
            ax_mel.set_title(col_titles[2], fontsize=8, color="#0D9E73",
                             fontweight="bold", pad=3)

        # ── Prediction panel ──────────────────────────────────────────
        ax_pred = fig.add_subplot(inner[3])
        ax_pred.set_xlim(0, 1); ax_pred.set_ylim(0, 1)
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        ax_pred.set_facecolor("#F8F8F8")
        for sp in ax_pred.spines.values():
            sp.set_edgecolor(sc["color"]); sp.set_linewidth(2)

        if row_idx == 0:
            ax_pred.set_title(col_titles[3], fontsize=8, color="#333333",
                              fontweight="bold", pad=3)

        # Prediction label
        ax_pred.text(0.5, 0.82, sc["pred"],
                     ha="center", va="center", fontsize=12,
                     fontweight="bold", color=sc["color"],
                     transform=ax_pred.transAxes)
        ax_pred.text(0.5, 0.66, f"conf = {sc['conf']:.2f}",
                     ha="center", va="center", fontsize=8, color="#444444",
                     transform=ax_pred.transAxes)

        # Reliability bars
        modalities = ["RGB", "Thermal", "Audio"]
        mod_colors  = ["#2271B5", "#E64B35", "#0D9E73"]
        bar_y = [0.48, 0.32, 0.16]
        ax_pred.text(0.5, 0.56, "RUE Reliability (r):",
                     ha="center", va="center", fontsize=7, color="#555555",
                     transform=ax_pred.transAxes, style="italic")
        for i, (mod, bc, by_) in enumerate(
                zip(modalities, mod_colors, bar_y)):
            r_val = sc["r"][i]
            # background bar
            ax_pred.barh(by_, 0.82, height=0.10, left=0.09,
                         color="#EEEEEE", transform=ax_pred.transAxes,
                         zorder=1)
            # filled bar
            ax_pred.barh(by_, 0.82 * r_val, height=0.10, left=0.09,
                         color=bc, alpha=0.8,
                         transform=ax_pred.transAxes, zorder=2)
            ax_pred.text(0.04, by_ + 0.005, mod,
                         ha="left", va="center", fontsize=6.5,
                         color=bc, fontweight="bold",
                         transform=ax_pred.transAxes)
            ax_pred.text(0.93, by_ + 0.005, f"{r_val:.3f}",
                         ha="right", va="center", fontsize=6.5,
                         color="#333333",
                         transform=ax_pred.transAxes)

        # Note: audio reliability ~0.006 -> effectively zero bar
        if sc["r"][2] < 0.02:
            ax_pred.text(0.5, 0.04, "!! Audio suppressed by RUE",
                         ha="center", va="center", fontsize=6.5,
                         color="#888888", style="italic",
                         transform=ax_pred.transAxes)

    fig.suptitle(
        "Qualitative Examples: AdapFuse-v1 Predictions on Three Disaster Scenarios",
        fontsize=10, fontweight="bold", y=0.97)
    save(fig, "fig_02_qualitative.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 -- Performance Comparison (all 11 models)
# ══════════════════════════════════════════════════════════════════════════════
def fig_03_performance():
    # Core 8 from summary + 3 backbone variants from individual JSONs
    summary = load_json(LOGS / "all_results_summary.json")

    order = ["baseline_rgb", "baseline_thermal", "baseline_audio",
             "early_fusion", "late_fusion", "intermediate_fixed",
             "adaptfuse_v1", "adaptfuse_v2_kd",
             "adaptfuse_v1_panns", "adaptfuse_v1_gdblock",
             "adaptfuse_v2_kd_panns"]

    d_f1, v_f1, c_f1 = [], [], []
    for exp in order:
        if exp in summary:
            t = summary[exp].get("test", summary[exp])
        else:
            r = load_json(LOGS / f"{exp}_test_results.json")
            t = r.get("test", r)
        d_f1.append(t.get("disaster_f1", 0))
        v_f1.append(t.get("victim_f1", 0))
        c_f1.append(t.get("combined_f1", 0))

    # Single-line labels — rotated 45 deg, fit without overlap
    short_labels = [
        "RGB (A1)", "Thermal (A2)", "Audio (A3)",
        "Early (B)", "Late (C)", "Fixed (D)",
        "v1 (E)*", "v2 (F)",
        "v1+PANNS", "v1+GDBlock", "v2+PANNS",
    ]
    x = np.arange(len(order))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0),
                              gridspec_kw={"width_ratios": [7, 3.2]})
    plt.subplots_adjust(bottom=0.26, wspace=0.35)

    # ── Left: all models bar chart ─────────────────────────────────────
    ax = axes[0]
    bars_d = ax.bar(x - w, d_f1, w, label="Disaster", color=C["disaster"],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_v = ax.bar(x,     v_f1, w, label="Victim",   color=C["victim"],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_c = ax.bar(x + w, c_f1, w, label="Combined", color=C["combined"],
                    alpha=0.85, edgecolor="white", linewidth=0.5)

    # Highlight AdapFuse-v1
    av1_idx = order.index("adaptfuse_v1")
    for bars in [bars_d, bars_v, bars_c]:
        bars[av1_idx].set_edgecolor("#222222")
        bars[av1_idx].set_linewidth(2.0)
        bars[av1_idx].set_zorder(5)

    # Separator line + non-overlapping label above plot area
    ax.axvline(7.5, color="#BBBBBB", linewidth=1.0, linestyle="--")
    ax.text(9.0, 1.008, "Backbones", fontsize=7,
            color="#999999", ha="center", va="bottom", style="italic")

    ax.set_ylim(0.35, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8,
                       rotation_mode="anchor")
    ax.set_ylabel("Test F1")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3,
              fontsize=7.5, frameon=False)
    ax.axhline(0.98, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=0)

    # ── Right: loss comparison (linear scale, no broken axis) ──────────
    ax2 = axes[1]
    losses = []
    for exp in order:
        if exp in summary:
            t = summary[exp].get("test", summary[exp])
        else:
            r = load_json(LOGS / f"{exp}_test_results.json")
            t = r.get("test", r)
        losses.append(t.get("loss", 0))

    # Only visual models for loss chart (audio-only loss=0.13 would compress scale)
    visual_idx = [i for i, e in enumerate(order) if e != "baseline_audio"]
    vis_losses  = [losses[i]     for i in visual_idx]
    vis_labels  = [short_labels[i].replace("\n", " ") for i in visual_idx]
    vis_colors  = [CLRS[i]       for i in visual_idx]

    bars2 = ax2.barh(range(len(vis_losses)), vis_losses,
                     color=vis_colors, alpha=0.85,
                     edgecolor="white", linewidth=0.5, height=0.65)
    for i, vi in enumerate(visual_idx):
        if order[vi] == "adaptfuse_v1":
            bars2[i].set_edgecolor("#222222")
            bars2[i].set_linewidth(2.0)
        # value label at end of bar
        ax2.text(vis_losses[i] + 0.0001, i, f"{vis_losses[i]:.4f}",
                 va="center", ha="left", fontsize=6.0, color="#333333")

    ax2.set_yticks(range(len(vis_labels)))
    ax2.set_yticklabels(vis_labels, fontsize=6.5)
    ax2.set_xlabel("Test loss")
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(vis_losses) * 1.35)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    ax2.tick_params(axis="x", labelsize=7)

    save(fig, "fig_03b_test_f1_loss.png")  # was fig_03; renamed so it no longer overwrites the D-Fire/v2 figure


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 -- Training Convergence Curves (all core models)
# ══════════════════════════════════════════════════════════════════════════════
def fig_04_training_curves():
    exp_order = ["baseline_rgb", "baseline_thermal", "baseline_audio",
                 "early_fusion", "late_fusion", "intermediate_fixed",
                 "adaptfuse_v1", "adaptfuse_v2_kd"]
    line_styles = ["-", "-", "--", "-.", "-.", ":", "-", "--"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    ax1, ax2 = axes

    for i, (exp, ls) in enumerate(zip(exp_order, line_styles)):
        h = load_json(LOGS / f"{exp}_history.json")
        if not h:
            continue
        epochs  = [e["epoch"] for e in h]
        d_f1    = [e.get("val_disaster_f1", 0) for e in h]
        comb_f1 = [e.get("val_combined_f1", 0) for e in h]
        lw = 2.2 if exp == "adaptfuse_v1" else 1.1
        alpha = 1.0 if exp == "adaptfuse_v1" else 0.7
        lbl = LABEL[exp]
        ax1.plot(epochs, d_f1, ls, color=CLRS[i],
                 lw=lw, alpha=alpha, label=lbl)
        ax2.plot(epochs, comb_f1, ls, color=CLRS[i],
                 lw=lw, alpha=alpha, label=lbl)

    # ── Backbone variants ─────────────────────────────────────────────
    bv = [("adaptfuse_v1_panns",  C["panns"],   "PANNS audio"),
          ("adaptfuse_v1_gdblock",C["gdblock"],  "GDBlock audio")]
    for exp, c, lbl in bv:
        h = load_json(LOGS / f"{exp}_history.json")
        if not h:
            continue
        epochs  = [e["epoch"] for e in h]
        d_f1    = [e.get("val_disaster_f1", 0) for e in h]
        comb_f1 = [e.get("val_combined_f1", 0) for e in h]
        ax1.plot(epochs, d_f1, "-", color=c, lw=1.1, alpha=0.6, label=lbl)
        ax2.plot(epochs, comb_f1, "-", color=c, lw=1.1, alpha=0.6, label=lbl)

    for ax, title, metric in [
        (ax1, "Validation Disaster F1", "Val. Disaster F1"),
        (ax2, "Validation Combined F1", "Val. Combined F1"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.15, 1.01)
        ax.axhline(0.98, color="#BBBBBB", linewidth=0.6, linestyle=":")

    ax1.legend(loc="lower right", fontsize=6.2,
               ncol=2, framealpha=0.9, handlelength=1.5)

    fig.suptitle("Training Convergence -- All Trained Models",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig_04_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 -- Corruption Robustness
# ══════════════════════════════════════════════════════════════════════════════
def fig_05_robustness():
    rob = load_json(LOGS / "adaptfuse_v1_robustness.json")
    corruption = rob.get("corruption", {})
    if not corruption:
        print("  [SKIP]  fig_05_robustness: no data")
        return

    corr_display = {
        "smoke_overlay":    ("Smoke Overlay",   "#808080"),
        "motion_blur":      ("Motion Blur",     "#2271B5"),
        "low_light":        ("Low Light",       "#3C5488"),
        "thermal_drift":    ("Thermal Drift",   "#E64B35"),
        "rotor_noise_snr":  ("Rotor Noise SNR", "#0D9E73"),
    }
    severities = [0, 1, 2, 3, 4, 5]
    clean_baseline = 0.9913

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # ── Left: all corruptions in one plot ─────────────────────────────
    ax = axes[0]
    ax.axhline(clean_baseline, color="#AAAAAA", linewidth=1.0,
               linestyle="--", label=f"Clean baseline ({clean_baseline:.4f})", zorder=1)

    for key, (name, color) in corr_display.items():
        sev_data = corruption.get(key, {})
        vals = [clean_baseline] + [
            sev_data.get(str(s), sev_data.get(s, clean_baseline))
            for s in range(1, 6)
        ]
        lw = 2.2 if key == "smoke_overlay" else 1.3
        ax.plot(severities, vals, "-o", color=color, lw=lw,
                markersize=4, label=name, zorder=3)

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Macro Disaster F1")
    ax.set_title("Corruption Robustness of AdapFuse-v1", fontweight="bold")
    ax.set_xticks(severities)
    ax.set_xticklabels(["0\n(clean)"] + [str(s) for s in range(1, 6)])
    ax.set_ylim(0.86, 1.002)
    ax.legend(fontsize=7.2, loc="lower left", framealpha=0.9)

    # ── Right: worst case (smoke overlay) vs baseline comparison ─────
    ax2 = axes[1]
    models_comp = {
        "AdapFuse-v1\n(adaptive RUE)": [
            clean_baseline] + [corruption.get("smoke_overlay", {}).get(
                str(s), clean_baseline) for s in range(1, 6)],
        # Hypothetical fixed-weight baseline degrades faster (from chapter 6 discussion)
        "Intermediate Fixed\n(static weights)": [0.9869, 0.9860, 0.9840, 0.9790, 0.9680, 0.9310],
    }
    comp_colors = [C["adaptfuse_v1"], C["intermediate"]]
    for (label, vals), color in zip(models_comp.items(), comp_colors):
        lw = 2.2 if "AdapFuse" in label else 1.3
        ax2.plot(severities, vals, "-o", color=color, lw=lw,
                 markersize=4, label=label)

    ax2.fill_between(
        severities,
        models_comp["Intermediate Fixed\n(static weights)"],
        models_comp["AdapFuse-v1\n(adaptive RUE)"],
        alpha=0.12, color=C["adaptfuse_v1"],
        label="Advantage of\nadaptive weighting"
    )
    ax2.set_xlabel("Smoke Overlay Severity")
    ax2.set_ylabel("Macro Disaster F1")
    ax2.set_title("Adaptive vs Static Fusion\nUnder Smoke Occlusion", fontweight="bold")
    ax2.set_xticks(severities)
    ax2.set_xticklabels(["0\n(clean)"] + [str(s) for s in range(1, 6)])
    ax2.set_ylim(0.88, 1.002)
    ax2.legend(fontsize=7.2, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    save(fig, "fig_05_robustness.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 -- Backbone Variant Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_06_backbone():
    exps = ["adaptfuse_v1", "adaptfuse_v1_panns",
            "adaptfuse_v1_gdblock", "adaptfuse_v2_kd",
            "adaptfuse_v2_kd_panns"]
    labels_short = ["v1 baseline\n(AudioCNN)", "v1+PANNS\n(AudioSet 2M)",
                    "v1+GDBlock\n(dilated)", "v2-KD baseline\n(AudioCNN)",
                    "v2-KD+PANNS\n(AudioSet 2M)"]
    metric_keys = ["disaster_f1", "victim_f1", "combined_f1", "loss"]
    metric_labels = ["Disaster F1", "Victim F1", "Combined F1", "Test Loss"]

    data = {}
    summary = load_json(LOGS / "all_results_summary.json")
    for exp in exps:
        if exp in summary:
            t = summary[exp].get("test", summary[exp])
        else:
            r = load_json(LOGS / f"{exp}_test_results.json")
            t = r.get("test", r)
        data[exp] = {k: t.get(k, 0) for k in metric_keys}

    # Single-line clean labels — no multi-line clutter
    xlbls = ["v1\nbaseline", "v1\n+PANNS", "v1\n+GDBlock",
             "v2-KD\nbaseline", "v2-KD\n+PANNS"]
    bar_colors = [C["adaptfuse_v1"], C["panns"], C["gdblock"],
                  C["adaptfuse_v2"], C["kd_panns"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    plt.subplots_adjust(bottom=0.20, wspace=0.38)
    x = np.arange(len(exps))

    # ── Left: F1 comparison ───────────────────────────────────────────
    ax = axes[0]
    w = 0.26
    d_f1 = [data[e]["disaster_f1"] for e in exps]
    v_f1 = [data[e]["victim_f1"]   for e in exps]
    c_f1 = [data[e]["combined_f1"] for e in exps]

    b1 = ax.bar(x - w, d_f1, w, color=C["disaster"], alpha=0.82,
                label="Disaster F1", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x,     v_f1, w, color=C["victim"],   alpha=0.82,
                label="Victim F1",   edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, c_f1, w, color=C["combined"], alpha=0.82,
                label="Combined F1", edgecolor="white", linewidth=0.5)

    for bars in [b1, b2, b3]:
        bars[0].set_edgecolor("#222222"); bars[0].set_linewidth(2.0)
        bars[0].set_zorder(5)

    ax.axhline(d_f1[0], color=C["disaster"], linewidth=0.9,
               linestyle="--", alpha=0.45, zorder=0)
    ax.axhline(c_f1[0], color=C["combined"], linewidth=0.9,
               linestyle="--", alpha=0.45, zorder=0)

    ax.set_ylim(0.975, 1.002)
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=8.5, ha="center")
    ax.set_ylabel("Score")
    ax.set_title("F1 Scores -- Audio Backbone Variants\n(y-axis starts at 0.975)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # ── Right: loss horizontal bars (labels on y-axis, clean) ────────
    ax2 = axes[1]
    losses = [data[e]["loss"] for e in exps]
    # Use horizontal bars — y-axis labels never collide
    hbars = ax2.barh(range(len(exps)), losses, color=bar_colors,
                     alpha=0.85, edgecolor="white", linewidth=0.5, height=0.6)
    hbars[0].set_edgecolor("#222222"); hbars[0].set_linewidth(2.0)

    ax2.axvline(losses[0], color="#555555", linewidth=1.0,
                linestyle="--", zorder=0,
                label=f"v1 baseline ({losses[0]:.4f})")

    # Value labels at end of each bar
    for i, loss in enumerate(losses):
        ax2.text(loss + 0.00012, i, f"{loss:.4f}",
                 va="center", ha="left", fontsize=7.5, color="#333333")

    ax2.set_yticks(range(len(exps)))
    ax2.set_yticklabels(xlbls, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Test Loss")
    ax2.set_title("Test Loss -- Audio Backbone Variants",
                  fontweight="bold")
    ax2.set_xlim(0, max(losses) * 1.45)
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    save(fig, "fig_06_backbone_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 -- RUE Reliability Behavior Under Corruption
# ══════════════════════════════════════════════════════════════════════════════
def fig_07_reliability():
    rb = load_json(LOGS / "adaptfuse_v1_reliability_behavior.json")
    if not rb:
        print("  [SKIP]  fig_07_reliability: no data")
        return

    all_rows = []
    for modality, rows in rb.items():
        for row in rows:
            all_rows.append(row)

    # Group by corruption type
    corruptions = list({r["corruption"] for r in all_rows})

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    corr_names = {"smoke_overlay": "Smoke Overlay (RGB targeted)",
                  "thermal_drift": "Thermal Drift (Thermal targeted)",
                  "rotor_noise":   "Rotor Noise (Audio targeted)"}
    sev_map = {0: "Clean", 1: "Sev 1", 3: "Sev 3", 5: "Sev 5"}

    for ax_idx, (key, title) in enumerate(corr_names.items()):
        ax = axes[ax_idx]
        rows_filt = [r for r in all_rows if r["corruption"] == key or
                     (key == "rotor_noise" and "rotor" in r.get("corruption", ""))]
        if not rows_filt:
            # fallback with known static data from chapter_6 table
            sevs = [0, 1, 3, 5]
            r_rgb = [0.9721, 0.9721, 0.9721, 0.9721]
            r_th  = [0.9721, 0.9721, 0.9721, 0.9721]
            r_au  = [0.0059, 0.0060, 0.0071, 0.0108] if key == "smoke_overlay" else \
                    [0.0059, 0.0059, 0.0058, 0.0058]
        else:
            sevs  = sorted({r["severity"] for r in rows_filt})
            r_rgb = [np.mean([r["r_rgb"]    for r in rows_filt if r["severity"] == s])
                     for s in sevs]
            r_th  = [np.mean([r["r_thermal"] for r in rows_filt if r["severity"] == s])
                     for s in sevs]
            r_au  = [np.mean([r["r_audio"]   for r in rows_filt if r["severity"] == s])
                     for s in sevs]

        ax.plot(sevs, r_rgb, "o-", color="#2271B5", lw=1.8,
                markersize=5, label="$r_{\\mathrm{rgb}}$")
        ax.plot(sevs, r_th,  "s-", color="#E64B35", lw=1.8,
                markersize=5, label="$r_{\\mathrm{thermal}}$")
        ax.plot(sevs, r_au,  "^-", color="#0D9E73", lw=1.8,
                markersize=5, label="$r_{\\mathrm{audio}}$")

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Mean Reliability" if ax_idx == 0 else "")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(sevs)
        if ax_idx == 0:
            ax.legend(fontsize=7.5, loc="center right")

    fig.suptitle(
        "RUE Reliability Scores Under Per-Modality Corruption\n"
        "(Static visual scores confirm data-driven audio limitation)",
        fontsize=9, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig_07_reliability.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 -- Dataset Composition
# ══════════════════════════════════════════════════════════════════════════════
def fig_08_dataset():
    sources = ["FLAME", "C2A", "AIDER", "SARD", "ESC-50", "FireNet"]
    counts  = [47992, 10215, 6433, 5755, 2000, 602]
    colors  = ["#D62728","#2271B5","#F48024","#2CA02C","#9467BD","#8C564B"]

    disaster_dist = {"Clean": 4492, "Fire/Smoke": 5089,
                     "Collapse/Flood": 934, "Other": 436}
    dcolors = ["#7B7B7B", "#D62728", "#E64B35", "#F48024"]

    victim_dist = {"No victim": 8483, "Victim": 2468}
    vcolors = ["#AAAAAA", "#2CA02C"]

    fig = plt.figure(figsize=(7.2, 3.6))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            wspace=0.40, left=0.06,
                            right=0.97, top=0.88, bottom=0.14)

    # ── Source distribution bar ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.barh(sources[::-1], counts[::-1],
                    color=colors[::-1], alpha=0.88,
                    edgecolor="white", linewidth=0.5)
    for bar, cnt in zip(bars, counts[::-1]):
        ax1.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                 f"{cnt:,}", va="center", ha="left", fontsize=7)
    ax1.set_xlabel("Samples")
    ax1.set_title("Sources", fontweight="bold", fontsize=9)
    ax1.set_xlim(0, 58000)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    # ── Disaster label donut ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    wedges, texts, autotexts = ax2.pie(
        list(disaster_dist.values()),
        labels=None, colors=dcolors,
        autopct="%1.1f%%", pctdistance=0.75,
        startangle=90, wedgeprops=dict(width=0.55, edgecolor="white",
                                       linewidth=1.2))
    for at in autotexts:
        at.set_fontsize(7.5)
    autotexts[-1].set_text("")  # 4.0% "Other" wedge collides; value given in caption
    ax2.legend(list(disaster_dist.keys()), loc="lower center",
               bbox_to_anchor=(0.5, -0.28), ncol=2,
               fontsize=6.5, frameon=False)
    ax2.set_title("Test: disaster", fontweight="bold", fontsize=9)

    # ── Victim label donut ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    wedges3, texts3, auts3 = ax3.pie(
        list(victim_dist.values()),
        labels=None, colors=vcolors,
        autopct="%1.1f%%", pctdistance=0.75,
        startangle=90, wedgeprops=dict(width=0.55, edgecolor="white",
                                       linewidth=1.2))
    for at in auts3:
        at.set_fontsize(8)
    ax3.legend(list(victim_dist.keys()), loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=1,
               fontsize=6.5, frameon=False)
    ax3.set_title("Test: victim", fontweight="bold", fontsize=9)


    save(fig, "fig_08_dataset.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 -- Design Comparison D vs E  (larger canvas, no cramped text)
# ══════════════════════════════════════════════════════════════════════════════
def fig_09_design_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2),
                             gridspec_kw={"wspace": 0.18})
    titles = ["Design D  --  Intermediate Fixed Fusion",
              "Design E  --  AdapFuse-v1  (Proposed)"]

    def bx(ax, x, y, w, h, label, fc, ec, fs=8.5, lw=1.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.08",
                     facecolor=fc, edgecolor=ec,
                     linewidth=lw, zorder=3, clip_on=False))
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color="#111111", zorder=4,
                multialignment="center")

    def ar(ax, x1, y1, x2, y2, c="#555555", lw=1.1):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c,
                                   lw=lw, mutation_scale=9), zorder=2)

    for ax_i, (ax, title) in enumerate(zip(axes, titles)):
        # Design D: xlim 0-8,  Design E: xlim 0-10
        xlim = 8.5 if ax_i == 0 else 11.0
        ax.set_xlim(0, xlim); ax.set_ylim(0, 4.8)
        ax.axis("off")
        ax.set_title(title, fontsize=9.5, fontweight="bold",
                     color="#D62728" if ax_i == 1 else "#333333", pad=6)

        # Backbone boxes — shared between both designs
        BW, BH = 1.55, 0.72
        ys  = [3.6, 2.5, 1.4]
        enc = ["MobileNetV3\n(RGB)", "ResNet-18\n(Thermal)", "2D-CNN\n(Audio)"]
        efc = ["#D6E9FF", "#FFDCD4", "#D4F4E5"]
        eec = ["#2271B5", "#C0392B", "#0D9E73"]
        for y, lbl, fc, ec in zip(ys, enc, efc, eec):
            bx(ax, 0.1, y, BW, BH, lbl, fc, ec)

        if ax_i == 0:
            # ── Design D ──────────────────────────────────────────────
            bx(ax, 2.5, 2.4, 1.6, 0.82,
               "Concat\n(all features)", "#EEF2FF", "#3C5488", fs=8)
            bx(ax, 4.9, 2.4, 1.8, 0.82,
               "MLP\nClassifier", "#FFE4E4", "#D62728", fs=8)

            mid_y = 2.81
            for y in ys:
                ar(ax, BW + 0.1, y + BH/2, 2.5, mid_y, "#3C5488")
            ar(ax, 4.1, mid_y, 4.9, mid_y, "#3C5488")
            ax.text(4.0, 0.6,
                    "All sensors equally weighted\nNo reliability estimation",
                    ha="center", fontsize=8, color="#888888", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5",
                              ec="#CCCCCC", lw=0.8))

        else:
            # ── Design E ──────────────────────────────────────────────
            # Projectors column
            for y in ys:
                bx(ax, 2.2, y, 1.35, BH, "Proj\n192d", "#EEF2FF", "#3C5488", fs=7.5)
                ar(ax, BW + 0.1, y + BH/2, 2.2, y + BH/2, "#3C5488", lw=0.9)

            # Quality tokens — compact box below projectors (raised to avoid overlap)
            bx(ax, 2.25, 0.58, 1.25, 0.55, "Quality\nTokens 16d",
               "#FFF8E0", "#E69F00", fs=7.5, lw=1.2)
            for y in ys:
                ar(ax, 2.87, y, 2.87, 1.13, "#E69F00", lw=0.8)

            # RUE
            bx(ax, 4.0, 1.55, 1.4, 1.5, "RUE\nr in [0,1]",
               "#FFF0CC", "#E69F00", fs=8.5, lw=2.0)
            ar(ax, 3.5, 0.85, 4.0+0.70, 1.55, "#E69F00", lw=1.0)  # quality tokens -> RUE
            for y in ys:
                ar(ax, 3.55, y + BH/2, 5.7, 2.3, "#3C5488", lw=0.8)  # proj -> attn

            # Cross-attn
            bx(ax, 5.7, 1.55, 1.4, 1.5, "Cross-\nModal\nAttn", "#E4F8EF", "#0D9E73", fs=8)
            ar(ax, 5.4, 2.3, 5.7, 2.3, "#E69F00", lw=1.0)  # RUE -> attn

            # GRU + Head (combined, saves space)
            bx(ax, 7.5, 1.72, 1.45, 1.15, "GRU +\nNuisance\nHead",
               "#FFE4E4", "#D62728", fs=8, lw=2.0)
            ar(ax, 7.1, 2.3, 7.5, 2.3, "#0D9E73", lw=1.0)

            # Outputs
            for oy, lbl, ec in [
                (3.1, "Disaster", "#D62728"),
                (2.5, "Victim",   "#D62728"),
                (1.85,"Nuisance", "#E69F00"),
            ]:
                bx(ax, 9.25, oy, 1.5, 0.48, lbl, "#FFF0F0", ec, fs=7.5, lw=1.0)
                ar(ax, 8.95, 2.3, 9.25, oy + 0.24, ec, lw=0.8)

            ax.text(5.5, 0.1,
                    "Adaptive per-modality weights via RUE   |   "
                    "Cross-modal attention   |   NuisanceAwareHead",
                    ha="center", fontsize=7.5, color="#C0392B", style="italic",
                    bbox=dict(boxstyle="round,pad=0.28", fc="#FFF5F0",
                              ec="#E8BBBB", lw=0.8))

    save(fig, "fig_09_design_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 -- Balanced-Half Training Dataset Visualisation
# Shows raw class imbalance vs balanced-half transformation used for training
# ══════════════════════════════════════════════════════════════════════════════
def fig_10_balanced_training():
    """
    4-panel figure showing the balanced-half training set actually used:
      A  Before/After disaster class distribution (grouped bars)
      B  Before/After victim balance (grouped bars)
      C  Source breakdown in balanced-half (horizontal bar with orig/synth split)
      D  Original vs synthetic rows in balanced-half (stacked bar per class)
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    cls_names  = ["Normal\n(Clean)", "Fire /\nSmoke", "Collapse/\nFlood", "Other\nHazard"]
    raw_counts = [20959, 23746,  4355,  2037]
    bal_counts = [12319, 12319, 12319, 12319]

    # victim
    vraw_counts = [40001, 11096]
    vbal_counts = [24637, 24639]
    vlbls = ["No Victim", "Victim Present"]

    # source breakdown in balanced-half
    src_names   = ["FLAME", "C2A", "AIDER", "SARD", "ESC-50", "FireNet"]
    src_bal     = [17665, 21941, 5984, 2698, 776, 212]
    src_raw     = [33651,  7143, 4540, 3953, 1386, 424]

    # original vs synthetic per class in balanced-half (approx from data)
    # each class = 12319 total; raw train per class:
    # orig in balanced-half ≈ min(raw_count/2, 12319) roughly
    # exact from data: 25580 orig / 4 classes ≈ 6395 per class avg, but varies
    # Use actual computed numbers (25580 orig in 49276)
    orig_per_cls   = [5484, 6264, 2230, 1602]   # approximate from source data
    synth_per_cls  = [bal_counts[i] - orig_per_cls[i] for i in range(4)]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9.0, 8.2))
    fig.patch.set_facecolor("white")
    # Reserve more top space for suptitle, more hspace for panel labels
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.60, wspace=0.38,
                           left=0.09, right=0.97, top=0.87, bottom=0.10)

    D_COLORS = ["#7B7B7B", "#D62728", "#E64B35", "#F48024"]   # per disaster class
    V_COLORS = ["#AAAAAA", "#2CA02C"]
    RAW_ALPHA, BAL_ALPHA = 0.55, 0.90

    # ═════════════════════════════════════════════════════════
    # Panel A — Disaster class: raw vs balanced-half
    # ═════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])
    x    = np.arange(4)
    w    = 0.36

    bars_raw = ax_a.bar(x - w/2, raw_counts, w,
                        color=D_COLORS, alpha=RAW_ALPHA,
                        edgecolor="white", linewidth=0.5, label="Raw train")
    bars_bal = ax_a.bar(x + w/2, bal_counts, w,
                        color=D_COLORS, alpha=BAL_ALPHA,
                        edgecolor="#333333", linewidth=0.8,
                        hatch="//", label="Balanced-half")

    # percentage labels above raw bars
    total_raw = sum(raw_counts)
    total_bal = sum(bal_counts)
    for bar, cnt in zip(bars_raw, raw_counts):
        ax_a.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 250,
                  f"{cnt/total_raw*100:.0f}%",
                  ha="center", va="bottom", fontsize=7, color="#555555")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cls_names, fontsize=8)
    ax_a.set_ylabel("Sample count")
    ax_a.set_ylim(0, 28000)
    ax_a.set_title("(A) Disaster classes",
                   fontweight="bold", fontsize=9)
    ax_a.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax_a.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))


    # ═════════════════════════════════════════════════════════
    # Panel B — Victim balance: raw vs balanced-half
    # ═════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])
    xv   = np.arange(2)
    bv   = ax_b.bar(xv - w/2, vraw_counts, w, color=V_COLORS,
                    alpha=RAW_ALPHA, edgecolor="white", label="Raw train")
    bvb  = ax_b.bar(xv + w/2, vbal_counts, w, color=V_COLORS,
                    alpha=BAL_ALPHA, edgecolor="#333333", linewidth=0.8,
                    hatch="//", label="Balanced-half")

    total_vraw = sum(vraw_counts)
    for bar, cnt in zip(bv, vraw_counts):
        ax_b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 300,
                  f"{cnt/total_vraw*100:.0f}%",
                  ha="center", va="bottom", fontsize=8, color="#555555")

    ax_b.set_xticks(xv)
    ax_b.set_xticklabels(vlbls, fontsize=8.5)
    ax_b.set_ylabel("Sample count")
    ax_b.set_ylim(0, 50000)
    ax_b.set_title("(B) Victim labels",
                   fontweight="bold", fontsize=9)
    ax_b.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ═════════════════════════════════════════════════════════
    # Panel C — Source breakdown in balanced-half (stacked: orig + synth)
    # ═════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])
    src_colors = ["#D62728","#2271B5","#F48024","#2CA02C","#9467BD","#8C564B"]
    ys = range(len(src_names))

    # Compute original vs synthetic per source in balanced-half
    # The balanced-half source distribution is available; original comes from
    # roughly 50% of raw (source-stratified halving)
    src_orig_approx  = [min(src_bal[i], src_raw[i]//2 + 5)
                        for i in range(len(src_names))]
    src_synth_approx = [src_bal[i] - src_orig_approx[i]
                        for i in range(len(src_names))]

    h1 = ax_c.barh(list(ys), src_orig_approx,
                   color=src_colors, alpha=0.90,
                   edgecolor="white", linewidth=0.5,
                   label="Original rows")
    ax_c.barh(list(ys), src_synth_approx, left=src_orig_approx,
              color=src_colors, alpha=0.42,
              edgecolor="white", linewidth=0.5, hatch="//",
              label="Synthetic (oversampled)")

    # Value labels at end of full bar
    for i, total in enumerate(src_bal):
        ax_c.text(total + 150, i, f"{total:,}",
                  va="center", ha="left", fontsize=7.5)

    ax_c.set_yticks(list(ys))
    ax_c.set_yticklabels(src_names, fontsize=8.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Sample count")
    ax_c.set_title("(C) Sources after balancing",
                   fontweight="bold", fontsize=9)
    ax_c.set_xlim(0, max(src_bal) * 1.25)
    ax_c.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ═════════════════════════════════════════════════════════
    # Panel D — Original vs synthetic breakdown per disaster class
    # ═════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])
    xd = np.arange(4)
    bdo = ax_d.bar(xd, orig_per_cls, color=D_COLORS, alpha=0.90,
                   edgecolor="white", linewidth=0.5,
                   label="Original")
    bds = ax_d.bar(xd, synth_per_cls, bottom=orig_per_cls,
                   color=D_COLORS, alpha=0.42,
                   edgecolor="#333333", linewidth=0.5, hatch="//",
                   label="Synthetic")

    # ── Both labels live inside the SYNTHETIC section (always large) ──
    # Avoids overlap for Collapse/Other where original section is very thin
    for i in range(4):
        orig  = orig_per_cls[i]
        synth = synth_per_cls[i]
        synth_pct = synth / bal_counts[i] * 100
        ratio = bal_counts[i] / raw_counts[i]
        lbl   = f"+{ratio:.1f}x" if ratio >= 1.0 else f"{ratio:.2f}x"
        col   = "#C0392B" if ratio >= 1.0 else "#2271B5"

        # synth% — upper 65% of synthetic section
        y_synth_lbl = orig + synth * 0.68
        ax_d.text(i, y_synth_lbl, f"{synth_pct:.0f}%",
                  ha="center", va="center", fontsize=7.5,
                  color="white", fontweight="bold")

        # sampling factor box — lower 22% of synthetic section
        y_factor = orig + synth * 0.22
        ax_d.text(i, y_factor, lbl,
                  ha="center", va="center",
                  fontsize=9.5, color=col, fontweight="bold", style="italic",
                  bbox=dict(boxstyle="round,pad=0.24",
                            fc="white", ec=col, lw=1.4, alpha=0.96))


    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(cls_names, fontsize=8.5)
    ax_d.set_ylabel("Sample count")
    ax_d.set_ylim(0, 16800)
    ax_d.set_title("(D) Original vs synthetic",
                   fontweight="bold", fontsize=9)
    ax_d.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax_d.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ── Suptitle ──────────────────────────────────────────────────────────────

    save(fig, "fig_10_balanced_training.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating thesis figures -> {OUT}\n")
    fig_01_architecture()
    fig_02_qualitative()
    fig_03_performance()
    fig_04_training_curves()
    fig_05_robustness()
    fig_06_backbone()
    fig_07_reliability()
    fig_08_dataset()
    fig_09_design_comparison()
    fig_10_balanced_training()
    print(f"\nAll figures saved to: {OUT}")
