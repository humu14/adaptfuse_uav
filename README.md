# UniAdapFuse / AdapFuse-UAV

**Reliability-aware multimodal fusion for UAV disaster perception.**
B.Sc. thesis, Department of Computer Science and Engineering, Brac University (2026).
Humaira Ayesha · Md Saqline Hossain · Badrunnaher Pantho · Fatema Siddika Tanha.

The repository contains three things:

1. **Thesis source** (LaTeX): `main.tex`, `chapters/`, `core/`, `appendix/`, `bibliography/`, `images/`
2. **Research code**
   * `code_p2/adaptfuse_uav/`: tri-modal (RGB + thermal + audio) scene classification, including AdapFuse-V1/V2, UniAdapFuse, RUE, NuisanceAwareHead and all baselines
   * `adaptfuse_od/`: fire and smoke object detection on D-Fire (YOLO26 / YOLOv10 / RT-DETR, plus the Hazard Prior Gate)
3. **Demo app**, `code_p2/adaptfuse_uav/demo_app/`: upload one RGB *or* thermal video and see detections, scene classification and audio labels in real time

```text
.
├── main.tex  chapters/  core/  appendix/  bibliography/  images/   ← thesis
├── nomenclature.tex  .latexmkrc                                   ← thesis build config
├── code_p2/
│   ├── adaptfuse_uav/            ← classification code, configs, results (outputs/logs/*.json)
│   │   └── demo_app/             ← live + Gradio web interfaces (weights included)
│   └── *.md                      ← experiment plans
├── adaptfuse_od/                 ← detection code, configs, results (outputs/**/*.json)
└── *.md                          ← project reference / runbook / evaluation notes
```

---

## 1 · Run the demo (≈ 5 minutes)

```bash
git clone <this-repo-url> adaptfuse-uav
cd adaptfuse-uav/code_p2/adaptfuse_uav/demo_app

# PyTorch first, matching your CUDA (CPU also works, just slower):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python live/server.py        # real-time app → http://127.0.0.1:8000
python gradio_app.py         # batch app     → http://127.0.0.1:7860
```

On Windows, double-click `run_live.bat` / `run_gradio.bat`. If you clone into a deep folder
and see `Filename too long`, run `git config --global core.longpaths true` first. All demo weights (~102 MB) are
committed under `demo_app/weights/`, so nothing is downloaded on first run. Click
**sample_rgb** or **sample_thermal** to try it without your own footage. Details,
model provenance and limitations are in [`demo_app/README.md`](code_p2/adaptfuse_uav/demo_app/README.md).

| Shown in the app | Source |
|---|---|
| Disaster / victim / nuisance labels, RUE reliability | AdapFuse-V1 ensemble (grouped seeds 41 + 42), trained in pipeline |
| Audio-only label | AudioCNN baseline, trained in pipeline |
| Fire / smoke boxes | YOLO26s fine-tuned on CLAHE D-Fire, trained in pipeline |
| Person boxes | YOLO26n **COCO-pretrained** (not fine-tuned) |
| Sound tags | PANNs Cnn6 **AudioSet-pretrained** (not fine-tuned) |

---

## 2 · Build the thesis PDF

Requirements: a TeX distribution with `pdflatex`, `biber`, `makeindex` and `latexmk`
([MiKTeX](https://miktex.org/) on Windows, TeX Live on Linux/macOS). Packages used
include `biblatex` (backend `biber`, style `ieee`), `nomencl`, `hyperref`, `tikz`/`pgfplots`,
`booktabs`, `tabularx`, `amsmath` and `graphicx`. MiKTeX installs missing packages on first build.

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

`.latexmkrc` registers the `nomencl` rule, so the nomenclature list is built
automatically. Manual equivalent:

```bash
pdflatex main
biber main
makeindex -s nomencl.ist -o main.nls main.nlo
pdflatex main
pdflatex main
```

Clean up with `latexmk -c`. Build outputs (`*.aux`, `*.bbl`, `main.pdf`, …) are git-ignored.

| File | Content |
|---|---|
| `core/` | title page, declaration, approval, ethics, abstract, acknowledgement, dedication |
| `chapters/chapter_1.tex` | Introduction |
| `chapters/chapter_2.tex` | Literature review |
| `chapters/chapter_3.tex` | Requirements, impacts and constraints |
| `chapters/chapter_5.tex` | Proposed methodology |
| `chapters/chapter_6.tex` | Result analysis |
| `chapters/chapter_9.tex` | Conclusion |
| `appendix/` | appendices |
| `bibliography/references.bib` | all references (biblatex) |
| `nomenclature.tex` | symbol list |

---

## 3 · Method in brief

Each modality $m \in \{\text{rgb}, \text{th}, \text{au}\}$ has an encoder
(MobileNetV3-Small, 1-channel ResNet-18, AudioCNN on a $64\times63$ log-mel), a projection
$\mathbf f_m \in \mathbb R^{192}$ and a 16-d quality token $\mathbf q_m$. The
**Reliability & Uncertainty Estimator** reads all three,

$$
\mathbf s = \big[\,\mathbf q_m,\ \lVert \mathbf f_m \rVert_2,\ \operatorname{Var}(\mathbf f_m)\,\big]_{m},
\qquad
\mathbf r = \boldsymbol\delta \odot \sigma\!\big(g_r(\mathbf s)\big)\in[0,1]^3,
\qquad
\mathbf u = \operatorname{softplus}\!\big(g_u(\mathbf s)\big),
$$

where $\delta_m\in\{0,1\}$ marks whether modality $m$ is present. Features are
reliability-weighted, fused by cross-modal self-attention and passed through a GRU:

$$
\tilde{\mathbf f}_m = r_m\,\delta_m\,\mathbf f_m,\qquad
\mathbf h = \operatorname{GRU}\!\Big(\tfrac13\textstyle\sum_m
\operatorname{LN}\big(\operatorname{MHA}(\tilde{\mathbf F})+\tilde{\mathbf F}\big)_m\Big),
$$

and the **NuisanceAwareHead** predicts three outputs,

$$
p(y_\text{dis}\mid\mathbf h)\in\Delta^{3},\qquad
p(y_\text{vic}\mid\mathbf h)\in\Delta^{1},\qquad
p(y_\text{nui}\mid\mathbf h)\in\Delta^{4},
$$

(disaster: 4 classes · victim: 2 · nuisance: *clean, solar heating, RGB false fire,
audio false alarm, partial occlusion*).

The **Hazard Prior Gate** for detection adds physical cue maps at stride 4 as a residual
input to YOLO26n:

$$
\mathbf c_\text{fire}=2R-G-B,\qquad
\mathbf c_\text{smoke}=1-S,\qquad
\mathbf c_\text{edge}=\lVert\nabla Y\rVert_\text{Sobel}.
$$

## 4 · Headline results (from the thesis)

| Result | Value | Where |
|---|---|---|
| AdapFuse-V1 combined F1 (row-level test split) | 0.9874 | `code_p2/adaptfuse_uav/outputs/logs/adaptfuse_v1_test_results.json` |
| UniAdapFuse v2 combined F1 / detection mAP@50 | 0.9857 / **0.0** (joint detector not validated) | `…/uni_adaptfuse_v2_test_results.json` |
| Best D-Fire detector, YOLO26s (CLAHE) mAP@50 | 0.7710 | `adaptfuse_od/outputs/accuracy_study/` |
| YOLO26n baseline vs. HPG-YOLO26n mAP@50 | 0.7147 in 4.45 h vs. 0.7053 in 1.96 h | `adaptfuse_od/outputs/efficient_study/*/training_summary.json` |
| AdapFuse-V1 latency (RTX 5090, 32 GB) | 14.26 ms / frame | thesis Ch. 6 |

Limitations are discussed in Chapters 6 and 9. In short: row-level splits,
pseudo-thermal training data, randomly paired ESC-50 audio, RUE gates that saturate at
0/1, and no person-box evaluation.

## 5 · Reproducing training

* Classification: see [`code_p2/adaptfuse_uav/README.md`](code_p2/adaptfuse_uav/README.md)
  and `code_p2/NEW_EXPERIMENTS.md`. Configs are in `code_p2/adaptfuse_uav/configs/`.
  Rebuild the split CSVs with `scripts/build_metadata.py`. Large training splits are not
  committed because they contain absolute local paths.
* Detection: see [`adaptfuse_od/OBJECT_DETECTION_IMPLEMENTATION.md`](adaptfuse_od/OBJECT_DETECTION_IMPLEMENTATION.md).
* Datasets (not in git): FLAME, AIDER, SARD, C2A, FireNet, ESC-50, D-Fire. Download
  scripts: `code_p2/adaptfuse_uav/scripts/download_data.py` and
  `adaptfuse_od/data/download_dfire.py`.
* Training checkpoints (`*.pt`, `*.pth`) are git-ignored. Only the slim demo weights in
  `demo_app/weights/` are committed.

## Citation

```bibtex
@thesis{uniadapfuse2026,
  title  = {UniAdapFuse: Reliability-Aware Multimodal Fusion for UAV Disaster Perception},
  author = {Ayesha, Humaira and Hossain, Md Saqline and Pantho, Badrunnaher and Tanha, Fatema Siddika},
  school = {Brac University, Department of Computer Science and Engineering},
  type   = {B.Sc. thesis},
  year   = {2026}
}
```
