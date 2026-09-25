# UniAdapFuse thesis and repository evaluation

This report records the evidence audit used to revise `main.pdf`. It distinguishes what the delivered data, code, and saved results demonstrate from what remains a proposal or future experiment.

## Overall assessment

The project has two defensible experimental tracks:

1. A broad within-collection comparison of scene-level disaster and victim-presence classifiers on heterogeneous metadata.
2. A separately validated RGB smoke/fire detector on D-Fire, where YOLO26s+CLAHE reaches 0.7710 mAP50 and 0.4480 mAP50:95.

The strongest research value is the breadth of implemented baselines, saved experimental artifacts, missing-modality execution, and a useful HPG training-time/accuracy trade-off. The most important validity limitation is that the classification data are not genuinely synchronized RGB--thermal--audio observations and are split by row rather than by source video, flight, or capture sequence. Near-ceiling classification results must therefore be described as within-collection performance, not operational UAV generalization.

## Recomputed data provenance

Counts below were recomputed from `train.csv`, `val.csv`, and `test.csv` rather than copied from manuscript prose.

| Item | Audited value | Interpretation |
|---|---:|---|
| Metadata rows | 72,997 | 51,097 train; 10,949 validation; 10,951 test |
| Populated RGB paths | 70,997 | 70,997 unique RGB files |
| Populated thermal paths | 70,997 | Every path is under `datasets/pseudo_thermal`; no sensor-IR path is present |
| Audio assignments | 39,003 | Only 2,000 unique ESC-50 clips are reused and paired by label |
| Rows with box annotations | 15,970 | C2A: 10,215; SARD: 5,755 |
| Synchronized tri-modal sequences | 0 demonstrated | Audio is not captured with the visual scene; pseudo-thermal is derived from RGB |

Consequences:

- “Thermal” classification results are pseudo-thermal/image-transform results, not independent LWIR sensing results.
- Audio ablations measure a constructed label-matched channel, not synchronized UAV acoustics or measured rotor-noise robustness.
- RGB and pseudo-thermal are statistically dependent views of the same source image.

## Method and implementation audit

### Supported by the implementation

- Modality-specific encoders and 192-dimensional projected tokens.
- Sigmoid modality gates, attention over three modality tokens, missing-modality masks, and disaster/victim-presence heads.
- A nuisance-logit branch and optional degradation-logit output exist in the model.
- Separate D-Fire detector experiments and a joint UniAdapFuse prototype are present.
- Reproducible saved metrics exist for the main comparison and ablation runs.

### Important implementation boundaries

- The GRU receives `fused.unsqueeze(1)`, so its sequence length is one. In the reported experiments it is a gated nonlinear refinement block, not a temporal model.
- `training/trainer.py` passes `nuisance_gt=None` and `corruption_flags=None`. The nuisance and degradation-supervision losses are therefore zero in the saved training path.
- Configuration files set `corruption_prob: 0.3`, but the base loader is not given a `corruption_type`. The named smoke, blur, low-light, drift, and audio-noise transforms are used in separate stress scripts, not activated by the reported base-training path.
- Reliability outputs in the saved sweep are nearly constant: RGB and pseudo-thermal are approximately 0.972, while audio is approximately 0.006. This does not demonstrate corruption-sensitive dynamic gating.
- The classifier split is label-stratified by row. It is not grouped by video, flight, source sequence, or acquisition event, leaving a material adjacent-frame leakage risk.

## Results audit

| Result | Evidence-based reading |
|---|---|
| AdapFuse-v1 combined macro-F1: 0.9874 | Valid as a row-level, within-collection result; external/generalization performance is unknown |
| Design D combined macro-F1: 0.9869 | AdapFuse-v1 gains about 0.05 percentage points and has 2.9x lower saved test loss |
| Row-level FPR: 0.0073 | A classifier statistic; not an operational false-alarm study and not attributable to the unsupervised nuisance branch |
| Smoke-overlay severity-5 F1: 0.8973 | Valid for the offline synthetic transform; learned gates remain nearly constant |
| UniAdapFuse-v2 combined F1: 0.9857 | Classification recovery is demonstrated |
| UniAdapFuse-v2 detection mAP50: 0.0 | Usable joint localization is not demonstrated |
| YOLO26s+CLAHE D-Fire mAP50: 0.7710 | Strongest validated dense smoke/fire detection result in the repository |
| HPG YOLO26n: 0.7053 mAP50 in 1.96 h | Useful shorter-run trade-off versus baseline YOLO26n at 0.7147 in 4.45 h; not a controlled convergence proof |
| 14.26 ms / 70.1 FPS | RTX 5090 workstation measurement, not embedded UAV hardware |

## Figure revisions

All revised figures use project data or saved project logs. No synthetic or AI-generated dataset imagery is presented as evidence.

| Thesis figure | New asset | Revision |
|---|---|---|
| Figure 1.1 | `images/uav_multimodal_framework.png` | Separates the scene-classification study, independent D-Fire detector, failed joint-detection validation, and future embedded deployment |
| Figure 4.1 | `images/generated/fig_01_architecture.png` | Code-faithful AdapFuse-v1 architecture with real project thumbnails, one-step GRU disclosure, and unsupervised nuisance branch |
| Figure 4.2 | `images/generated/fig_02_qualitative.png` | Four fixed `test.csv` rows, one per class, with real RGB/pseudo-thermal inputs and actual strict checkpoint outputs; missing audio is explicit |
| Figure 4.5 | `images/generated/fig_09_design_comparison.png` | Clean top-conference-style comparison of Design D and E using saved F1/loss values and an honest structure/evidence boundary |
| New Figure 4.6 | `images/generated/fig_14_data_provenance.png` | Visualizes exact modality provenance, annotation coverage, and absent supervision |
| Figure 5.3 | `images/generated/fig_05_robustness.png` | Rebuilt from saved JSON logs only; removes the legacy hypothetical comparator and shows the near-constant gate response |

The figures are generated at 320 dpi with consistent typography, a colorblind-aware palette, restrained borders, and direct annotations. Regenerate them with:

```powershell
cd code_p2\adaptfuse_uav
$env:NO_ALBUMENTATIONS_UPDATE='1'
python -u scripts\generate_conference_figures.py
```

## Recommended next experiments

Implementation commands and hardware-specific planning estimates are provided in `EXPERIMENT_RUNBOOK.md`. The grouped split and repeated-seed tooling are implemented; no full training runs have been launched.

1. Rebuild splits by source video/flight/capture sequence and report repeated-seed confidence intervals.
2. Replace RGB-derived pseudo-thermal maps with registered sensor-IR data.
3. Replace label-matched ESC-50 reuse with synchronized UAV audio.
4. Supply explicit nuisance and degradation labels, then evaluate gate--corruption correlation and calibration.
5. Feed true frame sequences to the GRU or rename/remove it as a temporal component.
6. Repair and evaluate the joint detector until its mAP is non-zero, with qualitative box predictions and per-class AP.
7. Benchmark latency, memory, power, and thermal behavior on the intended embedded UAV device.
8. Add person bounding-box evaluation; current victim results are presence classification only.

## Publication-readiness judgment

With the revised wording, the thesis is substantially more defensible as an engineering study and repository audit. The standalone D-Fire detector track and the breadth of classification comparisons are useful. A top vision-conference claim of robust synchronized multimodal UAV perception would still require grouped/cross-source evaluation, real paired sensors, repeated trials, supervised reliability evidence, and a functioning joint detector. Until then, the safest framing is: heterogeneous-input scene classification with missing-modality handling, plus separately validated RGB hazard detection and a documented joint-model failure case.
