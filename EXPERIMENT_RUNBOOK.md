# Experiment runbook and hardware-specific time estimates

No training or evaluation is launched by this document. Commands are provided for the project authors to run manually.

## Detected local hardware

- GPU: NVIDIA GeForce GTX 960, 4 GB VRAM
- CPU: AMD 6 physical cores / 12 logical threads
- RAM: 15.9 GB
- Operating system: Windows

The existing thesis timings were recorded on an RTX 5090 workstation. They should not be used as estimates for this GTX 960. The ranges below are deliberately conservative because the requested local timing run was stopped before completing a batch. The current pipeline also computes log-mel features during loading, so audio caching is strongly recommended.

## Recommended execution order

1. Grouped, leakage-resistant repeated-seed classification.
2. Person bounding-box evaluation using the SARD annotations already present.
3. Repair the joint detector and require non-zero validation mAP before a full run.
4. Add supervised degradation/nuisance labels.
5. Add true temporal sequences.
6. Integrate registered sensor IR and synchronized UAV audio when those data become available.
7. Profile the final model on the intended embedded device.

This order gives the fastest increase in scientific credibility without pretending that unavailable sensors or hardware are already validated.

## Experiment 1: grouped split and repeated-seed confidence intervals

Status: implementation complete. The generated seed-42 split contains 72,997 rows with zero capture-group overlap and zero ESC-50 path overlap across train, validation, and test.

### A. Rebuild the grouped metadata if needed

```powershell
cd "E:\Research Projects\reserch writing\Co-Sup\code_p2\adaptfuse_uav"
python scripts\build_grouped_splits.py --seed 42 --flame-block-size 250 --overwrite
```

FLAME does not expose true video/flight IDs in the delivered files. Its 250-frame groups are therefore an explicit adjacent-frame proxy. C2A augmentations and SARD Roboflow variants are grouped by original-image identity. ESC-50 folds 1--3, 4, and 5 are assigned to train, validation, and test; visual rows are re-paired only with clips from their own partition.

### B. Cache the 2,000 unique audio features once

```powershell
$env:NO_ALBUMENTATIONS_UPDATE="1"
python scripts\cache_audio_logmel.py `
  --metadata-dir data\metadata_grouped\seed_42 `
  --output-dir data\audio_cache\logmel_64x63
```

This is preprocessing, not a model experiment. Expected GTX-960-machine time: approximately 30--120 minutes, including potentially slow first-use audio-library initialization; expected cache size: roughly 35--50 MB.

### C. Run one seed first

Start with batch size 2. If it is stable and GPU memory remains below about 3.5 GB, retry later seeds with batch size 4.

```powershell
python training\train.py `
  --config configs\adaptfuse_v1_grouped_gtx960.yaml `
  --metadata_dir data\metadata_grouped\seed_42 `
  --output_dir outputs\grouped_repeated\seed_41 `
  --seed 41 `
  --batch_size 2 `
  --num_workers 0
```

Expected time on the detected GTX 960: 24--72 hours for one seed, depending on early stopping and storage/audio-cache throughput. Expected RTX 5090-class time, based on existing repository runs: approximately 1.5--4 hours.

### D. Run the five-seed suite

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --metadata-dir data\metadata_grouped\seed_42 `
  --seeds 41 42 43 44 45 `
  --batch-size 2 `
  --num-workers 0
```

Expected sequential GTX 960 time: approximately 5--15 days continuously. Five seeds are recommended; three seeds are the minimum defensible fallback. Keep the grouped split fixed at seed 42 while varying only model/training seeds.

The final summary is written to:

```text
outputs/grouped_repeated/adaptfuse_v1_grouped_repeated_summary.json
```

To recompute the confidence intervals without training:

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --metadata-dir data\metadata_grouped\seed_42 `
  --seeds 41 42 43 44 45 `
  --summarize-only
```

Report mean, sample standard deviation, and the 95% Student-t confidence interval for combined macro-F1, disaster macro-F1, victim-presence macro-F1, accuracy, and loss.

## How to run Experiments 2--8

Open PowerShell once and establish the project directory:

```powershell
$ProjectRoot = "E:\Research Projects\reserch writing\Co-Sup\code_p2\adaptfuse_uav"
Set-Location -LiteralPath $ProjectRoot
$env:NO_ALBUMENTATIONS_UPDATE = "1"
```

The commands below never manufacture missing evidence. A section marked
**data-gated**, **implementation-gated**, or **hardware-gated** must pass its stated
prerequisite before its training command is scientifically valid. All training
commands automatically create the reproducibility bundle documented later in this
runbook.

### Experiment 2: replace pseudo-thermal with registered sensor IR

Status: **data-gated**. First create
`data\metadata_registered_ir\seed_42\{train,val,test}.csv`. Keep the grouped RGB
split fixed, replace `thermal_path` with the geometrically registered sensor-IR
frame, and retain `capture_group`, `source_dataset`, labels, and modality flags.
Do not point `thermal_path` to an RGB-derived image.

Check that the required manifests exist:

```powershell
$IrMeta = "data\metadata_registered_ir\seed_42"
@("train.csv", "val.csv", "test.csv", "split_report.json") | ForEach-Object {
  $RequiredPath = Join-Path $IrMeta $_
  if (-not (Test-Path -LiteralPath $RequiredPath)) {
    throw "Missing required registered-IR artifact: $RequiredPath"
  }
}
```

Run three training seeds after the metadata and registration audit are complete:

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --config configs\adaptfuse_v1_grouped_gtx960.yaml `
  --metadata-dir $IrMeta `
  --output-root outputs\registered_ir_repeated `
  --seeds 41 42 43 `
  --batch-size 2 `
  --num-workers 0
```

Compare its summary with the pseudo-thermal Experiment 1 summary. The new summary
will be `outputs\registered_ir_repeated\adaptfuse_v1_grouped_repeated_summary.json`.

### Experiment 3: replace ESC-50 reuse with synchronized UAV audio

Status: **data-gated**. Build
`data\metadata_uav_audio\seed_42\{train,val,test}.csv` so every `audio_path` is a
clip synchronized to the corresponding RGB/IR capture. Split by capture sequence,
not by individual frame or extracted audio window.

Cache the synchronized log-mel features:

```powershell
$AudioMeta = "data\metadata_uav_audio\seed_42"
$AudioCache = "data\audio_cache\synchronized_uav_logmel_64x63"
python scripts\cache_audio_logmel.py `
  --metadata-dir $AudioMeta `
  --output-dir $AudioCache
```

Run three seeds while explicitly selecting the new cache:

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --config configs\adaptfuse_v1_grouped_gtx960.yaml `
  --metadata-dir $AudioMeta `
  --audio-cache-dir $AudioCache `
  --output-root outputs\synchronized_audio_repeated `
  --seeds 41 42 43 `
  --batch-size 2 `
  --num-workers 0
```

### Experiment 4: supervised nuisance/degradation targets and RUE calibration

Status: **implementation- and label-gated**. The metadata must contain audited
`nuisance_label`, per-modality degradation flags, and degradation severity. The
dataset must return them, and the trainer must pass them to `CombinedLoss`.
Currently those loss inputs are intentionally `None`; therefore do not start this
experiment until this preflight check stops with no error:

```powershell
$Unwired = Select-String -Path training\trainer.py `
  -SimpleMatch "nuisance_gt=None", "corruption_flags=None"
if ($Unwired) {
  throw "Experiment 4 is not wired: trainer still discards supervised nuisance/degradation targets."
}
```

After the labels and code path are implemented, run the RUE model and its strictly
matched no-RUE control on the same three seeds:

```powershell
$NuisanceMeta = "data\metadata_nuisance_supervised\seed_42"
python scripts\run_grouped_repeated_seeds.py `
  --config configs\adaptfuse_v1_grouped_gtx960.yaml `
  --metadata-dir $NuisanceMeta `
  --output-root outputs\nuisance_supervised_rue `
  --seeds 41 42 43 `
  --batch-size 2 `
  --num-workers 0

python scripts\run_grouped_repeated_seeds.py `
  --config configs\adaptfuse_v1_grouped_no_rue_gtx960.yaml `
  --metadata-dir $NuisanceMeta `
  --output-root outputs\nuisance_supervised_no_rue `
  --seeds 41 42 43 `
  --batch-size 2 `
  --num-workers 0
```

Do not close Experiment 4 with classification F1 alone. The implementation must
also calculate reliability--severity Spearman correlation, degradation AUROC, and
ECE from held-out row-level predictions.

### Experiment 5: true temporal sequences for the GRU

Status: **implementation- and sequence-gated**. The current model calls the GRU on
a sequence of length one, so the following check is expected to find the unresolved
line and block the experiment:

```powershell
$LengthOneGRU = Select-String -Path models\full_model.py -SimpleMatch "fused.unsqueeze(1)"
if ($LengthOneGRU) {
  throw "Experiment 5 is not ready: the GRU still receives a one-step tensor."
}
```

Required implementation: construct sequence-disjoint metadata with
`sequence_id`, `frame_index`, and synchronized modality paths; return tensors shaped
`B x T x ...`; and make the model consume all `T` steps. Once a real temporal config
exists, use this command template:

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --config configs\adaptfuse_v1_temporal_gtx960.yaml `
  --metadata-dir data\metadata_sequences\seed_42 `
  --output-root outputs\temporal_t4_repeated `
  --seeds 41 42 43 `
  --batch-size 1 `
  --num-workers 0
```

`configs\adaptfuse_v1_temporal_gtx960.yaml` is intentionally not supplied yet:
creating that name without a sequence-aware loader/model would falsely imply the
experiment is implemented.

### Experiment 6: repair and gate the joint detector

Status: **deferred because of current resource constraints**. Detection is not
required to complete the minimum three-seed classification study. The diagnostic
configuration disables KD so a teacher projection-size mismatch cannot obscure
detector failures. When a stronger GPU is available, run only the 20-epoch gate
first:
Detection begins after the 15 classification-only epochs in this configuration:

```powershell
python training\train.py `
  --config configs\uni_adaptfuse_v2_grouped_gtx960.yaml `
  --metadata_dir data\metadata_grouped\seed_42 `
  --output_dir outputs\joint_detector_gate `
  --epochs 20 `
  --batch_size 1 `
  --num_workers 0 `
  --seed 41
```

Do not launch this gate on the GTX 960 unless joint detection is essential to the
immediate deliverable. Resume this experiment later on a stronger GPU, then apply
the stopping rule below. Only proceed to the three-seed detector study if the
gate produces positive validation mAP@0.5.

Apply the stopping rule to validation history:

```powershell
$History = Get-Content `
  outputs\joint_detector_gate\logs\uni_adaptfuse_v2_grouped_gtx960_history.json `
  -Raw | ConvertFrom-Json
$BestValMap = ($History | Measure-Object -Property val_map50 -Maximum).Maximum
if ($null -eq $BestValMap -or $BestValMap -le 0) {
  throw "STOP: validation mAP@0.5 is still zero; debug the detector before a full run."
}
```

Only after that gate passes, run three complete seeds:

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --config configs\uni_adaptfuse_v2_grouped_gtx960.yaml `
  --metadata-dir data\metadata_grouped\seed_42 `
  --output-root outputs\joint_detector_repeated `
  --seeds 41 42 43 `
  --batch-size 1 `
  --num-workers 0
```

### Experiment 7: latency, memory, power, and thermal behavior

Status: workstation latency is runnable now; embedded power/thermal claims are
**hardware-gated**. This GTX-960 command is a development measurement only:

```powershell
python evaluation\profile_latency.py `
  --config configs\adaptfuse_v1_grouped_gtx960.yaml `
  --checkpoint outputs\grouped_repeated\seed_41\checkpoints\adaptfuse_v1_grouped_best.pth `
  --output_dir outputs\grouped_repeated\seed_41 `
  --batch_size 1
```

For an NVIDIA Jetson target, copy the exact checkpoint/config/code snapshot to the
device, then record telemetry while profiling:

```bash
mkdir -p outputs/embedded_profile
tegrastats --interval 1000 --logfile outputs/embedded_profile/tegrastats.log &
TEGRASTATS_PID=$!
END_TIME=$((SECONDS + 600))
while [ "$SECONDS" -lt "$END_TIME" ]; do
  python evaluation/profile_latency.py \
    --config configs/adaptfuse_v1_grouped_gtx960.yaml \
    --checkpoint outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth \
    --output_dir outputs/embedded_profile \
    --batch_size 1
done
kill "$TEGRASTATS_PID"
```

Run at least 10 minutes after warm-up and report device model, power mode, clocks,
software stack, input shape, batch size, latency percentiles, throughput, memory,
power, and temperature. For a non-Jetson target, replace `tegrastats` with the
vendor's measurement tool. The current profiler reports mean latency; extend its
raw timing output to compute p50/p95/p99 before making a publication claim.
GTX-960 measurements cannot support an embedded claim.

### Experiment 8: person bounding-box evaluation

Status: annotation audit is runnable; a person-specific detector and AP evaluator
are **implementation-gated**. Inspect how many grouped SARD rows currently have
boxes:

```powershell
$SardRows = Import-Csv data\metadata_grouped\seed_42\test.csv | `
  Where-Object { $_.source_dataset -eq "sard" }
$SardRows | Group-Object has_bbox | Select-Object Name, Count
```

Before training, verify that retained YOLO class IDs denote people and are not mixed
with fire/smoke/object classes. Then implement a person detector configuration and
an evaluator that reports AP50, AP50:95, precision, recall, and deterministic box
visualizations. The eventual three-seed command is:

```powershell
python scripts\run_grouped_repeated_seeds.py `
  --config configs\person_detector_grouped_gtx960.yaml `
  --metadata-dir data\metadata_person_boxes\seed_42 `
  --output-root outputs\person_detector_repeated `
  --seeds 41 42 43 `
  --batch-size 1 `
  --num-workers 0
```

`configs\person_detector_grouped_gtx960.yaml` is intentionally not supplied until
the class-specific detector loss, decoding/NMS, and COCO-style AP evaluator exist.

## Estimated time for all proposed experiments

| # | Experiment | Preparation or data time | GTX 960 compute estimate | Important constraint |
|---:|---|---|---|---|
| 1 | Grouped split + five training seeds | Split: already complete; audio cache: 30--120 min | 24--72 h/seed; 5--15 days for five | FLAME grouping is a frame-block proxy because true flight IDs are absent |
| 2 | Registered sensor-IR replacement | 2--10 days for an existing paired dataset; 2--8 weeks if collecting/registering new data | 24--72 h/seed; 3--9 days for three | Cannot be completed credibly with RGB-derived pseudo-thermal data |
| 3 | Synchronized UAV audio | 2--5 days for an existing dataset; 2--6 weeks for hardware capture and synchronization | 24--72 h/seed plus 1--3 h preprocessing | ESC-50 label matching is not a substitute |
| 4 | Nuisance/degradation supervision | 4--8 h for synthetic degradation targets; 2--5 days for audited nuisance labels | 24--72 h/seed; 3--9 days for three | Must report gate--severity correlation, AUROC, ECE, and a matched no-RUE baseline |
| 5 | True temporal GRU sequences | 1--3 days to recover sequence manifests and verify boundaries | 48--160 h/seed for a four-frame sequence; 6--20 days for three | Do not use random independent frames as a temporal sequence |
| 6 | Repair joint detection | 2--7 days debugging and validation | 48--120 h/seed; 6--15 days for three | Four GB VRAM may require batch 1, gradient accumulation, and reduced resolution; stop if validation mAP remains zero |
| 7 | Embedded latency/power/thermal profile | 1--3 days export/integration once hardware exists | 4--8 h benchmark plus 1--2 days thermal/power soak | Cannot be estimated or claimed from the GTX 960; the target device is required |
| 8 | Person bounding-box evaluation | 4--12 h to normalize existing SARD boxes and audit classes | 24--72 h/seed; 3--9 days for three | Current victim head predicts presence only; use AP50/AP50:95 and qualitative boxes |

These ranges are wall-clock planning estimates, not measured results. Data acquisition time and human annotation time are separate from GPU time and can dominate Experiments 2, 3, 4, and 7.

## Minimum outputs to save for every run

Every `python training\train.py ...` command above now saves these automatically; no
additional flag is required. For an output directory such as
`outputs\grouped_repeated\seed_41`, the run produces:

```text
checkpoints/
  <experiment>_best.pth
  <experiment>_final.pth
  <experiment>_latest.pth
logs/
  <experiment>_history.json
  <experiment>_test_results.json
run_artifacts/
  command.txt
  source_config.yaml
  resolved_config.yaml
  dataset_manifest.json
  split_report.json
  code_snapshot.json
  environment.json
  epoch_metrics.csv
  test_predictions.csv
  test_metrics_per_class.csv
  test_metrics_per_source.csv
  failure_gallery_manifest.csv
  failure_gallery.png
  run_summary.json
```

The dataset manifest contains SHA-256 checksums for `train.csv`, `val.csv`,
`test.csv`, and the split report, plus one aggregate split checksum. The code
snapshot records the Git commit and dirty state when Git exists; otherwise it
records a deterministic SHA-256 digest and per-file hashes for the Python and YAML
source tree. The resolved configuration includes command-line overrides and safe
runtime fallbacks. `environment.json` records the random seed, Python and library
versions, CUDA/cuDNN and NVIDIA driver versions, GPU identity, and system
information. `run_summary.json` records total end-to-end wall time, peak allocated
and reserved VRAM, test metrics, and hashes of the best and final checkpoints.

The failure gallery is reproducible rather than hand-picked: errors are ordered by
the number of failed tasks (disaster/victim), then predicted error confidence, then
sample ID. The exact selected rows and rule are stored in
`failure_gallery_manifest.csv`.

## Publication stopping rules

- Do not claim temporal modeling unless sequence length is greater than one and groups are sequence-disjoint.
- Do not claim dynamic reliability unless reliability changes correlate with held-out degradation severity and beat a matched fixed/no-RUE baseline.
- Do not claim nuisance suppression unless nuisance targets are supplied and evaluated independently.
- Do not claim joint localization while detection mAP is zero.
- Do not claim sensor-thermal robustness using pseudo-thermal images derived from RGB.
- Do not claim embedded performance from workstation measurements.
