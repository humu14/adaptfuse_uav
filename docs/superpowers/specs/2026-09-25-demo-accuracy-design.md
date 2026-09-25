# Demo app accuracy upgrade — design spec

**Date:** 2026-09-25
**Scope:** `code_p2/adaptfuse_uav/demo_app/` (live FastAPI app, Gradio app, shared `engine/`)
**Status:** approved in chat; awaiting written-spec review

## 1. Intent

The supervisor demo takes one RGB **or** thermal video and shows detections, scene
classification and audio labels in real time. In use it showed two problems:

1. **Fire boxes on normal scenes.** The bundled detector
   (`efficient_yolo26n_baseline/best.pt`) has swapped class names: its checkpoint maps
   `{0: fire, 1: smoke}`, but D-Fire labels are `0 = smoke, 1 = fire`. Every "fire" box
   the app drew was really smoke, fog or snow, and every "smoke" box was really fire.
2. **Weak collapse accuracy.** The app uses the row-level AdapFuse-V1 checkpoint. Its
   0.987 score comes from the row-level split. On the grouped split, collapse/flood F1 is
   0.82–0.87 and other-hazard F1 is 0.75–0.76.

A third issue makes both worse. The scene label and the boxes come from independent
models and are never compared, so they can silently contradict each other frame by frame.

**Success means:**

* no swapped-class boxes;
* the detector is the user's best checkpoint and runs with the preprocessing it was trained with;
* measured per-class F1 on the grouped test split is higher than the current single-model baseline, especially for collapse/flood;
* stable labels, with a visible warning whenever scene and boxes disagree;
* the live app still runs at ≥ 8 analyses/s on the GTX 960.

### User decisions (2026-09-25)

| Question | Decision |
|---|---|
| Detector | the user's best: `adaptfuse_od/outputs/accuracy_study/accuracy_yolo26s_clahe_640/weights/best.pt` |
| Scene classifier | ensemble of grouped AdapFuse-V1 seeds 41 + 42 |
| Scene vs. box disagreement | show both, add a "models disagree" warning; nothing is overridden |
| Temporal smoothing | yes, ~1 s window |
| Retraining | none; inference-side improvements only |

### Assumptions (not stated by the user)

* The two grouped seeds share one split. Verified: both runs used
  `data/metadata_grouped/seed_42` with the same aggregate SHA-256 `5d2cf592…`. So an
  ensemble can be scored fairly on that split's test set.
* Evaluation uses the app's real input condition: **one** visual modality plus audio when
  present. It does not use the tri-modal rows as the training script does.
* Person boxes (COCO YOLO26n) and sound tags (PANNs Cnn6) are unchanged.

## 2. Detection

**Model.** YOLO26s fine-tuned on CLAHE-enhanced D-Fire. Test mAP@50 = 0.771,
mAP@50:95 = 0.448, AP@50 smoke 0.849 and fire 0.694 (`training_summary.json`). Measured at
19 ms per frame on the GTX 960 at 640 px.

**Preprocessing, matching `adaptfuse_od/data/prepare_dfire_clahe.py`:**
BGR → LAB, CLAHE on L with `clipLimit=2.0`, `tileGridSize=(8, 8)`, merge, LAB → BGR. Only the
detector input is enhanced. The classifier and the displayed frame use the original frame.
Inference size is 640 (`imgsz` from `args.yaml`), and the confidence threshold defaults to 0.25.

**Class-name guard.** `load_models()` asserts that the detector's `names` equal
`{0: "smoke", 1: "fire"}` and that the person detector's `names[0] == "person"`. If not,
it raises with a message naming the checkpoint. Class labels are always read from
`model.names`, never from a hard-coded list.

**Bundle.** Add `weights/dfire_yolo26s_clahe.pt` (copied as is). Remove
`weights/dfire_yolo26n.pt`.

**Verification.**

* (a) On 20 raw D-Fire test images, the app's CLAHE output matches the stored
  `dfire_clahe` images to within JPEG noise (mean absolute difference < 2 grey levels).
* (b) `ultralytics val` on the `dfire_clahe` test split reproduces mAP@50 0.771 ± 0.005.

## 3. Scene classification

**Ensemble.** Load `outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth`
and `seed_42/.../adaptfuse_v1_grouped_best.pth`. Average the **softmax probabilities** of
the disaster, victim and nuisance heads. Average the RUE reliability and uncertainty.
Replace the row-level `weights/adaptfuse_v1.pth`.

**Flip TTA.** Each model receives `[x, hflip(x)]` as one batch of 2 (audio duplicated) and
the probabilities are averaged. Both models run within one lock acquisition.

**Class-bias calibration.** Store a 4-vector `b` of logit offsets for the disaster head in
`weights/calibration.json`, and apply it as `softmax(log p + b)`. Tune `b` by coordinate
search over `[-2, 2]` in steps of 0.1, maximizing disaster macro-F1 on the grouped **val**
split. Store it per input condition (`rgb`, `thermal`) because the two conditions shift
the class balance differently. The test split is used only for reporting.

**Storage.** Slim checkpoints hold `model_state` in fp16 and are cast to fp32 on load
(~26 MB each). Acceptance: identical argmax on ≥ 99.9% of grouped-test rows, and the
maximum absolute probability difference is reported.

**Evaluation script, `demo_app/scripts/eval_classifier.py`.**

* Reads `data/metadata_grouped/seed_42/{val,test}.csv`.
* Uses the app's preprocessing: `engine.pipeline._rgb_tensor`, `_thermal_tensor` and `_logmel`.
* Covers two conditions:
  * **rgb:** rows with an RGB image, thermal off;
  * **thermal:** rows with a thermal image, RGB off.
  * Audio is on only where the row has audio.
* Stages: `seed42` → `seed41` → `ensemble` → `ensemble+tta` → `ensemble+tta+bias`.
* Writes a per-class precision / recall / F1 table and macro-F1 per stage and condition
  to `demo_app/eval/classifier_eval.md` and `.json`.
* **Adoption rule:** a stage is kept in the app only if it does not lower test disaster
  macro-F1 in either condition. Otherwise the app uses the best earlier stage, and the
  report says so.

**Runtime budget.** Two models × batch of 2 must fit the ≥ 8 analyses/s target together with
the detectors. If it doesn't, drop TTA first and record that in the report.

## 4. Temporal smoothing and disagreement

New module `engine/temporal.py`, owned by `Analyzer`. It is reset by `Analyzer.reset()`,
and automatically when `t` jumps backwards or forwards by more than 2 s.

**Probability smoothing.** For each head, a time-based EMA:
$\bar p_t = \bar p_{t'} + (1 - e^{-\Delta t/\tau})(p_t - \bar p_{t'})$ with $\tau = 0.5$ s.
Displayed labels and confidences come from $\bar p$. Raw values stay in the result under
`raw`.

**Box persistence.**

* Keep the detections from the last 3 analyzed frames.
* A current box is *confirmed* if a box of the same class with IoU ≥ 0.3 appears in at
  least 2 of those 3 frames (the current one included).
* Only confirmed boxes are drawn.
* The displayed confidence is the mean over the matched frames.
* Person boxes follow the same rule.
* Unconfirmed boxes are returned under `raw_boxes` for the export.

**Disagreement flags,** computed on smoothed and confirmed outputs:

* `detector_only`: a confirmed fire or smoke box with confidence ≥ 0.5, while the smoothed
  scene is not `fire / smoke`.
* `classifier_only`: the smoothed scene has been `fire / smoke` for ≥ 2 s of video time
  with no confirmed fire or smoke box in that time.

**Result dict additions:** `agreement: {"status": "agree" | "detector_only" |
"classifier_only", "since": t}`, plus `raw` and `raw_boxes`.

## 5. UI and exports

**Live app**

* An amber "models disagree" badge on the video, with a short reason, e.g. "detector: smoke 0.71 · scene: normal".
* An event-log entry whenever the agreement status changes.
* The provenance note now names the YOLO26s CLAHE detector and the grouped ensemble.

**Gradio app**

* The burned-in panel shows the badge.
* CSV gains `agreement`, `raw_disaster` and `n_raw_boxes` columns.
* The summary reports the % of frames in disagreement.

**Docs.** Update `demo_app/README.md`, the root `README.md` provenance table, and the
`MODEL_NOTE` in Gradio.

## 6. Error handling

* Missing weights, a class-name mismatch, or a missing `calibration.json` each give a
  clear startup error naming the file. The exception: a missing calibration falls back to
  zero bias with a printed warning.
* The ensemble tolerates exactly one missing seed checkpoint by running with the other and
  printing a warning. With none present, startup fails.

## 7. Testing

`demo_app/tests/` (pytest, CPU-only, no dataset needed except where marked):

* `test_clahe.py`: CLAHE matches the reference implementation on a synthetic image.
  *(dataset)* matches the stored `dfire_clahe` files.
* `test_temporal.py`: EMA converges and resets on a time jump; box persistence (1 of 3
  hidden, 2 of 3 shown, class must match, IoU threshold); both disagreement flags incl.
  the 2 s rule.
* `test_models_guard.py`: the name guard rejects a fake detector with swapped names.
* `test_pipeline_smoke.py`: `Analyzer.analyze` on a synthetic frame returns every
  documented key, with probabilities summing to 1.
* Acceptance, run by hand:
  * the eval report shows the before/after table;
  * both sample clips go through both apps, and the smoke plume at ~7 s is labelled `smoke`;
  * the measured live analysis rate is ≥ 8/s.

## 8. Out of scope

* Retraining anything, including fine-tuning the person detector on SARD.
* Fixing the PANNs backbone bug in the research code (separate task).
* Real radiometric thermal support.
* Pushing the repository to GitHub.
