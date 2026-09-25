# AdapFuse-UAV demo app

Two local web interfaces that run the trained AdapFuse-UAV models on **one uploaded
video, either RGB or thermal**, together with the video's own audio track.

| App | Command | URL | What it does |
|---|---|---|---|
| **Live** (FastAPI + WebSocket) | `python live/server.py` | <http://127.0.0.1:8000> | Plays the video with sound and draws boxes and labels on it while it plays |
| **Batch** (Gradio) | `python gradio_app.py` | <http://127.0.0.1:7860> | Processes the whole video and returns an annotated MP4, a timeline plot, a summary and per-frame CSV/JSON |

Both apps use the same inference engine (`engine/`), so they give the same results for the same frame.

## Quick start

```bash
# from the repository root
cd code_p2/adaptfuse_uav/demo_app
pip install -r requirements.txt        # install PyTorch for your CUDA version first
python live/server.py                  # or: python gradio_app.py
```

On Windows, double-click `run_live.bat` or `run_gradio.bat`. Both apps accept
`--host`, `--port` and `--device {cuda,cpu}`. Two sample clips in `samples/` can be
opened with one click in either app.

All weights ship in `weights/` (~102 MB). No download is needed on first run.

## What is shown

| Output | Model | Weights | Provenance |
|---|---|---|---|
| Disaster class (normal · fire/smoke · collapse/flood · other), victim yes/no, nuisance type | Ensemble of 2 × AdapFuse-V1 (grouped split, seeds 41 + 42) + flip TTA + class calibration — see [`eval/classifier_eval.md`](eval/classifier_eval.md) | `adaptfuse_v1_grouped_s41.pth`, `adaptfuse_v1_grouped_s42.pth`, `calibration.json` | trained in the AdapFuse-UAV pipeline |
| RUE reliability $r_m$ and uncertainty $u_m$ per modality | same network | same | trained in pipeline |
| Audio-only disaster / victim label | AudioCNN baseline (A3) | `baseline_audio.pth` | trained in pipeline |
| **Fire / smoke boxes** | YOLO26s fine-tuned on CLAHE-enhanced D-Fire (test mAP@50 = 0.771); frames are CLAHE-enhanced before detection | `dfire_yolo26s_clahe.pt` | trained in pipeline |
| **Person boxes** | YOLO26n, `person` class | `yolo26n_coco.pt` | **COCO-pretrained, not fine-tuned** |
| **Sound tags** (siren, screaming, crackle, explosion, helicopter, …) | PANNs Cnn6 | `panns_cnn6.pth` | **AudioSet-pretrained, not fine-tuned** |

## How one RGB-or-thermal video is handled

```text
video file ─┬─ frames ──► modality check (auto: greyscale / false-colour palette / file name; manual override)
            │                 ├─ RGB     → RGB branch,     has_thermal = 0
            │                 └─ thermal → thermal branch, has_rgb     = 0
            │             ──► CLAHE → YOLO26s (D-Fire) fire/smoke  +  YOLO26n (COCO) person
            └─ audio  ──► 2 s window ending at the frame time, refreshed every 0.5 s
                          ├─ 16 kHz, 64-mel log-power → AdapFuse audio branch + AudioCNN
                          └─ 32 kHz waveform          → PANNs Cnn6 → AudioSet tags
```

The missing visual modality is handled the way it was during training
(`missing_modality_prob = 0.2`): its feature is multiplied by the availability flag
$\delta_m \in \{0,1\}$ and its RUE reliability is forced to zero,

$$
\tilde{\mathbf f}_m = \delta_m\, r_m\, \mathbf f_m, \qquad
r_m = \delta_m\,\sigma\!\big(g([\mathbf q_m,\ \lVert\mathbf f_m\rVert,\ \mathrm{Var}(\mathbf f_m)]_{m})\big).
$$

A video without an audio track, or a silent 2 s window, sets $\delta_\text{audio}=0$.

### Stability and disagreement

* **Smoothing:** class probabilities are averaged over time,
  $\bar p_t = \bar p_{t'} + (1-e^{-\Delta t/\tau})(p_t-\bar p_{t'})$ with $\tau = 0.5$ s,
  and reset on seeking.
* **Box persistence:** a box is drawn only if the same class (IoU ≥ 0.3) is detected in
  2 of the last 3 analyzed frames.
* **"Models disagree"** appears when a persistent fire/smoke box with confidence ≥ 0.5
  contradicts a non-fire scene, or when the scene has said fire/smoke for ≥ 2 s without
  any fire/smoke box. Both outputs stay visible and nothing is overridden.
* The detector refuses to load a checkpoint whose class names are not
  `{0: smoke, 1: fire}`. An earlier bundled checkpoint had them swapped.

### Live app timing

The browser plays the file natively, with sound. While it plays, it copies the frame
on screen to a canvas, downsizes it to 640 px wide, and sends it as JPEG with its
`currentTime`. The server cuts the audio window ending at that same time. The next
frame is sent only after the previous result arrives, so the analysis rate adapts to
the GPU and never builds up a backlog. The *Lag* readout shows how far the overlay is
behind the playhead.

Measured on the development machine (GeForce GTX 960, 4 GB), 640×360 input:

| | per frame | analysis rate |
|---|---|---|
| CUDA, live app | ≈ 70–80 ms | ≈ 12.5–12.8 analyses/s |
| CUDA, engine loop (incl. audio refresh) | ≈ 100 ms | ≈ 9.8 analyses/s |
| CPU, engine loop | ≈ 430 ms | ≈ 2.3 analyses/s |

Per frame this runs YOLO26s (CLAHE) + YOLO26n (COCO), and 2 AdapFuse-V1 members × 2
views (flip TTA) in one batch. Pausing the live player shows that frame's raw detections,
because box persistence only applies during playback.

## Known limitations

* **Thermal input.** Training used *pseudo-thermal* images derived from RGB, and the
  D-Fire detector is RGB-only. On the grouped test split, scene classification from
  thermal + audio alone reaches disaster macro-F1 **0.36** (collapse/flood and other are
  never predicted), against **0.92** for RGB + audio. See
  [`eval/classifier_eval.md`](eval/classifier_eval.md). The app shows a warning in
  thermal mode. Treat thermal labels and boxes as indicative only.
* **Person boxes** come from a COCO model trained on mostly ground-level photos, so
  small or distant people in aerial views are often missed. No person mAP is
  reported in the thesis.
* **Audio.** The AdapFuse audio branch was trained on ESC-50 clips paired at random
  with images, and RUE usually gives audio a reliability close to 0. The AudioSet
  tags are the easier audio signal to read.
* **GRU memory** is applied per frame (sequence length 1), as in training. Stability
  across frames comes from the EMA smoothing and box persistence described above.

## Layout

```text
demo_app/
├── engine/                 shared inference core (no web code)
│   ├── models.py           loads the 5 networks; PANNs Cnn6 port
│   ├── pipeline.py         Analyzer.analyze(frame, t) → result dict
│   ├── media.py            ffmpeg audio extraction, H.264 transcoding
│   ├── modality.py         RGB vs thermal detection
│   ├── render.py           burns results into frames (Gradio output)
│   └── labels.py           class names, AudioSet subset
├── live/                   App 1: server.py + static/{index.html, app.js, style.css}
├── gradio_app.py           App 2
├── weights/                demo weights (model_state only) + AudioSet label list
├── samples/                two 14 s test clips (RGB and pseudo-thermal)
├── eval/                   classifier_eval.md — stage-by-stage accuracy on the grouped split
├── tests/                  pytest suite: python -m pytest tests -q
└── scripts/
    ├── export_weights.py   rebuild weights/ from training checkpoints
    ├── check_detector.py   verify detector names, CLAHE parity, test mAP
    ├── eval_classifier.py  measure classifier stages, write calibration.json
    └── make_sample_videos.py  rebuild samples/ from FLAME, SARD, ESC-50
```

## Attribution

* PANNs Cnn6 weights: Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks
  for Audio Pattern Recognition*, IEEE/ACM TASLP 2020 (Zenodo record 3987831).
* AudioSet ontology labels (`weights/audioset_labels.csv`): Gemmeke et al., ICASSP 2017,
  CC BY 4.0.
* YOLO26n COCO weights: Ultralytics (AGPL-3.0).
* Sample clips are built from FLAME (Shamsoshoara et al., 2021), SARD and ESC-50
  (Piczak, 2015, CC BY-NC 3.0). Use them for research only.
