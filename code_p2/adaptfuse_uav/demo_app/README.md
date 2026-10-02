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
`--host`, `--port` and `--device {cuda,cpu}`. Six sample clips in `samples/` (forest
fire, thermal, collapsed buildings, flood, traffic accident, city fire) can be opened with
one click in either app.

All weights ship in `weights/` (~420 MB, every file < 90 MB). No download is needed on
first run.

## What is shown

| Output | Model | Weights | Provenance |
|---|---|---|---|
| Disaster class (normal · fire/smoke · collapse/flood · other), victim yes/no, nuisance type | Ensemble of 2 × AdapFuse-V1 (grouped split, seeds 41 + 42) + flip TTA + class calibration — see [`eval/classifier_eval.md`](eval/classifier_eval.md) | `adaptfuse_v1_grouped_s41.pth`, `adaptfuse_v1_grouped_s42.pth`, `calibration.json` | trained in the AdapFuse-UAV pipeline |
| RUE reliability $r_m$ and uncertainty $u_m$ per modality | same network | same | trained in pipeline |
| Audio-only disaster / victim label | AudioCNN baseline (A3) | `baseline_audio.pth` | trained in pipeline |
| **Fire / smoke boxes** | YOLO26s fine-tuned on CLAHE-enhanced D-Fire (test mAP@50 = 0.771); frames are CLAHE-enhanced before detection | `dfire_yolo26s_clahe.pt` | trained in pipeline |
| **Person and vehicle boxes** (person, car, truck, bus, motorcycle, bicycle, boat) | YOLO26n | `yolo26n_coco.pt` | **COCO-pretrained, not fine-tuned** |
| **Hazard event** (normal, fire, smoke, explosion, collapsed building, flood, landslide, traffic accident) and **hazard zones** | CLIP ViT-B/16 image encoder, zero-shot against prompt-ensemble text embeddings; per-event bias tuned on the grouped val split; zones from ClearCLIP-style dense patch features | `hazard_clip/`, `hazard_text.pt`, `hazard_calibration.json` | **pretrained (OpenAI CLIP), not fine-tuned; no extra labels** |
| **Scene shown in the app** | weighted product of the AdapFuse ensemble and the hazard model (see below) — see [`eval/hazard_eval.md`](eval/hazard_eval.md) | both of the above | fusion of pipeline + pretrained |
| **Sound events** (fire crackling, explosion, siren, screaming, crash/collapse, rushing water, rain/thunder, helicopter, traffic, wind, speech, music) | PANNs Cnn14_16k, AudioSet classes pooled into groups, per-group thresholds tuned on ESC-50 — see [`eval/audio_eval.md`](eval/audio_eval.md) | `panns_cnn14_16k-*.safetensors`, `sound_thresholds.json` | **AudioSet-pretrained, not fine-tuned** |

The live page and the Gradio app do not print model names next to each output; this
table is the reference for which network produced what.

## How one RGB-or-thermal video is handled

```text
video file ─┬─ frames ──► modality check (auto: greyscale / false-colour palette / file name; manual override)
            │                 ├─ RGB     → RGB branch,     has_thermal = 0
            │                 └─ thermal → thermal branch, has_rgb     = 0
            │             ──► CLAHE → YOLO26s (D-Fire) fire/smoke  +  YOLO26n (COCO) person/vehicles
            │             ──► every 0.5 s: CLIP ViT-B/16 (aspect kept, short side 224)
            │                     ├─ pooled embedding → hazard event probabilities  ─┐
            │                     └─ patch embeddings → hazard zones (RGB only)       ├─► fused scene
            │             ──► AdapFuse-V1 ensemble (scene / victim / nuisance, RUE) ─┘
            └─ audio  ──► refreshed every 0.5 s, window ending at the frame time
                          ├─ 2 s, 16 kHz, 64-mel log-power → AdapFuse audio branch + AudioCNN
                          └─ 3 s, 16 kHz waveform          → PANNs Cnn14_16k → sound events
```

The missing visual modality is handled the way it was during training
(`missing_modality_prob = 0.2`): its feature is multiplied by the availability flag
$\delta_m \in \{0,1\}$ and its RUE reliability is forced to zero,

$$
\tilde{\mathbf f}_m = \delta_m\, r_m\, \mathbf f_m, \qquad
r_m = \delta_m\,\sigma\!\big(g([\mathbf q_m,\ \lVert\mathbf f_m\rVert,\ \mathrm{Var}(\mathbf f_m)]_{m})\big).
$$

A video without an audio track, or a silent 2 s window, sets $\delta_\text{audio}=0$.

### Scene fusion and hazard zones

AdapFuse-V1 was trained on aerial dataset images (FLAME, AIDER, C2A, SARD) and is very
confident but often wrong on other footage (news and phone video). The scene shown in the
app therefore combines it with the zero-shot hazard model as a weighted product of the two
temperature-scaled experts,

$$
p(c) \;\propto\; p_\text{AdapFuse}(c)^{\,w/T_a}\;\; p_\text{zero-shot}(c)^{\,(1-w)/T_z},
\qquad p_\text{zero-shot}(c) = \sum_{e \in c} p(e),
$$

where $e$ runs over the hazard events of scene class $c$. $T_a, T_z$ minimise the val
NLL; $w$ is the smallest weight whose val macro-F1 is within 0.01 of the best ($w = 0.35$
RGB, $0.40$ thermal). The event label is the most likely event inside the winning class.

Hazard zones use the same forward pass: the last ViT block is re-run ClearCLIP-style
($q q^\top$ attention, no residual, no MLP) to get one embedding per 16 px patch. A zone
is a connected region of patches whose (blurred) event probability exceeds 0.5, kept
only if the event has scene probability ≥ 0.25 and belongs to the reported scene class.
Zones are coarse (one patch ≈ 1/14 of the frame height) and are not drawn on thermal input.

Accuracy on stratified subsets of the grouped split (≤ 150 images per dataset × class;
bias, temperatures and $w$ tuned on val, numbers on test) and on 72 frames from 9
real-world news/phone clips (6 fire, 2 flood, 1 collapse; not used for tuning):

| Scene (4 classes) | test macro-F1, RGB | test macro-F1, thermal | real clips, frame acc. (RGB) |
|---|---|---|---|
| AdapFuse ensemble alone (previous app) | 0.901 | 0.298 | 0.21 |
| zero-shot hazard model alone | 0.895 | 0.825 | 0.97 |
| **fused (app)** | **0.967** | **0.852** | **0.94** |

Fine-grained hazard type (test, RGB, zero-shot): F1 normal 0.85, fire 0.92, collapsed
building 0.90, flooded area 0.98, traffic incident 0.88. Details:
[`eval/hazard_eval.md`](eval/hazard_eval.md).

Sound events on ESC-50 (classes mapped to the sound groups, thresholds tuned on folds
1–3, numbers on folds 4–5; [`eval/audio_eval.md`](eval/audio_eval.md)):

| Tagger | window | mean group F1 | top-group accuracy | hazard false alarms on unrelated sounds |
|---|---|---|---|---|
| PANNs Cnn6 (previous app) | 2 s | 0.557 | 0.602 | 3.7 % |
| **PANNs Cnn14_16k (app)** | **3 s** | **0.703** | **0.727** | **3.7 %** |

### Stability and disagreement

* **Smoothing:** class probabilities are averaged over time,
  $\bar p_t = \bar p_{t'} + (1-e^{-\Delta t/\tau})(p_t-\bar p_{t'})$ with $\tau = 0.5$ s,
  and reset on seeking.
* **Box persistence:** a box is drawn only if the same class (IoU ≥ 0.3) is detected in
  2 of the last 3 analyzed frames.
* **"Models disagree"** appears when a persistent fire/smoke box with confidence ≥ 0.5
  contradicts a non-fire scene, or when the scene has said fire/smoke for ≥ 2 s without
  any fire/smoke box. Both outputs stay visible and nothing is overridden.
  In a collapse / flood scene a smoke-only box does not count (dust and spray look like
  smoke to the detector); fire boxes still do.
* The detector refuses to load a checkpoint whose class names are not
  `{0: smoke, 1: fire}`. An earlier bundled checkpoint had them swapped.
* **Sound events** are the maximum AudioSet probability within each group, smoothed
  with $\tau = 1$ s. A group is *active* above its threshold (tick mark on the bar). The
  headline is the strongest active hazard group, else the strongest active background
  group, else "background noise".

### Live app timing

The browser plays the file natively, with sound. While it plays, it copies the frame
on screen to a canvas, downsizes it to 640 px wide, and sends it as JPEG with its
`currentTime`. The server cuts the audio window ending at that same time. The next
frame is sent only after the previous result arrives, so the analysis rate adapts to
the GPU and never builds up a backlog. The *Lag* readout shows how far the overlay is
behind the playhead.

Measured on the development machine (GeForce GTX 960, 4 GB), 640×360 input, after warm-up:

| | per frame | analysis rate |
|---|---|---|
| CUDA, engine loop (incl. audio + hazard refresh) | ≈ 93 ms mean, 74 ms median | ≈ 10.7 analyses/s |
| CPU, engine loop | ≈ 830 ms mean | ≈ 1.2 analyses/s |

Every frame runs YOLO26s (CLAHE) + YOLO26n (COCO) and 2 AdapFuse-V1 members × 2 views
(flip TTA) in one batch (≈ 70 ms). Every 0.5 s of video the CLIP encoder (≈ 65 ms) and,
offset by 0.25 s, the audio networks (≈ 35 ms) run once more, so the extra cost is spread
over two frames. The live page's rate is lower than the engine loop because of frame
capture, JPEG and WebSocket round trips (≈ 5 analyses/s measured in the desktop app's
browser pane with two other model servers sharing the GPU). On GPUs without tensor cores
(compute capability < 7) the CLIP encoder runs in fp32, which is faster there than fp16.
Pausing the live player shows that frame's raw detections, because box persistence only
applies during playback.

## Known limitations

* **Thermal input.** Training used *pseudo-thermal* images derived from RGB, and the
  D-Fire detector is RGB-only. AdapFuse alone reaches scene macro-F1 0.30 on thermal;
  the fused scene reaches 0.85 (test subset, see [`eval/hazard_eval.md`](eval/hazard_eval.md)).
  Hazard zones are not drawn on thermal input. Treat thermal fire/smoke boxes as
  indicative only.
* **Hazard model.** It is zero-shot: its accuracy depends on the prompt wording in
  `engine/labels.py` (HAZARD_EVENTS). It confuses landslide with muddy floods and with
  dense forest seen from above; both landslide and flood map to the same scene class.
  The real-clip check is small (9 clips, mostly fire) and is not a benchmark.
* **Person boxes** come from a COCO model trained on mostly ground-level photos, so
  small or distant people in aerial views are often missed. No person mAP is
  reported in the thesis.
* **Audio.** The AdapFuse audio branch was trained on ESC-50 clips paired at random
  with images, and RUE usually gives audio a reliability close to 0. The sound events
  come from the AudioSet tagger instead; AudioSet has no "building collapse" class, so
  that group is approximated by crash / breaking / crushing / rumble sounds.
  A roaring wildfire is often heard as "rain / thunder" rather than "fire crackling".
* **GRU memory** is applied per frame (sequence length 1), as in training. Stability
  across frames comes from the EMA smoothing and box persistence described above.

## Layout

```text
demo_app/
├── engine/                 shared inference core (no web code)
│   ├── models.py           loads the networks; PANNs Cnn6 / Cnn14_16k ports
│   ├── hazard.py           zero-shot hazard events, zones, scene fusion
│   ├── pipeline.py         Analyzer.analyze(frame, t) → result dict
│   ├── media.py            ffmpeg audio extraction, H.264 transcoding
│   ├── modality.py         RGB vs thermal detection
│   ├── render.py           burns results into frames (Gradio output)
│   └── labels.py           class names, hazard prompts, sound groups, COCO classes
├── live/                   App 1: server.py + static/{index.html, app.js, style.css}
├── gradio_app.py           App 2
├── weights/                demo weights (model_state only) + AudioSet label list
├── samples/                six 14-20 s test clips (see make_sample_videos.py)
├── eval/                   classifier_eval.md, hazard_eval.md, audio_eval.md
├── tests/                  pytest suite: python -m pytest tests -q
└── scripts/
    ├── export_weights.py   rebuild weights/ from training checkpoints
    ├── check_detector.py   verify detector names, CLAHE parity, test mAP
    ├── eval_classifier.py  measure classifier stages, write calibration.json
    ├── export_hazard.py    build hazard_clip/ + hazard_text.pt from openai/clip-vit-base-patch16
    ├── eval_hazard.py      zero-shot + fusion accuracy, write hazard_calibration.json
    ├── export_audio.py     PANNs Cnn14_16k → fp16 shards
    ├── eval_audio.py       compare taggers on ESC-50, write sound_thresholds.json
    └── make_sample_videos.py  rebuild samples/ from FLAME, SARD, AIDER (test split), ESC-50
```

## Attribution

* PANNs Cnn6 and Cnn14_16k weights: Kong et al., *PANNs: Large-Scale Pretrained Audio Neural
  Networks for Audio Pattern Recognition*, IEEE/ACM TASLP 2020 (Zenodo record 3987831).
* CLIP ViT-B/16: Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision*, ICML 2021 (`openai/clip-vit-base-patch16`, MIT licence). Dense patch features
  follow ClearCLIP (Lan et al., ECCV 2024).
* AudioSet ontology labels (`weights/audioset_labels.csv`): Gemmeke et al., ICASSP 2017,
  CC BY 4.0.
* YOLO26n COCO weights: Ultralytics (AGPL-3.0).
* Sample clips are built from FLAME (Shamsoshoara et al., 2021), SARD, AIDER (Kyrkou &
  Theocharides, 2020; grouped *test* split images only) and ESC-50 (Piczak, 2015,
  CC BY-NC 3.0). Use them for research only.
