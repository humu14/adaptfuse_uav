# Revised object-detection implementation

The core study has three focused runs, not an eight-model sweep:

1. pretrained YOLOv10n historical baseline;
2. pretrained YOLO26n current baseline;
3. the same YOLO26n with the proposed Hazard Prior Gate (HPG).

All use the same D-Fire split, 512 px input, batch size 16, maximum 40 epochs,
two warm-up epochs, and patience-8 early stopping. The default study therefore
has 120 maximum epoch-runs instead of the previous 800. Two 15-epoch cue
ablations are available only when requested.

HPG is a detection-label-only residual input module. It
combines a fire red-excess cue, a smoke achromaticity cue, and a luminance-edge
cue. The learned gate operates at stride 4 and is bilinearly restored to input
resolution, limiting activation memory on the 4 GB training GPU. Ultralytics trains the
complete model end-to-end using the backbone's native detection objective; no
pseudo-target loss is used. YOLO26 uses its native end-to-end loss and does not
use the older DFL head.

Run a quick end-to-end verification:

```powershell
python training/run_efficient_study.py --smoke_test
```

Run the three core experiments:

```powershell
python training/run_efficient_study.py --skip_existing
```

Run the optional short ablations only after the core result is promising:

```powershell
python training/run_efficient_study.py --include_ablations --skip_existing
```

Compare completed runs using held-out test mAP@50:95 as the primary metric:

```powershell
python evaluation/compare_models.py --results_dir outputs/efficient_study `
  --output outputs/efficient_study/comparison.json
```
