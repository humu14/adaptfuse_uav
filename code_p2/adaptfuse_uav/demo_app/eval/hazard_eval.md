# Zero-shot hazard model and fused scene (grouped split, seed_42)

Stratified subsets, at most 150 images per (dataset, class): val n = 1597, test n = 1597, real-world clip frames n = 72.
Event bias and fusion weight are tuned on **val**; test and real-clip numbers are reported only.
Coarse = the 4 scene classes (normal, fire / smoke, collapse / flood, other disaster); fine = normal / fire / collapsed building / flooded area / traffic incident.
Fused = p ∝ p_AdapFuse^(w/T_a) · coarse(p_zero-shot)^((1-w)/T_z), temperatures fit by val NLL.

## Condition: rgb

| Model | split | coarse macro-F1 | coarse acc | fine macro-F1 |
|---|---|---|---|---|
| AdapFuse ensemble (app before) | val | 0.915 | 0.919 | — |
| zero-shot, no bias | val | 0.789 | 0.789 | 0.804 |
| zero-shot + val bias | val | 0.869 | 0.865 | 0.887 |
| fused w=0.35 (app now) | val | 0.958 | 0.961 | — |
| AdapFuse ensemble (app before) | test | 0.901 | 0.900 | — |
| zero-shot, no bias | test | 0.836 | 0.843 | 0.844 |
| zero-shot + val bias | test | 0.895 | 0.900 | 0.904 |
| fused w=0.35 (app now) | test | 0.967 | 0.967 | — |
| AdapFuse ensemble (app before) | real | 0.119 | 0.208 | — |
| zero-shot, no bias | real | 0.484 | 0.972 | 0.399 |
| zero-shot + val bias | real | 0.484 | 0.972 | 0.419 |
| fused w=0.35 (app now) | real | 0.475 | 0.944 | — |

Test F1 per fine class (zero-shot + val bias): normal 0.847, fire 0.915, collapsed_building 0.904, flooded_areas 0.975, traffic_incident 0.882

Real-world clips (frame accuracy on the coarse class):

| clip | truth | AdapFuse | fused | zero-shot events |
|---|---|---|---|---|
| fb54edf3b153 | fire | 0.00 | 1.00 | explosion×8 |
| 42928c7787e9 | fire | 0.00 | 1.00 | fire×7, smoke×1 |
| bd9e04a162fa | fire | 0.88 | 1.00 | smoke×8 |
| d3dd366ac602 | fire | 0.00 | 1.00 | smoke×8 |
| 69e9a09c9ce2 | fire | 0.88 | 1.00 | fire×8 |
| f04359e13141 | fire | 0.12 | 1.00 | fire×8 |
| f829f6eef594 | flooded_areas | 0.00 | 0.88 | flood×2, landslide×6 |
| 02fa4278b67d | flooded_areas | 0.00 | 0.62 | collapsed building×3, flood×3, smoke×2 |
| 0cab2631ba44 | collapsed_building | 0.00 | 1.00 | landslide×7, collapsed building×1 |

Fusion weight curve (coarse macro-F1):

| w (AdapFuse) | val | test | real |
|---|---|---|---|
| 0.00 | 0.869 | 0.895 | 0.484 |
| 0.05 | 0.879 | 0.904 | 0.484 |
| 0.10 | 0.890 | 0.913 | 0.484 |
| 0.15 | 0.900 | 0.926 | 0.484 |
| 0.20 | 0.911 | 0.933 | 0.484 |
| 0.25 | 0.929 | 0.947 | 0.487 |
| 0.30 | 0.944 | 0.959 | 0.489 |
| 0.35 ← | 0.958 | 0.967 | 0.475 |
| 0.40 | 0.964 | 0.969 | 0.440 |
| 0.45 | 0.967 | 0.973 | 0.405 |
| 0.50 | 0.966 | 0.973 | 0.372 |
| 0.55 | 0.967 | 0.970 | 0.372 |
| 0.60 | 0.962 | 0.966 | 0.323 |
| 0.65 | 0.956 | 0.960 | 0.242 |
| 0.70 | 0.949 | 0.955 | 0.201 |
| 0.75 | 0.945 | 0.948 | 0.178 |
| 0.80 | 0.936 | 0.948 | 0.164 |
| 0.85 | 0.935 | 0.940 | 0.155 |
| 0.90 | 0.931 | 0.932 | 0.157 |
| 0.95 | 0.922 | 0.926 | 0.157 |
| 1.00 | 0.915 | 0.901 | 0.119 |

## Condition: thermal

| Model | split | coarse macro-F1 | coarse acc | fine macro-F1 |
|---|---|---|---|---|
| AdapFuse ensemble (app before) | val | 0.292 | 0.432 | — |
| zero-shot, no bias | val | 0.725 | 0.718 | 0.734 |
| zero-shot + val bias | val | 0.825 | 0.816 | 0.847 |
| fused w=0.40 (app now) | val | 0.855 | 0.851 | — |
| AdapFuse ensemble (app before) | test | 0.298 | 0.439 | — |
| zero-shot, no bias | test | 0.730 | 0.727 | 0.746 |
| zero-shot + val bias | test | 0.825 | 0.817 | 0.844 |
| fused w=0.40 (app now) | test | 0.852 | 0.843 | — |
| AdapFuse ensemble (app before) | real | 0.115 | 0.208 | — |
| zero-shot, no bias | real | 0.469 | 0.944 | 0.320 |
| zero-shot + val bias | real | 0.421 | 0.875 | 0.338 |
| fused w=0.40 (app now) | real | 0.422 | 0.861 | — |

Test F1 per fine class (zero-shot + val bias): normal 0.748, fire 0.764, collapsed_building 0.870, flooded_areas 0.955, traffic_incident 0.882

Real-world clips (frame accuracy on the coarse class):

| clip | truth | AdapFuse | fused | zero-shot events |
|---|---|---|---|---|
| fb54edf3b153 | fire | 0.00 | 1.00 | explosion×8 |
| 42928c7787e9 | fire | 0.00 | 1.00 | fire×8 |
| bd9e04a162fa | fire | 1.00 | 1.00 | smoke×8 |
| d3dd366ac602 | fire | 0.00 | 1.00 | smoke×8 |
| 69e9a09c9ce2 | fire | 0.88 | 1.00 | explosion×6, fire×2 |
| f04359e13141 | fire | 0.00 | 1.00 | fire×8 |
| f829f6eef594 | flooded_areas | 0.00 | 0.88 | smoke×1, flood×1, landslide×6 |
| 02fa4278b67d | flooded_areas | 0.00 | 0.00 | explosion×6, smoke×2 |
| 0cab2631ba44 | collapsed_building | 0.00 | 0.88 | landslide×6, smoke×1, collapsed building×1 |

Fusion weight curve (coarse macro-F1):

| w (AdapFuse) | val | test | real |
|---|---|---|---|
| 0.00 | 0.825 | 0.825 | 0.421 |
| 0.05 | 0.828 | 0.827 | 0.421 |
| 0.10 | 0.834 | 0.832 | 0.421 |
| 0.15 | 0.834 | 0.835 | 0.421 |
| 0.20 | 0.837 | 0.841 | 0.421 |
| 0.25 | 0.840 | 0.843 | 0.421 |
| 0.30 | 0.842 | 0.850 | 0.425 |
| 0.35 | 0.847 | 0.852 | 0.425 |
| 0.40 ← | 0.855 | 0.852 | 0.422 |
| 0.45 | 0.863 | 0.852 | 0.422 |
| 0.50 | 0.861 | 0.853 | 0.424 |
| 0.55 | 0.855 | 0.854 | 0.397 |
| 0.60 | 0.845 | 0.845 | 0.353 |
| 0.65 | 0.839 | 0.837 | 0.312 |
| 0.70 | 0.821 | 0.825 | 0.196 |
| 0.75 | 0.784 | 0.799 | 0.114 |
| 0.80 | 0.705 | 0.748 | 0.114 |
| 0.85 | 0.603 | 0.606 | 0.114 |
| 0.90 | 0.428 | 0.402 | 0.114 |
| 0.95 | 0.312 | 0.311 | 0.114 |
| 1.00 | 0.292 | 0.298 | 0.115 |

## Calibration written to `weights/hazard_calibration.json`

```json
{
 "bias": {
  "rgb": [
   0.0,
   -1.0,
   1.75,
   -1.5,
   -1.75,
   -2.0,
   -3.0,
   -3.0
  ],
  "thermal": [
   0.0,
   0.75,
   1.5,
   0.0,
   -3.0,
   -2.25,
   -3.0,
   -2.75
  ]
 },
 "fusion_weight": {
  "rgb": 0.35,
  "thermal": 0.4
 },
 "temperature": {
  "rgb": {
   "adaptfuse": 0.85,
   "zero_shot": 1.2
  },
  "thermal": {
   "adaptfuse": 1.5,
   "zero_shot": 1.5
  }
 },
 "zone": {
  "gate": 0.25,
  "thr": 0.5,
  "min_frac": 0.02
 },
 "tuned_on": "data/metadata_grouped/seed_42/val.csv (stratified subset)"
}
```

Run time: 0.0 min.