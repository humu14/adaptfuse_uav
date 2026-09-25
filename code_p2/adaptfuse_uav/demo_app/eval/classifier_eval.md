# Scene classifier evaluation (grouped split, seed_42)

Input condition as in the demo app: **one** visual modality + audio where the row has it. Bias tuned on **val**; numbers below are **test**. The adoption rule compares stages on test macro-F1, so the chosen stage is selected on test.

## Condition: rgb  (n = 10697)

| Stage | Disaster macro-F1 | F1 normal | F1 fire/smoke | F1 collapse/flood | F1 other | Victim macro-F1 |
|---|---|---|---|---|---|---|
| seed41 | 0.8685 | 0.9161 | 0.9365 | 0.8719 | 0.7497 | 0.8850 |
| seed42 | 0.8898 | 0.9048 | 0.9247 | 0.8801 | 0.8498 | 0.9220 |
| ensemble | 0.9120 | 0.9286 | 0.9490 | 0.9141 | 0.8561 | 0.9277 |
| ensemble+tta | 0.9183 | 0.9344 | 0.9534 | 0.9227 | 0.8627 | 0.9285 |
| ensemble+tta+bias **(in app)** | 0.9193 | 0.9245 | 0.9403 | 0.9179 | 0.8946 | 0.9285 |

Per-class precision / recall for the app stage (`ensemble+tta+bias`):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.8742 | 0.9808 | 0.9245 | 4223 |
| fire / smoke | 0.9858 | 0.8988 | 0.9403 | 5100 |
| collapse / flood | 0.9614 | 0.8782 | 0.9179 | 936 |
| other disaster | 0.8789 | 0.9110 | 0.8946 | 438 |

## Condition: thermal  (n = 10697)

| Stage | Disaster macro-F1 | F1 normal | F1 fire/smoke | F1 collapse/flood | F1 other | Victim macro-F1 |
|---|---|---|---|---|---|---|
| seed41 | 0.3476 | 0.6709 | 0.7194 | 0.0000 | 0.0000 | 0.4382 |
| seed42 | 0.2636 | 0.5480 | 0.5064 | 0.0000 | 0.0000 | 0.4369 |
| ensemble | 0.2607 | 0.5996 | 0.4433 | 0.0000 | 0.0000 | 0.4369 |
| ensemble+tta | 0.2607 | 0.5996 | 0.4433 | 0.0000 | 0.0000 | 0.4369 |
| ensemble+tta+bias **(in app)** | 0.3552 | 0.6736 | 0.7471 | 0.0000 | 0.0000 | 0.4369 |

Per-class precision / recall for the app stage (`ensemble+tta+bias`):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.9995 | 0.5079 | 0.6736 | 4223 |
| fire / smoke | 0.5963 | 0.9998 | 0.7471 | 5100 |
| collapse / flood | 0.0000 | 0.0000 | 0.0000 | 936 |
| other disaster | 0.0000 | 0.0000 | 0.0000 | 438 |

## Calibration written to `weights/calibration.json`

```json
{
 "stage": "ensemble+tta+bias",
 "members": [
  "adaptfuse_v1_grouped_s41.pth",
  "adaptfuse_v1_grouped_s42.pth"
 ],
 "tta": true,
 "bias": {
  "rgb": [
   0.0,
   -0.1,
   0.3,
   1.1
  ],
  "thermal": [
   0.0,
   0.1,
   0.0,
   0.0
  ]
 },
 "tuned_on": "data/metadata_grouped/seed_42/val.csv",
 "selected_by": "adoption rule on data/metadata_grouped/seed_42/test.csv disaster macro-F1"
}
```

## fp16 vs fp32 check (test, rgb)

| Member | Argmax agreement | Max abs Δp | n (finite) | NaN rows fp16 / fp32 |
|---|---|---|---|---|
| adaptfuse_v1_grouped_s41.pth | 0.99944 | 0.43830 | 10696 | 0 / 1 |
| adaptfuse_v1_grouped_s42.pth | 0.99953 | 0.86646 | 10697 | 0 / 0 |

Run time: 20.2 min.