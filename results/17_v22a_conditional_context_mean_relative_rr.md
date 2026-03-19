# v22a — Conditional context mean (RR support) + patient-level split

## Key idea
Instead of making embeddings patient-invariant, v22a **conditions** each query prediction on a patient context vector computed from RR **support** segments from the same patient.

Model:
- Encoder -> GRU -> segment embedding `h`
- Context `c = mean(h_support)`
- Query prediction uses `head([h_query ; c])`

## Setup
- Patient pool: `tier == "excellent"` from `v15c_patient_quality.json`
- Split: patient-level, `train:val = 0.85:0.15`
- RR preprocessing: `relative_rr = True`
- Episode parameters:
  - `support_k = 5` support segments per patient
  - `query_k = 3` query segments per patient
  - `batch_patients = 8`
  - `episodes_per_epoch = 150`

## Metrics (fill after run)
- Best val MAE:
- Best val Spearman rho:
- Best val R2:
- Centroid diagnostic on embeddings `h`:
  - seen_acc:
  - unseen_auroc:

## Run results (from terminal)
- Excellent patients: `56`
- Patient split: `47 train / 9 val` (patient-level)
- RR preprocessing: `relative_rr=True`
- Early stopping: epoch `23` (best val MAE)
- Best validation metrics:
  - Val MAE = `17.58` min
  - Val Spearman rho = `0.2505` (printed as `0.251` at end)
  - Val R2 = `0.0558` (printed as `0.056` at end)
- Centroid probe on embeddings `h`:
  - centroid_seen_acc = `0.0904`
  - centroid_unseen_auroc = `0.5204`

