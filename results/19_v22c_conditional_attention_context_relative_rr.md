# v22c — Conditional attention context (relative RR) + patient-level split

## Setup
- Pool: `tier == "excellent"` from `v15c_patient_quality.json`
- Split: patient-level `train:val = 0.85:0.15` (no patient leakage)
- Conditioning: attention-weighted context from RR **support** segments of the same patient
- RR preprocessing: `relative_rr=True` (subtract per-patient RR baseline)
- Episode parameters: `support_k=5`, `query_k=3`, `batch_patients=8`, `episodes_per_epoch=150`

## Metrics (after run)
- Best val MAE:
- Best val Spearman rho:
- Best val R2:
- Centroid diagnostic on embeddings `h`:
  - centroid_seen_acc:
  - centroid_unseen_auroc:

## Run results (from terminal)
- Excellent patients: `56`
- Patient split: `47 train / 9 val` (patient-level)
- RR preprocessing: `relative_rr=True`
- Early stopping: epoch `24` (best val MAE)
- Best validation metrics:
  - Val MAE = `17.47` min
  - Val Spearman rho = `0.1264`
  - Val R2 = `-0.2751` (best checkpoint; printed at end)
- Centroid probe on embeddings `h`:
  - centroid_seen_acc = `0.0659`
  - centroid_unseen_auroc = `0.4951`

