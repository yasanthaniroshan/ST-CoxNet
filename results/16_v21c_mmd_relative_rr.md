# v21c — MMD regularization + Relative RR (excellent-only)

## Patient selection / split
- Patients: `tier == "excellent"` from `v15c_patient_quality.json` (10 total)
- Patient-level split: 8 train / 2 val (0.85 / 0.15)
- Segments: 458 train / 100 val

## Training outcome (best checkpoint)
- Val MAE (min): `19.1398` min
- Val Spearman rho: `-0.2949`
- Val R^2: `-0.00031`
- Mean per-patient rho: `-0.0032`
- Early stop: epoch `39` (best MAE)
- lambda_mmd (max): `0.50`

## Embedding patient-invariance diagnostic (centroids)
- Seen closed-set accuracy: `0.1790`
- Unseen detection AUROC: `0.4789`

