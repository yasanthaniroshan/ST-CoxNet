# v21d — MMD regularization + Raw RR (excellent-only)

## Patient selection / split
- Patients: `tier == "excellent"` from `v15c_patient_quality.json` (10 total)
- Patient-level split: 8 train / 2 val (0.85 / 0.15)
- Segments: 458 train / 100 val

## Training outcome (best checkpoint)
- Val MAE (min): `19.1384` min
- Val Spearman rho: `0.1414`
- Val R^2: `-0.00004`
- Mean per-patient rho: `-0.1874`
- Early stop: epoch `39` (best MAE)
- lambda_mmd (max): `0.50`

## Embedding patient-invariance diagnostic (centroids)
- Seen closed-set accuracy: `0.1769`
- Unseen detection AUROC: `0.6244`

