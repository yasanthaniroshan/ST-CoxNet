# v21b — GRL adversarial + Raw RR (excellent-only)

## Patient selection / split
- Patients: `tier == "excellent"` from `v15c_patient_quality.json` (10 total)
- Patient-level split: 8 train / 2 val (0.85 / 0.15)
- Segments: 458 train / 100 val

## Training outcome (best checkpoint)
- Val MAE (min): `19.1337` min
- Val Spearman rho: `-0.2260`
- Val R^2: `0.00040`
- Mean per-patient rho: `-0.5847`
- Early stop: epoch `31` (best MAE)
- lambda_adv (approx): `1.00`

## Embedding patient-invariance diagnostic (centroids)
- Seen closed-set accuracy: `0.3319`
- Unseen detection AUROC: `0.5651`

