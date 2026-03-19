# v21a — GRL adversarial + Relative RR (excellent-only)

## Patient selection / split
- Patients: `tier == "excellent"` from `v15c_patient_quality.json` (10 total)
- Patient-level split: 8 train / 2 val (0.85 / 0.15)
- Segments: 458 train / 100 val

## Training outcome (best checkpoint)
- Val MAE (min): `19.1357` min
- Val Spearman rho: `-0.3514`
- Val R^2: `-0.00016`
- Mean per-patient rho: `-0.0630`
- Early stop: epoch `30` (best MAE)
- Final lambda_adv (approx): `0.97`

## Embedding patient-invariance diagnostic (centroids)
- Seen closed-set accuracy: `0.1266`
- Unseen detection AUROC: `0.5207`

