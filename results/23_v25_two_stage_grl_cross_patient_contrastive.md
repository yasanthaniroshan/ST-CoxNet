# v25 — Two-stage GRL + Cross-patient Contrastive → TTE Regression

## Plan / Method
Stage 1:
- Train encoder+GRU with:
  - GRL adversarial patient-ID invariance (`lambda_adv` ramp)
  - Cross-patient contrastive loss (Gaussian soft supervision on TTE distance)
- No TTE regression loss in Stage 1.

Stage 2:
- Turn off GRL + contrastive terms.
- Fine-tune TTE regression (`SmoothL1Loss`) with smaller LR.
- Freeze projection + patient classifier heads for stability.

## Config (script)
- Excellent patients only
- Patient-level split (85/15 via `validation_split`)
- Relative RR normalization
- Stage 1: `stage1_epochs=80`, `stage1_lr=1e-3`
- Stage 2: `stage2_epochs=120`, `stage2_lr=3e-4`
- Batching: `PatientBatchSampler` with `P=8`, `K=10`, `batches_per_epoch=80`

## Metrics (fill after run)
- Train patients / Val patients: `47 / 9`
- Best val MAE (stage2): `19.08`
- Final global MAE: `19.08`
- Final Spearman rho: `0.1991`
- Final R2: `-0.0817`
- Final per-patient rho mean: `0.1578`
- Centroid probe:
  - `seen_acc`: `0.1319`
  - `unseen_auroc`: `0.4851`

## Interpretation (fill after run)
- Stage 1 contrastive/GRL invariance achieved strong adversarial failure (patient-ID head accuracy stays near ~0.01),
  but the validation contrastive loss barely moves (stays ~3.118), suggesting contrastive training is not effectively shaping the representation.
- Stage 2 regression improves Spearman to ~0.20, but R2 is negative and centroid unseen AUROC is ~0.49 (near chance),
  meaning the learned embedding still does not transfer well across held-out patients.

