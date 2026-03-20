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
- Train patients / Val patients: ``
- Best val MAE (stage2): ``
- Final global MAE: ``
- Final Spearman rho: ``
- Final R2: ``
- Final per-patient rho mean: ``
- Centroid probe:
  - `seen_acc`: ``
  - `unseen_auroc`: ``

## Interpretation (fill after run)
``

