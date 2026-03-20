# v24 — GRL + Cross-Patient Contrastive Learning

## Motivation
Previous experiments showed that:
- **GRL alone (v21a):** adversarial patient-ID removal helped reduce centroid clustering
  but did not improve TTE generalization.
- **Cross-patient contrastive alone** was not tried in isolation.
- **Combining both** could be complementary: GRL removes identity information
  while contrastive learning actively organises the embedding space so that
  segments from different patients with similar TTE are pulled together.

## Method
Three-loss training on excellent patients with patient-level split:

| Loss | Weight | Purpose |
|------|--------|---------|
| `SmoothL1Loss(TTE_pred, TTE)` | 1.0 (fixed) | TTE regression |
| `CrossEntropyLoss(pat_logits, cls)` through GRL | `λ_adv` (ramp 0→1 over 30 ep) | Adversarial patient invariance |
| Cross-patient contrastive | `λ_con = 0.5` | Pull cross-patient similar-TTE embeddings together |

### Cross-patient contrastive loss
- Projection head maps GRU hidden state `h` → 64-d, L2-normalised.
- Positive weights: `w_ij = exp(-|TTE_i - TTE_j|² / (2σ²))` for **different** patients.
- Temperature τ = 0.1, σ_TTE = 10 min.
- InfoNCE-style: `L = -Σ_j w_ij log(softmax(sim/τ)) / Σ_j w_ij`

## Architecture
```
Encoder → GRU → h ─┬→ TTE head       → scalar TTE (minutes)
                    ├→ GRL → PatID    → N-class patient logits
                    └→ Proj head      → 64-d L2-norm (contrastive)
```

## Configuration
- **Patients:** excellent only, patient-level 85/15 split
- **RR preprocessing:** relative RR (per-patient z-score)
- **Batching:** PatientBatchSampler (P=8, K=10, 80 batches/epoch)
- **Epochs:** 200 max, patience=25
- **Optimizer:** AdamW (lr=1e-3, wd=1e-4) + CosineAnnealingLR

## Script
`cpc_train_v24_grl_cross_patient_contrastive.py`

## Metrics (fill after run)
- Train patients / Val patients: `` / ``
- Best val MAE: ``
- Final global Spearman ρ: ``
- Final per-patient ρ (mean): ``
- Final R²: ``
- Centroid seen accuracy: ``
- Centroid unseen AUROC: ``

## Interpretation
(fill after run)
