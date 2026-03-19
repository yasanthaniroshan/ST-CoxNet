# Experiment v14 — CPC + Within-Patient Temporal Ranking

## Goal

Fix v13's cross-patient shortcut problem by restricting temporal ranking loss to operate **within patients only**, using patient-balanced batching.

## Changes from v13

1. **Within-patient ranking loss** — `TemporalRankingLoss._within_patient()` computes ranking comparisons only within segments belonging to the same patient.
2. **Patient-balanced batching** — `PatientBatchSampler` yields batches of P=16 patients x K=32 segments, enabling vectorized within-patient computation via reshape(P, K, ...).
3. **New dataset** — `CreateTemporalDataset.py` with stride=100 (reduced from 20), patient IDs saved per segment, TTE in raw seconds.
4. **Cox early stopping** on validation C-index (patience=10).
5. **Within-SR Spearman metric** — the key evaluation metric for temporal signal quality.

## Architecture

Same as v13 (CPC encoder + temporal projection + Cox head), but with within-patient temporal loss.

## Script

`cpc_train_v14.py`

## Results

| Metric | Value |
|--------|-------|
| CPC epochs | 50 |
| Final SR Spearman ρ | ~0.47 |
| Final sim_gap | ~0.179 |
| CPC accuracy | ~14.7% |
| **Cox C-index** | **0.553** |
| Cox early stopped at | Epoch 12 |

### Phase 1 (CPC + Temporal) progression:
- SR-ρ improved from 0.28 (epoch 1) to ~0.47 (epoch 50)
- sim_gap steadily increased from 0.08 to 0.18
- Temporal loss decreased from 3.30 to 3.03

### Phase 2 (Cox):
- Best C-index: 0.553 at epoch 2
- Overfitting: val loss increased while train loss decreased
- Early stopped at epoch 12

## Conclusion

Within-patient ranking improved SR-ρ from ~0 (v13 baseline) to ~0.47, confirming that restricting comparisons within patients prevents identity shortcuts. However, the Cox C-index only marginally improved (0.553 vs ~0.55 baseline), suggesting the representations still lack sufficient discriminative power for survival prediction across patients.

## Key Insight

The encoder learns some temporal structure within patients (ρ=0.47), but this doesn't translate well into cross-patient survival prediction through a Cox head. The question becomes: is the temporal signal actually present in RR intervals, or is the architecture/loss insufficient?
