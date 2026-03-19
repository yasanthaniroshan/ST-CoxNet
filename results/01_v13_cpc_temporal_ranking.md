# Experiment v13 — CPC + Batch-Level Temporal Ranking

## Goal

Train an encoder using CPC (InfoNCE) combined with a ranking-based temporal contrastive loss that uses time-to-event (TTE) information, so the encoder captures temporal proximity to AF onset.

## Architecture

- **Encoder**: Multi-scale ResNet (kernel sizes 3, 5, 7) with SE attention block
- **Autoregressive**: GRU (context_dim=128)
- **CPC Loss**: InfoNCE contrastive prediction (6 prediction steps)
- **Temporal Loss**: `TemporalRankingLoss` — soft supervised contrastive loss using a Gaussian kernel on normalized TTE distances (batch-level, cross-patient comparisons)
- **Phase 2**: Cox survival head (MLP) trained on frozen encoder representations

## Config

| Parameter | Value |
|-----------|-------|
| CPC epochs | 50 |
| Cox epochs | 50 (early stopping, patience=10) |
| Latent dim | 64 |
| Context dim | 128 |
| Temporal loss weight | 1.0 |
| Temporal sigma | 0.15 |
| Temperature | 0.2 |
| Batch size | P=16 patients x K=32 segments |
| Stride | 100 (90% overlap) |

## Script

`cpc_train_v13.py`

## Issues Encountered

- **NaN in temporal loss**: Caused by `0 * (-inf) = NaN` in the log_softmax computation. Fixed by masking diagonal elements of `log_prob` to `0.0` before multiplication.

## Results

- **CPC Phase**: Temporal loss trained but didn't show strong generalization. The sim_gap (difference in similarity between TTE-close vs TTE-far pairs) was low.
- **Cox Phase**: C-index ~0.55, indicating near-random survival prediction. The Cox model overfitted quickly.
- **SR Spearman ρ**: Low (~0.47 at best) — weak temporal ordering in the learned representations.

## Conclusion

Batch-level temporal ranking across patients caused the model to learn **patient-identity shortcuts** rather than universal temporal signals. The high segment overlap (90%) and cross-patient comparisons made it easy for the model to distinguish patients rather than temporal positions within patients.

## Key Insight

The temporal ranking loss needs to operate **within patients**, not across patients, to prevent identity shortcuts.
