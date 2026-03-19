# Experiment 3: Time-to-Event Regression

## Objective

Directly predict the continuous time-to-AFib-onset from SR segment representations. This tests whether the CPC encoder or a dedicated encoder can capture temporal proximity information.

## Script

- `cpc_train_v7.py`

## Architecture

- **Encoder**: `SimpleTimeEncoder` -- Conv1D + GRU
- **Prediction Head**: `TimeToEventHead` -- MLP regression head
- **Loss**: Joint objective -- `time_contrastive_loss` (InfoNCE with temporal soft labels) + MSE regression loss

## Key Design

- Segments closer in time to AFib onset should have similar representations (contrastive)
- The regression head directly predicts the normalized time-to-event [0, 1]
- Warmup: contrastive-only for first N epochs, then add regression loss

## Results

| Metric | Train | Validation |
|--------|-------|------------|
| Contrastive Loss | Plateaued ~3.5 | ~3.5 |
| MAE | ~0.25 | ~0.28 |
| RMSE | ~0.30 | ~0.32 |

## Analysis

- MAE of ~0.25 on a [0,1] scale means predictions are off by ~25% of the total SR duration
- The model essentially learned to predict the mean time, showing very little discriminative ability
- Contrastive loss did not decrease meaningfully, indicating the encoder could not learn temporally-ordered representations

## Conclusion

Direct time-to-event regression from individual SR segments **failed**. The temporal proximity signal in short RR interval windows is too weak for a continuous regression target. This motivated the switch to a discretized classification approach.
