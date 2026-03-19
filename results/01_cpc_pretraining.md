# Experiment 1: CPC Self-Supervised Pretraining

## Objective

Train a Contrastive Predictive Coding (CPC) model to learn useful representations from RR interval segments via self-supervised learning (next-step prediction).

## Script Versions

- `cpc_train_v4.py` (initial), `cpc_train_v5.py`, `cpc_train_v6.py` (iterative fixes)
- `cpc_train_v8.py` (final successful version with cross-batch InfoNCE)

## Architecture

- **Encoder**: `BaseEncoder` -- 3-layer Conv1D (1→8→16→32 channels, stride=2) + Linear projection to 64d latent
- **AR Block**: 1-layer GRU (64d input → 128d context)
- **Prediction Heads**: Linear projections `Wk` from context to latent space for k=1..3 future steps
- **Loss**: Cross-batch InfoNCE (logits[i,j] = sim(pred_i, z_future_j) / temperature)

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| latent_dim | 64 |
| context_dim | 128 |
| prediction_steps | 3 |
| temperature | 0.07 |
| window_size | 100 RR intervals |
| windows_per_segment | 10 |
| batch_size | 512 |
| learning_rate | 1e-3 |
| optimizer | AdamW (weight_decay=0.01) |
| scheduler | CosineAnnealing |

## Dataset

- **Source**: IRIDIA AFib Dataset -- RR intervals before AFib episodes
- **SR length**: 1.5 hours before AFib onset
- **AFib length**: 1 hour
- **Segments**: Sliding window with stride=20, each segment = 10 windows × 100 RR intervals
- **Preprocessing**: RobustScaler on RR intervals (ms), times normalized to [0,1]
- **Train/Val split**: 85/15 at patient level (~107 train patients, ~21 val patients)

## Evolution of Approaches

### v4-v6: Within-Batch InfoNCE (Failed)

- Used within-batch contrastive loss where positives were the true future embeddings and negatives were other time steps within the same sample
- **Problem**: Loss plateaued, accuracy stayed near random (~50%)
- **Root cause**: Within-batch negatives were too easy (different time steps of same sample are already dissimilar)

### v7: Time-Contrastive Loss (Failed)

- Added auxiliary `time_infonce` loss using temporal proximity as soft labels
- Attempted tau parameter tuning, diagonal masking, k-NN positives
- **Problem**: Time loss did not decrease meaningfully

### v8: Cross-Batch InfoNCE (Successful)

- Changed to standard cross-batch InfoNCE: for each prediction, all other samples in the batch serve as negatives
- logits[i,j] = dot(pred_i, z_future_j) / temperature, labels = arange(B)
- **Result**: CPC loss decreased properly, accuracy improved to ~75-80%

## Results (v8 -- Final)

| Metric | Train | Validation |
|--------|-------|------------|
| CPC Loss | ~0.8 | ~1.2 |
| CPC Accuracy | ~80% | ~70% |

## Key Bug Fixes

1. **Dimension mismatch** (`BaseEncoder`): Hardcoded linear layer dimension was wrong → fixed with dynamic dummy forward pass
2. **Variable typo**: `cxt` vs `ctx` caused batch size mismatch during validation
3. **Time loss not backpropagated**: `time_loss` was computed but excluded from `loss.backward()`
4. **Within-batch vs cross-batch**: The critical fix -- switching to cross-batch InfoNCE

## Saved Artifacts

- `cpc_model_v8.pth` -- checkpoint used as the pretrained backbone for all downstream tasks
- `plots/cpc_v8_accuracy.png` -- training curves

## Conclusion

Cross-batch InfoNCE is essential for CPC to work on this dataset. The pretrained encoder learns meaningful representations that distinguish different RR interval patterns across the batch, achieving ~70% val accuracy on the next-step prediction task.
