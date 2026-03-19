# Experiment 10: Sequence-Level Classification -- Lightweight GRU (v11 revised)

## Objective

Address the severe overfitting of the Transformer by replacing it with a much smaller model (1-layer bidirectional GRU + temporal attention), adding heavy regularization (dropout 0.4, label smoothing 0.15, weight decay 0.05), and keeping the CPC backbone frozen throughout.

## Script

- `cpc_train_v11.py` (revised)

## Architecture

- **Per-segment features**: Same 207d as Transformer version
- **Input projection**: Linear(207 → 64) + LayerNorm + GELU + Dropout(0.4)
- **Temporal model**: 1-layer bidirectional GRU (64d → 128d) + Linear(128 → 64) + LayerNorm + Dropout(0.4)
- **Temporal attention**: Single-head attention using last position as query, all positions as keys/values
- **Classifier**: MLP(128 → 64 → 3) -- concatenation of last hidden state + attended context
- **Total parameters**: **88,643** (vs ~500K+ for Transformer)

## Regularization

| Technique | Value |
|-----------|-------|
| Dropout | 0.4 (all layers) |
| Label smoothing | 0.15 |
| Weight decay | 0.05 |
| Backbone | Frozen entirely (never unfreezing) |
| Gradient clipping | 1.0 |

## Training

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-3 |
| Max epochs | 120 |
| Patience | 25 epochs |
| Scheduler | CosineAnnealing |

## Results

| Epoch | Train Acc | Train F1 | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| E1 (best val) | 39.9% | 0.395 | 41.9% | **0.407** |
| E19 | 74.5% | 0.742 | 43.5% | 0.410 |
| E46 (early stop) | 90.9% | 0.909 | 37.6% | 0.352 |

**Best Val F1 Macro: 0.4097** (epoch ~19-21)

**wandb**: `CPC-SeqLevel-GRU-v11` / run `charmed-brook-1`

## Analysis

- Despite 6x fewer parameters (88K vs 500K+), 4x more dropout, and label smoothing, the model **still overfits** (91% train vs 38% val at convergence)
- Regularization slows overfitting but does not prevent it
- Best val F1 (0.410) is slightly worse than the Transformer (0.435)
- The smaller model has less capacity to learn the subtle temporal patterns

## Conclusion

Model architecture and regularization are **not the bottleneck**. Whether using a large Transformer or a tiny GRU, with or without heavy regularization, the validation ceiling is ~41-44%. The problem is fundamental: the within-patient temporal signal in short SR segments is too weak and inconsistent to support cross-patient generalization with 107 training patients.
