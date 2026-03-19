# Experiment 8: Multi-Modal Time-Bin Classification (v10)

## Objective

Combine CPC encoder features with hand-crafted HRV features in a multi-modal neural network for time-bin classification. Tests whether adding the discriminative HRV features that worked for RF can improve the NN.

## Script

- `cpc_train_v10.py`

## Architecture

- **Backbone**: Pretrained CPC encoder from v8 (frozen initially, then fine-tuned)
- **HRV Branch**: Linear(15 → 64) + LayerNorm + GELU + Dropout
- **Fusion**: Concatenate CPC context (128d) + CPC latent (64d) + HRV projection (64d) = 256d
- **Classifier**: MLP(256 → 128 → 64 → 3 classes)
- **Loss**: CrossEntropyLoss with class weights

## Two Variants Tested

### 8A: Global Normalization

HRV features standardized using global train mean/std (standard z-scoring across all segments).

| Metric | Train | Validation |
|--------|-------|------------|
| Best Val F1 Macro | - | **0.340** |
| Best Val Accuracy | - | 0.331 |
| Final Train Accuracy | 63% | - |

**wandb**: `CPC-MultiModal-TimeBin-v10` / run `helpful-capybara-2`

### 8B: Per-Patient Normalization

HRV features z-scored within each patient (removes inter-patient baseline variability).

| Metric | Train | Validation |
|--------|-------|------------|
| Best Val F1 Macro | - | **0.428** |
| Best Val Accuracy | - | 0.435 |
| Final Train Accuracy | 74.6% | - |

**wandb**: `CPC-MultiModal-PatientNorm-v10` / run `vocal-moon-1`

## Key Observations

- **Global norm (0.340)** performs at random chance -- confirms inter-patient variability dominates
- **Patient norm (0.428)** matches the RF patient-normalized result (0.425) -- the NN successfully learns the same weak signal
- Train-val gap is ~32pp in both cases, indicating overfitting to training patients

## Patient Statistics

- Train: 107 patients (26,964 SR segments)
- Validation: 21 patients (5,763 SR segments)

## Conclusion

Per-patient normalization is essential. It brings the NN from random chance (34%) to match the RF ceiling (42.8%). However, this ceiling is fundamentally limited by the weak within-patient temporal signal. Adding HRV features to the NN helps, but only when properly normalized per patient.
