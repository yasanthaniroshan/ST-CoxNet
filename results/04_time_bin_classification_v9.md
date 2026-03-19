# Experiment 4: Time-Bin Classification (v9)

## Objective

Instead of predicting continuous time-to-event, discretize into 3 bins (Near/Mid/Far) and treat it as a multi-class classification problem. This simplifies the target and may be more learnable.

## Script

- `cpc_train_v9.py`

## Architecture

- **Backbone**: Pretrained CPC encoder from v8
- **Classification Head**: `TimeBinHead` -- MLP on CPC context + latent features → 3 classes
- **Loss**: CrossEntropyLoss with class weights

## Dataset

- `CPCTimeBinDataset`: SR-only segments with normalized time-to-event binned into 3 classes
- Bin edges: [0, 1/3, 2/3, 1.01] of normalized time
  - **Near (0-33%)**: Closest to AFib onset
  - **Mid (33-67%)**: Middle range
  - **Far (67-100%)**: Furthest from AFib onset
- Random chance = 33.3%

## Training Protocol

1. Phase 1: Head-only (encoder frozen)
2. Phase 2: Fine-tune encoder with reduced LR

## Results

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | ~55% | ~37% |
| F1 Macro | ~55% | **37.8%** |

## Analysis

- Validation F1 of 37.8% is barely above random chance (33.3%)
- Large train-val gap indicates the model memorizes patient-specific patterns
- The CPC features alone (trained for next-step prediction) do not carry sufficient temporal proximity information

## Conclusion

Multi-class time-bin classification from CPC features alone shows only a marginal improvement over random. This raised the central hypothesis: **short SR segments may not carry strong temporal proximity information to AFib onset in their RR intervals alone**. This led to the diagnostic investigation (Experiments 5-6).
