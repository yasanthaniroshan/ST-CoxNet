# Summary — All Experiments So Far

This folder contains exported results for the experiments run so far in this repo.

## Experiment list

1. `results/experiment_01_raw_dataset_analysis.md`
2. `results/experiment_02_cpc_temporal_signal_verification.md`

## Cross-experiment conclusions

### 1) The dataset has measurable temporal structure (but it’s largely captured by HRV)
- From `Experiment 01`, within-patient temporal correlations exist (e.g., pNN50 shows the strongest magnitude correlation with time-to-event, even after patient-level handling).
- From `Experiment 02`, CPC encoder/context representations trained with the currently used checkpoint do **not** show stronger temporal monotonic signal than the best HRV statistic (`pNN50`).

### 2) Patient dependency is present but not dominating HRV features
- `Experiment 01` reports mean ICC around **0.274** across HRV features, indicating moderate patient dependency.
- This makes patient-level splitting important, but the dominant temporal signal is not purely patient identity.

### 3) Next step implications for CPC
- Because CPC does not beat HRV on temporal monotonicity here, likely next actions include:
  - retrain / finetune CPC with the *more modern* encoder architecture used elsewhere in the codebase
  - increase training/adjust contrastive objective details
  - evaluate CPC representations across multiple checkpoints (not just `cpc_model.pth`)

## Where to find plots and raw logs

- Dataset quality plots: `plots/dataset_analysis/`
- CPC verification plots: `plots/cpc_temporal_signal/`

