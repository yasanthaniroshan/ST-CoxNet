# Experiment 6: Diagnostic Phase 2 -- Random Forest Baseline

## Objective

Establish an upper bound for time-bin classification using classical ML (Random Forest) on hand-crafted HRV features. If RF can't do it, a neural network is unlikely to succeed either.

## Script

- `debug_time_signal.py` (Phase 2 section)

## Method

- **Model**: RandomForestClassifier (n_estimators=500, max_depth=15)
- **Features**: 15 HRV features computed from scaled RR segments
- **Cross-validation**: 5-fold StratifiedKFold on the **training set** segments
- **Target**: 3-class time bins (Near/Mid/Far)

## Results

| Metric | Value |
|--------|-------|
| F1 Macro (5-fold CV) | **82.2%** |
| Accuracy (5-fold CV) | ~82% |

### Feature Importance (top 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | mean | 0.15 |
| 2 | p5 | 0.12 |
| 3 | median | 0.11 |
| 4 | p95 | 0.10 |
| 5 | std | 0.08 |

## Critical Observation

This 82.2% result was **misleadingly high**. The cross-validation was segment-level (`StratifiedKFold`), meaning segments from the same patient could appear in both train and test folds. Since each patient has ~250 segments with highly correlated features, the model was effectively memorizing patient-level baselines rather than learning generalizable temporal patterns.

This was the primary motivation for Experiment 7 (patient-aware diagnostics).

## Comparison with Neural Network

| Model | Evaluation Method | F1 Macro |
|-------|------------------|----------|
| Random Forest | Segment-level 5-fold CV (leaky) | 82.2% |
| NN v9 (CPC features) | Patient-held-out validation | 37.8% |
| NN v10 (CPC + HRV) | Patient-held-out validation | 33.4% |

The enormous gap (82% vs 34%) strongly suggested data leakage in the RF evaluation.

## Conclusion

The 82.2% RF result is **not reliable** due to patient-level data leakage. It represents within-patient pattern memorization, not cross-patient generalization. Proper patient-grouped evaluation is essential.
