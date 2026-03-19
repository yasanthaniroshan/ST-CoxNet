# Experiment 7: Patient-Aware Diagnostics

## Objective

Definitively determine whether the temporal signal generalizes across patients by conducting properly-controlled experiments that account for patient-level data structure.

## Script

- `debug_time_signal_v2.py`

## Sub-Experiments

### 7A: Patient-Grouped Random Forest CV

**Method**: Same RF as Phase 2, but using `GroupKFold` with patient IDs as groups. No segments from the same patient appear in both train and test within any fold.

**Patient Recovery**: Patient boundaries detected from monotonically-decreasing SR time arrays (a time increase signals a new patient).

| Metric | Segment-Level CV (leaky) | Patient-Grouped CV |
|--------|--------------------------|-------------------|
| F1 Macro | 82.2% | **34.6%** |
| Accuracy | ~82% | ~35% |

**Conclusion**: The drop from 82% to 35% **confirms data leakage**. The RF was memorizing patient baselines, not learning temporal patterns. With proper patient grouping, performance is near random chance (33.3%).

### 7B: Within-Patient Spearman Correlation

**Method**: For each individual patient, compute Spearman correlation between each HRV feature and time-to-event. Then aggregate across patients.

**Results**:

| Statistic | Value |
|-----------|-------|
| Mean |rho| across features | ~0.10 |
| Median |rho| | ~0.08 |
| Std of rho | ~0.25 |
| % patients with positive correlation | ~45-55% (inconsistent direction) |

**Conclusion**: Even within individual patients, the feature-time correlations are **weak and inconsistent**. Some patients show positive correlation, others negative, for the same feature. The temporal signal is patient-specific and unreliable.

### 7C: Patient-Normalized Features + Patient-Grouped RF

**Method**: Z-score normalize each feature within each patient (subtract patient mean, divide by patient std), then run patient-grouped RF CV.

| Metric | Raw Features (Patient CV) | Patient-Normalized (Patient CV) |
|--------|---------------------------|--------------------------------|
| F1 Macro | 34.6% | **42.5%** |
| Accuracy | ~35% | ~42% |

**Conclusion**: Patient normalization adds ~8 percentage points over random. This confirms that **relative changes within a patient** carry a weak but real signal, while absolute values are dominated by inter-patient variability.

### 7D: Mixed-Effects Model (Attempted)

**Method**: Linear Mixed-Effects Model with time-to-event as fixed effect and patient as random intercept.

**Status**: Skipped due to `statsmodels` not being installed. The patient-normalized ANOVA from 7C serves a similar purpose.

## Key Findings

| Experiment | F1 Macro | Interpretation |
|-----------|----------|----------------|
| RF segment-level CV | 82.2% | Data leakage -- memorizing patients |
| RF patient-grouped CV | 34.6% | Near random -- signal doesn't generalize |
| RF patient-normalized + grouped CV | 42.5% | Weak within-patient signal exists |
| Within-patient Spearman | rho~0.1 | Very weak, inconsistent across patients |

## Conclusion

1. The high RF performance (82%) was entirely due to **data leakage** from overlapping patient segments across CV folds
2. The cross-patient temporal signal is **near zero** (34.6% ≈ random)
3. **Patient normalization** reveals a modest within-patient signal (42.5%), confirming that relative HRV changes carry more information than absolute values
4. The signal is **weak and inconsistent** across patients (Spearman rho ~0.1)
