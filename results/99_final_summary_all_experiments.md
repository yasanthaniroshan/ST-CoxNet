## Final Summary (all experiments so far)

This document consolidates results from:
- `01_v13_cpc_temporal_ranking.md`
- `02_v14_within_patient_ranking.md`
- `01_v15_single_patient_sanity_check.md`
- `02_v15b_per_patient_10_random_models.md`
- `03_v15c_full_patient_audit.md`
- `04_v16_multi_patient_direct_regression_all_patients.md`
- `05_v17_excellent_only_patient_level_split.md`
- `06_v17_excellent_only_segment_level_leakage.md`
- `01_cpc_pretraining.md`
- `02_afib_vs_sr_classification.md`
- `03_time_to_event_regression.md`
- `04_time_bin_classification_v9.md`
- `05_diagnostic_phase1_feature_analysis.md`
- `06_diagnostic_phase2_rf_baseline.md`
- `07_patient_aware_diagnostics.md`
- `08_multimodal_v10.md`
- `09_sequence_level_v11_transformer.md`
- `10_sequence_level_v11_gru.md`

## Additional summary: this conversation branch (v4–v11)

The experiments in this branch used RR-window segments and discretized the target time-to-AFib onset into 3 bins: **Near / Mid / Far**.

### Main numeric results (patient-held-out evaluation)

Random chance baseline for 3 classes is **33.3%**.

- Segment-level CPC-only time bins (`cpc_train_v9.py`): **37.8%** macro F1
- Multi-modal CPC + HRV with global normalization (`cpc_train_v10.py`): **34.0%** macro F1
- Multi-modal CPC + HRV with per-patient normalization (`cpc_train_v10.py`): **42.8%** macro F1
- Sequence-level Transformer trajectory (`cpc_train_v11.py`, Transformer): **43.5%** macro F1
- Sequence-level lightweight GRU trajectory (`cpc_train_v11.py`, revised GRU): **41.0%** macro F1

### Diagnostics and root cause

1. HRV features show a **weak temporal signal**, but it does not generalize well across patients.
2. Segment-level RandomForest CV was inflated by **data leakage**: **82.2%** (leaky) vs **34.6%** (patient-grouped).
3. Per-patient normalization recovers most of the usable signal:
   - RF: **34.6% → 42.5%**
   - NN: **34.0% → 42.8%**
4. Sequence modeling helps slightly, but the cross-patient ceiling remains in the low-40% macro F1 range.

### Key conclusion for v4–v11

With short SR segments and RR-only inputs, the time-proximity signal is **too weak/inconsistent across patients** for robust prediction. Patient baseline variability dominates, so the model must either (a) normalize/condition on patients, or (b) use longer context / additional modalities / patient-adaptive modeling.

### Key findings

0. **Early experiments (v13 → v14) revealed the core failure mechanism: identity shortcuts**
   - v13 used batch-level temporal ranking across patients and produced weak SR-ρ and near-random Cox C-index; it also had a NaN issue fixed by masking the diagonal in `TemporalRankingLoss`.
   - v14 corrected the loss to operate within patients only and improved SR-ρ to ~0.47, but Cox C-index still only reached ~0.553, suggesting the representation/transformation needed for cross-patient generalization was still missing.

1. **RR windows contain a strong temporal signal per patient**
   - v15: single patient achieves MAE ~2.38 min and Spearman ρ ~0.986 (highly learnable).
   - v15c: across 87 patients, the distribution is dominated by “excellent” cases (56/87), with mean ρ ~0.769.

2. **A non-trivial minority of patients are hard/noisy for direct RR→TTE regression**
   - From v15c: 17 “weak” and 11 “poor” patients (ρ < 0.3 for 11).
   - This means that naive pooling across all patients can be contaminated by outliers.

3. **The major failure mode is patient-level generalization/domain shift**
   - v16 (all patients pooled): global ρ ~0.154 and MAE ~18.77 min.
   - v17 (excellent-only, *patient-unseen* split): still fails (global ρ ~-0.0271, MAE ~19.34 min).
   - Therefore: even when removing noisy patients, a single global regression model on raw RR windows does not generalize to new patients.

4. **When leakage is allowed, the model performs well**
   - v17 (excellent-only, segment-level leakage split): global ρ ~0.860 and MAE ~7.13 min.
   - This confirms the model can learn a pooled temporal mapping **if it has already seen patient-specific evidence during training** (even indirectly via leaked segments).

### What this implies for the next modeling direction
- Filtering alone (removing poor patients) is not sufficient for strict patient-unseen performance.
- The model likely needs explicit mechanisms to handle patient-to-patient variability, e.g.:
  - input normalization per patient (relative RR features),
  - patient conditioning/embeddings,
  - learning a relative/ranking target instead of absolute minutes,
  - or a two-stage approach that adapts per patient.

### Recommended next experiment (high value)
- Rework the target/task so that it is less sensitive to patient-specific RR scale, and evaluate with strict patient-level splits.

