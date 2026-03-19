# ST-CoxNet: Experiment Summary

## Project Goal

Predict temporal proximity to Atrial Fibrillation (AFib) onset from Sinus Rhythm (SR) RR interval segments using self-supervised (CPC) and supervised learning approaches.

## Dataset

| Property | Value |
|----------|-------|
| Source | IRIDIA AFib Dataset |
| Signal | RR intervals (inter-beat intervals from ECG) |
| SR window | 1.5 hours before each AFib episode |
| AFib window | 1 hour per episode |
| Segment size | 10 windows × 100 RR intervals = 1,000 RR intervals |
| Stride | 20 RR intervals |
| Preprocessing | RobustScaler, times normalized to [0,1] |
| Train patients | ~107 |
| Validation patients | ~21 (held out at patient level) |
| Train SR segments | 26,964 |
| Val SR segments | 5,763 |

## Experiment Timeline

| # | Experiment | Script | Best Val F1 | Status |
|---|-----------|--------|------------|--------|
| 1 | [CPC Pretraining](01_cpc_pretraining.md) | v4-v8 | 70% acc (self-supervised) | Successful |
| 2 | [AFib vs SR Classification](02_afib_vs_sr_classification.md) | v4/v8 | **92.7%** | Successful |
| 3 | [Time-to-Event Regression](03_time_to_event_regression.md) | v7 | MAE ~0.25 | Failed |
| 4 | [Time-Bin Classification (CPC only)](04_time_bin_classification_v9.md) | v9 | 37.8% | Near random |
| 5 | [Diagnostic: HRV Feature Analysis](05_diagnostic_phase1_feature_analysis.md) | debug_time_signal.py | N/A | Completed |
| 6 | [Diagnostic: RF Baseline (leaky)](06_diagnostic_phase2_rf_baseline.md) | debug_time_signal.py | 82.2%* | *Data leakage* |
| 7 | [Diagnostic: Patient-Aware Analysis](07_patient_aware_diagnostics.md) | debug_time_signal_v2.py | 42.5% (norm) | Completed |
| 8 | [Multi-Modal NN (CPC + HRV)](08_multimodal_v10.md) | v10 | 42.8% | Moderate |
| 9 | [Sequence Transformer](09_sequence_level_v11_transformer.md) | v11 | **43.5%** | Best NN result |
| 10 | [Sequence GRU (lightweight)](10_sequence_level_v11_gru.md) | v11 revised | 41.0% | Overfits less |

*\*82.2% was due to data leakage; proper patient-grouped evaluation gives 34.6%*

## Key Results Comparison

### Time-Bin Classification (3-class, random = 33.3%)

| Model | Normalization | Evaluation | Val F1 Macro |
|-------|-------------|------------|-------------|
| CPC features only (v9) | Global | Patient-held-out | 37.8% |
| CPC + HRV (v10) | Global | Patient-held-out | 34.0% |
| CPC + HRV (v10) | Per-patient | Patient-held-out | 42.8% |
| Sequence Transformer (v11) | Per-patient | Patient-held-out | **43.5%** |
| Sequence GRU (v11) | Per-patient | Patient-held-out | 41.0% |
| Random Forest | Global, segment-CV | Segment-level (leaky!) | 82.2%* |
| Random Forest | Global, patient-CV | Patient-grouped | 34.6% |
| Random Forest | Per-patient, patient-CV | Patient-grouped | 42.5% |

### AFib vs SR Binary Classification

| Model | Val F1 Macro |
|-------|-------------|
| CPC encoder + classification head (v8) | **92.7%** |

## Critical Findings

### 1. CPC Representations are Effective for AFib Detection

The CPC encoder successfully learns representations that distinguish AFib from SR rhythms (92.7% F1). Cross-batch InfoNCE was essential -- within-batch contrastive learning failed completely.

### 2. Temporal Proximity Signal is Extremely Weak

The central finding across all experiments: short SR segments (~2 minutes of RR intervals) carry very little information about how close they are to AFib onset.

- Spearman correlations between HRV features and time-to-event: |rho| < 0.1
- ANOVA shows statistical significance but negligible effect sizes
- PCA and t-SNE show no visible separation by time bin

### 3. Inter-Patient Variability Dominates

The most impactful finding from the diagnostic analysis:

- RF with segment-level CV: 82.2% (patient data leakage)
- RF with patient-grouped CV: **34.6%** (near random)
- The 47.6pp drop proves the RF was memorizing patient baselines, not temporal patterns

### 4. Per-Patient Normalization is Essential

When features are z-scored within each patient (removing baseline variability):
- RF: 34.6% → 42.5% (+7.9pp)
- NN v10: 34.0% → 42.8% (+8.8pp)

This confirms that **relative changes** within a patient carry a weak but real signal, while absolute values are useless across patients.

### 5. Sequence Modeling Provides Marginal Benefit

Modeling the trajectory of K=16 consecutive segments (v11 Transformer: 43.5%) provides only a ~1pp improvement over single-segment classification with patient normalization (v10: 42.8%). The temporal evolution is too subtle to capture reliably with the available data.

### 6. Model Capacity is Not the Bottleneck

All neural networks severely overfit regardless of architecture:

| Model | Parameters | Train F1 | Val F1 | Gap |
|-------|-----------|----------|--------|-----|
| v10 MLP | ~50K | 74.5% | 42.8% | 32pp |
| v11 Transformer | ~500K | 96.2% | 43.5% | 53pp |
| v11 GRU | 88K | 90.9% | 41.0% | 50pp |

Heavy regularization (dropout 0.4, label smoothing, weight decay 0.05) slows but does not prevent overfitting.

## Root Cause Analysis

The fundamental limitation is the combination of:

1. **Weak signal**: Within-patient temporal HRV changes before AFib are subtle (rho ~0.1)
2. **High inter-patient variability**: Baseline HRV varies enormously between patients
3. **Small patient count**: 107 training patients / 21 validation patients is insufficient for learning generalizable temporal patterns
4. **Short segments**: ~2 minutes of RR intervals per segment is too little temporal context
5. **Single modality**: RR intervals alone may not contain enough information about approaching AFib

## Recommendations for Future Work

### Most Promising

1. **More patients**: The 107-patient dataset is too small for cross-patient generalization. Scaling to 500+ patients would help the most.
2. **Longer temporal context**: Increase the SR window from 1.5 hours to 6-12+ hours to capture slower-evolving pre-AFib changes.
3. **Multi-modal input**: Combine RR intervals with raw ECG morphology (P-wave changes, T-wave alternans) which may carry stronger pre-AFib signals.

### Worth Exploring

4. **Binary task**: Simplify to Near vs Far (drop Mid) for a stronger signal-to-noise ratio.
5. **Patient-adaptive models**: Use few-shot learning or meta-learning to quickly adapt to a new patient's baseline.
6. **Domain adversarial training**: Force the model to learn patient-invariant features.

### Unlikely to Help

7. Larger/different neural network architectures (problem is data, not model)
8. Different loss functions on the same features
9. More complex HRV feature engineering (the 15 features already capture the available signal)

## File Structure

```
results/
├── 00_SUMMARY.md                          (this file)
├── 01_cpc_pretraining.md                  (CPC self-supervised learning)
├── 02_afib_vs_sr_classification.md        (binary AFib detection)
├── 03_time_to_event_regression.md         (direct regression -- failed)
├── 04_time_bin_classification_v9.md       (3-class from CPC features)
├── 05_diagnostic_phase1_feature_analysis.md (HRV features + statistics)
├── 06_diagnostic_phase2_rf_baseline.md    (RF baseline -- leaky)
├── 07_patient_aware_diagnostics.md        (patient-grouped evaluation)
├── 08_multimodal_v10.md                   (CPC + HRV neural network)
├── 09_sequence_level_v11_transformer.md   (trajectory Transformer)
└── 10_sequence_level_v11_gru.md           (trajectory lightweight GRU)
```

## Relevant Plots

| Plot | Path | Description |
|------|------|-------------|
| CPC training curves | `plots/cpc_v8_accuracy.png` | CPC pretraining accuracy |
| AFib/SR classification | `plots/cls_v8_f1.png` | Binary classification F1 |
| Time-bin v9 | `plots/timebin_v9_f1.png` | CPC-only time-bin F1 |
| Violin plots | `plots/debug_signal/violin_plots.png` | HRV feature distributions per bin |
| PCA | `plots/debug_signal/pca_plots.png` | PCA of HRV features by bin |
| t-SNE | `plots/debug_signal/tsne_plot.png` | t-SNE of HRV features |
| Feature importance | `plots/debug_signal/feature_importance.png` | RF feature importances |
| Within-patient Spearman | `plots/debug_signal/within_patient_spearman.png` | Per-patient correlations |
| Multi-modal v10 F1 | `plots/timebin_v10_f1.png` | CPC+HRV F1 curves |
| Sequence v11 F1 | `plots/timebin_v11_f1.png` | Sequence model F1 curves |

## wandb Projects

| Project | Description |
|---------|-------------|
| `CPC-MultiModal-TimeBin-v10` | v10 with global normalization |
| `CPC-MultiModal-PatientNorm-v10` | v10 with per-patient normalization |
| `CPC-SeqLevel-TimeBin-v11` | v11 Transformer sequence model |
| `CPC-SeqLevel-GRU-v11` | v11 lightweight GRU sequence model |
