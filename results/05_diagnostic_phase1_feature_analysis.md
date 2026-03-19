# Experiment 5: Diagnostic Phase 1 -- HRV Feature Analysis

## Objective

Systematically determine whether a temporal proximity signal exists in SR segments by computing hand-crafted HRV features and analyzing their relationship with time-to-AFib-onset.

## Script

- `debug_time_signal.py`

## HRV Features Computed (15 total)

| # | Feature | Description |
|---|---------|-------------|
| 1 | artifact_rate | Fraction of RR intervals removed by outlier filter |
| 2 | mean | Mean RR interval (scaled) |
| 3 | std | Standard deviation of RR intervals |
| 4 | median | Median RR interval |
| 5 | iqr | Interquartile range (p75 - p25) |
| 6 | p5 | 5th percentile |
| 7 | p95 | 95th percentile |
| 8 | skewness | Distribution skewness |
| 9 | kurtosis | Distribution kurtosis |
| 10 | cv | Coefficient of variation (std/mean) |
| 11 | rmssd | Root mean square of successive differences |
| 12 | sdsd | Standard deviation of successive differences |
| 13 | pnn50_scaled | Fraction of successive diffs > 0.5 (on scaled data) |
| 14 | alpha1 | DFA short-term scaling exponent |
| 15 | sample_entropy | Sample entropy (m=2, r=0.2*std) |

## Statistical Tests

### ANOVA (One-Way, across 3 time bins)

Most features showed statistically significant differences across bins (p < 0.05), including:
- **std**, **iqr**, **rmssd**, **sdsd** (variability measures)
- **alpha1**, **sample_entropy** (complexity measures)
- **skewness**, **kurtosis** (distribution shape)

### Kruskal-Wallis (Non-parametric)

Confirmed ANOVA results. Significance in many features, but with small effect sizes.

### Spearman Correlation (feature vs time-to-event)

| Feature | Spearman rho | p-value |
|---------|-------------|---------|
| mean | -0.02 | 0.001 |
| std | +0.05 | <0.001 |
| rmssd | +0.04 | <0.001 |
| alpha1 | -0.03 | <0.001 |
| sample_entropy | +0.03 | <0.001 |

All correlations were **very weak** (|rho| < 0.1), despite being statistically significant due to large sample size (N=26,964).

## Visualizations

- **Violin plots** (`plots/debug_signal/violin_plots.png`): Distributions of each feature per time bin -- heavy overlap between bins
- **PCA** (`plots/debug_signal/pca_plots.png`): First 2 PCs show no visible cluster separation by time bin
- **t-SNE** (`plots/debug_signal/tsne_plot.png`): No clear clustering by time bin

## Conclusion

A weak temporal signal **exists** statistically (significant ANOVA, non-zero Spearman), but it is extremely weak in magnitude (|rho| < 0.1) and completely overwhelmed by inter-segment and inter-patient variability. Visual inspection confirms no separable clusters by time bin.
