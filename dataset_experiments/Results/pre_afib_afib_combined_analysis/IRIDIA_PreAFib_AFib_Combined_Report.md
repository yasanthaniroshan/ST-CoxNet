# IRIDIA Pre-AFib + AFib + Combined Episode Analysis

> Generated on 2026-03-21 14:05:06

## Objective

For each AF episode, analyze RR windows of **10, 20, 30, 60, 120, 180 minutes** for: **pre_afib** (before onset), **afib** (within labeled AF, up to H min), and **both** (concatenation pre then AF).

## Method

- AF onset: global index = `file_offset[start_file_index] + start_rr_index`.
- AF interval: `[af_start, af_end)` from RR labels (`end_file_index`, `end_rr_index`).
- **pre_afib**: last H minutes of RR strictly before `af_start`.
- **afib**: first H minutes of RR inside the labeled AF interval (or all AF RR if shorter).
- **both**: `concat(pre_afib, afib)`; target duration for sufficiency/score = **2×H** minutes.
- Calibration handling: first **60 seconds** of RR are removed using cumulative RR duration.
- Same ML heuristics as pre-AFib script: coverage ≥95%, RR count, outliers [250,2500] ms, duplicates, quality score.

## Summary by segment_type × horizon

| segment_type | horizon_min | n_rows | n_nonempty | sufficient_% | ml_ready_% | mean_quality | mean_rr_count | mean_outlier_pct_nonempty | median_outlier_pct_nonempty | p95_outlier_pct_nonempty | global_outlier_pct | mean_mean_rr_ms | mean_std_rr_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| afib | 10 | 388 | 388 | 73.20 | 64.43 | 86.72 | 881.5 | 0.2932 | 0.0000 | 1.7438 | 0.4357 | 573.0 | 128.3 |
| afib | 20 | 388 | 388 | 67.78 | 60.82 | 83.67 | 1628.3 | 0.2434 | 0.0000 | 1.1712 | 0.3403 | 573.2 | 128.3 |
| afib | 30 | 388 | 388 | 64.43 | 59.02 | 81.92 | 2330.7 | 0.2328 | 0.0000 | 0.9206 | 0.2982 | 572.7 | 128.2 |
| afib | 60 | 388 | 388 | 55.67 | 52.32 | 78.51 | 4229.2 | 0.2239 | 0.0129 | 0.8831 | 0.2439 | 570.7 | 128.0 |
| afib | 120 | 388 | 388 | 43.56 | 42.53 | 74.10 | 7369.6 | 0.2114 | 0.0201 | 0.6970 | 0.1875 | 569.1 | 128.6 |
| afib | 180 | 388 | 388 | 33.25 | 32.22 | 70.77 | 9802.5 | 0.2148 | 0.0187 | 0.7834 | 0.1854 | 569.0 | 129.0 |
| both | 10 | 388 | 388 | 66.75 | 53.09 | 88.76 | 1594.6 | 0.2932 | 0.0000 | 1.3683 | 0.3695 | 692.8 | 198.9 |
| both | 20 | 388 | 388 | 61.86 | 52.06 | 87.21 | 3052.5 | 0.2513 | 0.0311 | 0.9977 | 0.3026 | 700.9 | 197.7 |
| both | 30 | 388 | 388 | 58.51 | 48.71 | 86.21 | 4454.3 | 0.2414 | 0.0414 | 0.8948 | 0.2806 | 707.7 | 196.1 |
| both | 60 | 388 | 388 | 50.26 | 43.81 | 84.13 | 8414.0 | 0.2450 | 0.0531 | 1.0058 | 0.2634 | 716.0 | 195.3 |
| both | 120 | 388 | 388 | 38.92 | 35.82 | 81.54 | 15645.1 | 0.2392 | 0.0584 | 1.0921 | 0.2438 | 721.3 | 194.8 |
| both | 180 | 388 | 388 | 28.87 | 26.29 | 79.47 | 22036.2 | 0.2382 | 0.0597 | 1.1879 | 0.2382 | 725.3 | 194.9 |
| pre_afib | 10 | 388 | 364 | 93.56 | 59.79 | 87.69 | 713.1 | 0.2198 | 0.0000 | 1.1368 | 0.2877 | 789.5 | 141.9 |
| pre_afib | 20 | 388 | 364 | 93.30 | 57.73 | 87.55 | 1424.2 | 0.2035 | 0.0000 | 1.0312 | 0.2595 | 790.5 | 141.1 |
| pre_afib | 30 | 388 | 364 | 93.04 | 57.22 | 87.34 | 2123.6 | 0.2079 | 0.0000 | 0.8721 | 0.2613 | 794.2 | 139.0 |
| pre_afib | 60 | 388 | 364 | 91.75 | 55.67 | 86.80 | 4184.9 | 0.2399 | 0.0366 | 1.1520 | 0.2831 | 794.4 | 142.2 |
| pre_afib | 120 | 388 | 364 | 89.43 | 53.09 | 86.12 | 8275.5 | 0.2641 | 0.0506 | 1.2495 | 0.2941 | 788.2 | 147.2 |
| pre_afib | 180 | 388 | 364 | 85.82 | 51.29 | 85.50 | 12233.7 | 0.2633 | 0.0522 | 1.4044 | 0.2804 | 783.8 | 150.0 |

## Physical Cleanliness Verification

| segment_type | horizon_min | n_nonempty | global_outlier_% | mean_outlier_% | p95_outlier_% | p99_outlier_% | fail(>2%)_% | fail(>5%)_% | fail(dup>10%)_% | fail(rr<200)_% | fail(min<250)_% | fail(max>2500)_% | fail(any rule)_% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| afib | 10 | 388 | 0.4357 | 0.2932 | 1.7438 | 4.5610 | 3.61 | 1.03 | 12.63 | 13.40 | 30.67 | 0.77 | 25.26 |
| afib | 20 | 388 | 0.3403 | 0.2434 | 1.1712 | 4.4146 | 3.09 | 1.03 | 12.37 | 13.40 | 39.69 | 2.06 | 25.00 |
| afib | 30 | 388 | 0.2982 | 0.2328 | 0.9206 | 4.0245 | 3.09 | 0.77 | 11.86 | 13.40 | 42.78 | 2.84 | 24.23 |
| afib | 60 | 388 | 0.2439 | 0.2239 | 0.8831 | 3.2297 | 2.58 | 1.03 | 11.60 | 13.40 | 51.03 | 3.87 | 24.23 |
| afib | 120 | 388 | 0.1875 | 0.2114 | 0.6970 | 3.2210 | 2.32 | 1.03 | 10.57 | 13.40 | 59.54 | 5.67 | 23.20 |
| afib | 180 | 388 | 0.1854 | 0.2148 | 0.7834 | 3.2210 | 2.58 | 1.03 | 10.57 | 13.40 | 62.11 | 7.99 | 23.20 |
| both | 10 | 388 | 0.3695 | 0.2932 | 1.3683 | 5.2438 | 3.35 | 1.29 | 20.10 | 0.00 | 43.56 | 2.06 | 21.39 |
| both | 20 | 388 | 0.3026 | 0.2513 | 0.9977 | 4.2304 | 3.09 | 0.52 | 19.33 | 0.00 | 56.96 | 4.12 | 19.85 |
| both | 30 | 388 | 0.2806 | 0.2414 | 0.8948 | 3.9395 | 2.84 | 0.26 | 21.13 | 0.00 | 62.89 | 5.41 | 21.39 |
| both | 60 | 388 | 0.2634 | 0.2450 | 1.0058 | 3.2614 | 2.84 | 0.26 | 23.45 | 0.00 | 78.09 | 9.02 | 23.71 |
| both | 120 | 388 | 0.2438 | 0.2392 | 1.0921 | 2.5684 | 1.29 | 0.00 | 23.20 | 0.00 | 87.89 | 14.18 | 23.20 |
| both | 180 | 388 | 0.2382 | 0.2382 | 1.1879 | 2.1085 | 1.29 | 0.00 | 23.20 | 0.00 | 93.04 | 17.53 | 23.20 |
| pre_afib | 10 | 364 | 0.2877 | 0.2198 | 1.1368 | 3.7758 | 2.75 | 0.82 | 35.16 | 0.00 | 28.02 | 1.37 | 35.99 |
| pre_afib | 20 | 364 | 0.2595 | 0.2035 | 1.0312 | 4.3294 | 2.20 | 0.82 | 37.09 | 0.00 | 40.66 | 2.47 | 37.91 |
| pre_afib | 30 | 364 | 0.2613 | 0.2079 | 0.8721 | 4.2248 | 2.47 | 0.27 | 38.19 | 0.00 | 48.35 | 3.02 | 38.46 |
| pre_afib | 60 | 364 | 0.2831 | 0.2399 | 1.1520 | 3.6116 | 2.75 | 0.27 | 38.74 | 0.00 | 63.46 | 6.04 | 39.01 |
| pre_afib | 120 | 364 | 0.2941 | 0.2641 | 1.2495 | 2.4308 | 2.75 | 0.27 | 40.11 | 0.00 | 78.85 | 11.26 | 40.38 |
| pre_afib | 180 | 364 | 0.2804 | 0.2633 | 1.4044 | 2.4364 | 1.65 | 0.00 | 39.56 | 0.00 | 86.26 | 13.74 | 39.56 |

## Notes

- **afib** segments are often shorter than H minutes because labeled AF duration in this dataset can be brief; `is_sufficient_duration` reflects whether ≥95% of requested H min was captured.
- **both** uses a **2H-minute** target for coverage/score; pre and AF parts are still each capped at H min.
- Outlier stats use non-empty segments only (`rr_count > 0`) and include `global_outlier_pct = sum(outliers)/sum(rr_count)*100`.

## Visualizations

### ML-ready by horizon and segment

![ML-ready by horizon and segment](plots/01_ml_ready_by_horizon_segment.png)

### Quality score by horizon and segment

![Quality score by horizon and segment](plots/02_quality_score_by_horizon_segment.png)

### Mean RR by horizon and segment

![Mean RR by horizon and segment](plots/03_mean_rr_by_horizon_segment.png)

### Outlier % by horizon and segment

![Outlier % by horizon and segment](plots/04_outlier_pct_by_horizon_segment.png)

### ML-ready heatmap

![ML-ready heatmap](plots/05_ml_ready_heatmap.png)

### Outlier % ECDF

![Outlier % ECDF](plots/06_outlier_pct_ecdf.png)

### Outlier band distribution

![Outlier band distribution](plots/07_outlier_band_distribution.png)

---

*Generated by `dataset_experiments/IRIDIA/pre_afib_afib_combined_analysis.py`*