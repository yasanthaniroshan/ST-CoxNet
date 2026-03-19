# Experiment 01 — Raw Dataset Quality Analysis (IRIDIA AFib)

**Source script:** `raw_dataset_analysis.py`  
**Dataset path:** `/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1`  
**Outputs folder:** `plots/dataset_analysis/`

## Key results (highlights)

### Dataset filtering / cohort
- Records passing filters: **128**
- Unique patients with records: **102**
- Skipped episodes:
  - cross-file episodes: **20**
  - AFib too short: **175**
  - SR too short: **65**

### RR interval quality (global)
- Mean RR: **721.88 ms**, Std: **256.14 ms**
- Physiological-range outliers: **36** (**0.002%**)  
- IQR-based outliers (factor=3.0): **71** (**0.004%**)  
- Combined outliers (union): **71** (**0.004%**)
- Successive-difference outliers:
  - per-record mean count: **4434.0**
  - per-record mean: **35.97%**
  - per-record max: **86.20%**

### Windowing / segmentation
- Total segments: **73,164**
- Segment label balance:
  - SR (label=-1): **32,727** (**44.7%**)
  - Mixed (label=0): **6,394** (**8.7%**)
  - AFib (label=1): **34,043** (**46.5%**)
- Segment overlap: **98.0%** (inflation factor **717.3x**)
- Effective independent N (patient-level): **102**

### RobustScaler distributions
- Post-scaling (all RR values):
  - Mean: **0.0409**, Std: **0.6307**
  - Median: **0.0000**

### Patient separability (SR vs AFib, patient-level HRV)
Strongest effect sizes (Cohen’s d reported in the log):
- pNN50: **d=2.52**
- MedianNN: **d=2.37**
- pNN20: **d=2.34**

### Patient dependency (ICC)
- Mean ICC across HRV features: **0.274**
- Interpretation: within-patient variance dominates; patient identity does not overwhelmingly drive features.

## Conclusions / recommendations
- Outlier rate is acceptable (**0.004%** combined physiological/IQR flags); no aggressive cleaning needed.
- Patient dependency is **moderate**: patient-level splits still recommended.
- Segment overlap is extremely high (98%); treat segment-level samples carefully to avoid leakage.
- Consider additional artifact rejection for successive-difference outliers.

## Generated plots
- `plots/dataset_analysis/01_rr_quality_outliers.png`
- `plots/dataset_analysis/02_label_balance_tte.png`
- `plots/dataset_analysis/03_before_after_scaling.png`
- `plots/dataset_analysis/04_hrv_violin_sr_vs_afib.png`
- `plots/dataset_analysis/05_temporal_dynamics.png`
- `plots/dataset_analysis/06_example_rr_traces.png`
- `plots/dataset_analysis/07a_dr_by_label.png`
- `plots/dataset_analysis/07b_dr_by_patient.png`
- `plots/dataset_analysis/09_patient_dependency_icc.png`
- `plots/dataset_analysis/10_ml_readiness.png`

## Full report export

```text
================================================================================
  IRIDIA AFIB DATASET — RAW DATA QUALITY ANALYSIS
  Generated: 2026-03-19 09:53:11
================================================================================

Dataset path : /home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1
Total patient directories found: 167

────────────────────────────────────────────────────────────────────────────────
SECTION 1: RECORD FILTERING & PATIENT STATISTICS
────────────────────────────────────────────────────────────────────────────────
  Records that pass all filters : 128
  Unique patients with records  : 102
  Skipped — missing CSV         : 0
  Skipped — cross-file episode  : 20
  Skipped — AFib too short      : 175
  Skipped — SR too short        : 65

  Records per patient — min: 1, max: 4, mean: 1.3, median: 1

  AF duration (s)  — min: 4, max: 73, mean: 21, median: 13
  SR duration (s)  — min: 11, max: 208, mean: 58, median: 43

────────────────────────────────────────────────────────────────────────────────
SECTION 2: RAW RR INTERVAL LOADING
────────────────────────────────────────────────────────────────────────────────

  Successfully loaded records       : 128
  Total RR intervals (all records)  : 1,589,954
  Total SR intervals                : 781,122
  Total AFib intervals              : 808,832

  Global RR stats (ms):
    Mean: 721.88  Std: 256.14
    Min : 200.00  Max: 17675.00
    Median: 705.00  IQR: 405.00

────────────────────────────────────────────────────────────────────────────────
SECTION 3: RR INTERVAL QUALITY & OUTLIER DETECTION
────────────────────────────────────────────────────────────────────────────────

  Physiological-range outliers (RR<200 or >2500 ms):
    Count : 36  (0.002%)
    Too short (<200 ms) : 0
    Too long  (>2500 ms): 36

  IQR-based outliers (factor=3.0):
    Q1=500.0, Q3=905.0, IQR=405.0
    Lower bound: -715.0, Upper bound: 2120.0
    Count : 71  (0.004%)

  Combined outliers (union):
    Count : 71  (0.004%)

  Successive-difference outliers (>20% of local median):
    Per-record mean count : 4434.0
    Per-record mean %     : 35.97%
    Per-record max  %     : 86.20%

  Outlier comparison — SR vs AFib:
    SR  physio outliers : 23 (0.003%)
    AFib physio outliers: 13 (0.002%)
    SR  IQR outliers   : 179 (0.023%)
    AFib IQR outliers  : 2,468 (0.305%)
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/01_rr_quality_outliers.png

────────────────────────────────────────────────────────────────────────────────
SECTION 4: SEGMENTATION & LABEL BALANCE
────────────────────────────────────────────────────────────────────────────────

  Segment size : 1000 RR intervals (10 windows × 100)
  Stride       : 20

  Total segments : 73,164
    SR   (label=-1) : 32,727  (44.7%)
    Mixed (label=0) : 6,394  (8.7%)
    AFib (label=1)  : 34,043  (46.5%)
    Unique patients : 102

  Segment overlap  : 98.0% (stride 20 / segment 1000)
  Segments per patient — min: 366, max: 2618, mean: 717, median: 598
  Effective independent N (patient-level): 102
  Inflation factor vs segment count: 717.3x

  Time-to-event (SR segments only, seconds):
    Min : 0.0
    Max : 4919.6
    Mean: 2274.3
    Median: 2270.6
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/02_label_balance_tte.png

────────────────────────────────────────────────────────────────────────────────
SECTION 5: DISTRIBUTIONS BEFORE & AFTER ROBUSTSCALER
────────────────────────────────────────────────────────────────────────────────

  Before scaling:
    Mean: 719.65  Std: 255.01
    Min : 200.00  Max: 17675.00
    Median: 705.00

  After RobustScaler:
    Mean: 0.0409  Std: 0.6307
    Min : -1.2625  Max: 41.9012
    Median: 0.0000
    Scaler center (median per feature): mean=703.10
    Scaler scale  (IQR per feature)   : mean=404.35

  SR (n=32,727 segments):
    Raw   — mean: 873.85, std: 219.03, median: 875.00
    Scaled— mean: 0.4223, std: 0.5417, median: 0.4198

  Mixed (n=6,394 segments):
    Raw   — mean: 752.75, std: 265.91, median: 730.00
    Scaled— mean: 0.1228, std: 0.6575, median: 0.0741

  AFib (n=34,043 segments):
    Raw   — mean: 565.19, std: 182.86, median: 515.00
    Scaled— mean: -0.3411, std: 0.4523, median: -0.4625
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/03_before_after_scaling.png

────────────────────────────────────────────────────────────────────────────────
SECTION 6: HRV FEATURE COMPARISON — SR vs AFIB (PATIENT-LEVEL)
────────────────────────────────────────────────────────────────────────────────

  Patients contributing SR segments  : 102
  Patients contributing AFib segments: 102

  Patient-level HRV (one mean per patient):
  SR patients: 102, AFib patients: 102
  Features: MeanNN, SDNN, RMSSD, pNN50, pNN20, CVNN, MedianNN, IQRNN, MeanHR

  Feature         SR mean     SR std    AF mean     AF std    Patient p        Seg p      d
  ──────────── ────────── ────────── ────────── ────────── ──────────── ──────────── ──────
  MeanNN           909.57     169.30     596.77     121.85     1.55e-26     3.69e-25   2.12 ***
  SDNN             114.83      63.87     136.50      47.78     2.70e-04     6.26e-05   0.38 ***
  RMSSD            126.31      85.41     182.92      69.77     1.98e-08     8.24e-08   0.73 ***
  pNN50             21.81      19.48      66.06      15.38     2.77e-27     2.87e-25   2.52 ***
  pNN20             39.69      21.88      82.24      13.44     9.14e-29     4.46e-28   2.34 ***
  CVNN               0.13       0.07       0.22       0.05     4.54e-17     2.02e-15   1.47 ***
  MedianNN         928.90     174.42     572.88     120.61     8.78e-29     8.27e-29   2.37 ***
  IQRNN            111.42      98.22     182.18      81.21     1.99e-12     1.60e-14   0.79 ***
  MeanHR            68.81      13.80     105.11      21.05     2.33e-26     3.77e-26   2.04 ***
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/04_hrv_violin_sr_vs_afib.png

────────────────────────────────────────────────────────────────────────────────
SECTION 7: TEMPORAL DYNAMICS — WITHIN-PATIENT CORRELATIONS
────────────────────────────────────────────────────────────────────────────────

  Within-patient Spearman correlations (feature vs time-to-event)
  Patients analyzed: 79 (min 10 SR segments required)
  Sub-sampling step: every 25 segments to reduce overlap

  Feature        median ρ     mean ρ    % sig   n_pts
  ──────────── ────────── ────────── ──────── ────────
  MeanNN          -0.0455    -0.0466    34.2%      79
  SDNN            -0.1203    -0.1036    34.2%      79
  RMSSD           -0.1879    -0.1322    39.2%      79
  pNN50           -0.2047    -0.1496    54.4%      79
  pNN20           -0.1216    -0.1182    43.0%      79
  CVNN            -0.1273    -0.1169    44.3%      79
  MedianNN        -0.0881    -0.0867    38.0%      79
  IQRNN           -0.0390    -0.0575    29.1%      79
  MeanHR           0.0455     0.0466    34.2%      79
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/05_temporal_dynamics.png

────────────────────────────────────────────────────────────────────────────────
SECTION 8: PER-RECORD OUTLIER DEEP DIVE & EXAMPLE RR TRACES
────────────────────────────────────────────────────────────────────────────────

  Top 5 worst records (highest outlier %):
    1. record_031 rec#1 — outlier: 2.70%, n_rr: 8,732, mean: 1030.6, std: 148.9
    2. record_105 rec#0 — outlier: 1.86%, n_rr: 10,943, mean: 822.3, std: 232.3
    3. record_121 rec#0 — outlier: 1.06%, n_rr: 14,430, mean: 623.6, std: 113.4
    4. record_110 rec#3 — outlier: 0.72%, n_rr: 11,456, mean: 423.3, std: 126.3
    5. record_026 rec#1 — outlier: 0.33%, n_rr: 11,863, mean: 758.6, std: 154.6

  Top 5 cleanest records (lowest outlier %):
    1. record_001 rec#0 — outlier: 0.00%, n_rr: 14,457, mean: 622.5, std: 172.9
    2. record_001 rec#1 — outlier: 0.00%, n_rr: 13,284, mean: 677.5, std: 217.7
    3. record_001 rec#2 — outlier: 0.00%, n_rr: 14,251, mean: 631.5, std: 216.0
    4. record_001 rec#3 — outlier: 0.00%, n_rr: 14,335, mean: 627.8, std: 193.1
    5. record_003 rec#0 — outlier: 0.00%, n_rr: 10,548, mean: 853.1, std: 271.7
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/06_example_rr_traces.png

────────────────────────────────────────────────────────────────────────────────
SECTION 9: DIMENSIONALITY REDUCTION (PCA & t-SNE)
────────────────────────────────────────────────────────────────────────────────

  PCA explained variance: PC1=0.564, PC2=0.319, total=0.883
  Components for 95% variance: 4 / 9
  t-SNE converged (KL divergence: 0.7268)
  Unique patients in DR sample: 102
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/07a_dr_by_label.png
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/07b_dr_by_patient.png

────────────────────────────────────────────────────────────────────────────────
SECTION 10: PATIENT DEPENDENCY — ICC & VARIANCE DECOMPOSITION
────────────────────────────────────────────────────────────────────────────────

  Intraclass Correlation Coefficient (ICC) per HRV feature
  ICC near 1.0 = variance dominated by patient identity
  ICC near 0.0 = variance dominated by within-patient differences

  Feature           ICC Interpretation
  ──────────── ──────── ────────────────────
  MeanNN          0.333 LOW — rhythm-driven
  SDNN            0.343 LOW — rhythm-driven
  RMSSD           0.360 LOW — rhythm-driven
  pNN50           0.171 LOW — rhythm-driven
  pNN20           0.199 LOW — rhythm-driven
  CVNN            0.214 LOW — rhythm-driven
  MedianNN        0.285 LOW — rhythm-driven
  IQRNN           0.227 LOW — rhythm-driven
  MeanHR          0.332 LOW — rhythm-driven

  Mean ICC across features: 0.274
  Good: Within-patient variance dominates — features capture rhythm, not patient identity.

  Segments per patient:
    min: 366, max: 2618, mean: 717, median: 598
    Top-5 patients by segment count:
      record_001: 2,618 segments
      record_110: 2,327 segments
      record_131: 1,644 segments
      record_025: 1,537 segments
      record_120: 1,503 segments
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/09_patient_dependency_icc.png

────────────────────────────────────────────────────────────────────────────────
SECTION 11: ML-READINESS CHECKS
────────────────────────────────────────────────────────────────────────────────

  NaN values in segments  : 0
  Inf values in segments  : 0
  Negative RR values      : 0
  Zero RR values          : 0
  Zero-variance windows   : 0 / 731,640

  Per-segment mean  — mean: 719.65, std: 211.26
  Per-segment std   — mean: 125.45, std: 68.26

  Suggested class weights (inverse frequency):
    SR: 0.745
    Mixed: 3.814
    AFib: 0.716

  SR/AFib imbalance ratio : 1.04
  Time-to-event monotonicity (Spearman with index): rho=-0.0011
  Post-scaling NaN: 0, Inf: 0
  Scaled distribution — skewness: -0.517, kurtosis: 10.113
  [saved] /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis/10_ml_readiness.png

════════════════════════════════════════════════════════════════════════════════
SUMMARY & RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════

  Dataset Overview:
    • 102 patients, 128 records pass filters
    • 1,589,954 total RR intervals (781,122 SR + 808,832 AFib)
    • 73,164 segments after windowing (102 patients)
    • Segment overlap: 98% — inflation factor: 717x

  Data Quality:
    • Physiological outliers : 36 (0.002%)
    • IQR-based outliers     : 71 (0.004%)
    • Combined outlier rate  : 0.004%
    • NaN/Inf in segments    : 0 / 0
    • Zero-variance windows  : 0

  Label Balance:
    • SR: 32,727 (44.7%)  |  Mixed: 6,394 (8.7%)  |  AFib: 34,043 (46.5%)
    • Imbalance ratio (SR/AFib): 1.04

  Signal Quality (SR vs AFib separability — patient-level):
    • Strongest effect sizes: pNN50 (d=2.52), MedianNN (d=2.37), pNN20 (d=2.34)

  Patient Dependency:
    • Mean ICC: 0.274
    • High ICC (>0.7, patient-driven): none
    • Low ICC  (<0.4, rhythm-driven) : MeanNN, SDNN, RMSSD, pNN50, pNN20, CVNN, MedianNN, IQRNN, MeanHR
    • Effective independent N: 102 patients (vs 73,164 segments)

  Scaling:
    • RobustScaler centers at median, scales by IQR — suitable for data with outliers
    • Post-scaling skewness: -0.517, kurtosis: 10.113

  Recommendations:
    1. Outlier rate is acceptable — no aggressive cleaning needed
    2. Label balance is reasonable
    3. No zero-variance window issues
    4. Patient dependency is moderate — patient-level splits still recommended
    5. Review the worst-quality records above for potential exclusion
    6. Consider additional artifact rejection for successive-difference outliers

================================================================================
  All plots saved to: /home/intellisense01/EML-Labs/ST-CoxNet/plots/dataset_analysis
================================================================================
```

