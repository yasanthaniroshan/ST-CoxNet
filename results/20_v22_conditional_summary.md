# v22 Conditional Modeling Summary (v22a/v22b/v22c)

### Setup (common across v22 variants)
- Patient pool: `tier == "excellent"` from `v15c_patient_quality.json` (56 patients)
- Split: patient-level disjoint train/val (`47 train / 9 val`, i.e., 0.85/0.15)
- Conditioning episode: support segments from the SAME patient episode, query segments predicted from conditioned representation
- RR preprocessing:
  - `relative RR` in v22a and v22c
  - `raw RR` in v22b

### Results
| Variant | Val MAE (min) | Val Spearman rho | Val R2 | Centroid unseen AUROC |
|---------|-----------------|------------------|--------|-------------------------|
| v22a (mean+relative RR) | 17.58 | 0.2505 | 0.0558 | 0.5204 |
| v22b (attention+raw RR) | 18.45 | -0.2263 | -0.3717 | 0.5511 |
| v22c (attention+relative RR) | 17.47 | 0.1264 | -0.2751 | 0.4951 |

### Interpretation
- Conditional modeling is **not consistently improving** patient-held-out prediction.
- **v22a** is the best behaved run here: it achieves **positive rho (~0.25)** and the best R² among the three variants (though still small).
- Switching from mean context to attention context (v22b/v22c) did **not** reliably help TTE ranking; v22b even produced **negative rho**.
- Centroid-based probing for patient identity/generalization (unseen AUROC) stays **near chance** across v22a–v22c, suggesting embeddings are still not producing a robust, separable patient-structure in the probe metric.

### Bottom line
- If we compare against the invariant-training family (v20/v21), the conditional strategy helps only marginally and **does not provide a stable win**.
- The next experiment should isolate which component drives the weak signal (RR mode vs context aggregation vs episode sampling sizes).

