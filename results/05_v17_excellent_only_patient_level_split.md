## Experiment: v17 excellent-only (patient-level split; no leakage)

**Goal:** Test if excluding non-excellent patients fixes generalization, while keeping validation **patient-unseen**.

**Script:** `cpc_train_v17_excellent_only.py`

### Setup
- Load per-patient quality tiers from `v15c_patient_quality.json`
- Select patients with tier **excellent** (ρ >= 0.9)
- Split the excellent-patient set at the **patient level** into:
  - train patients (subset of excellent)
  - val patients (remaining excellent)
- Train on **SR segments** and validate only on SR segments from unseen val patients.

### Results (run summary from terminal log)
- Global MAE: **19.34 min**
- Global Spearman ρ: **-0.0271**
- Global R²: **-0.0009**
- Mean per-patient ρ: **-0.1657** across **9** validation patients
- Early stopping: epoch **22**

### Conclusion
Even among excellent patients, the model still **cannot generalize** to unseen patients when evaluated in a strict patient-level split setting.
This strongly suggests that **a single shared global regression function** on raw RR windows cannot capture patient-specific baseline differences without additional mechanisms (normalization, patient conditioning, relative targets, etc.).

