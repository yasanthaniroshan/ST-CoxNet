## v20 (Excellent-only patient-level split) + relative RR baseline removal

### Run
- Script: `cpc_train_v20_excellent_only_patient_level_relative_rr.py`

### Data selection
- `v15c_patient_quality.json` tier counts:
  - `excellent`: 10 / 90 patients
- Excellent-only pool:
  - Excellent patients used: `10`
  - Relative RR normalization stats:
    - `mean_mu`: `0.4988`
    - `mean_sd`: `0.2833`
    - `median_sd`: `0.2664`
    - `min_sd`: `0.1123`
    - `max_sd`: `0.6350`

### Patient-level split (no leakage)
- Train patients: `8`
- Val patients: `2`
- Train segments: `458`
- Val segments: `100`
- TTE range (excellent only): `0 — 81.0 min`

### Training behavior
- Early stopping: triggered at epoch `51`
- Best validation MAE: `16.73 min`

### Final validation metrics (best checkpoint)
- Global MAE: `16.73 min`
- Global Spearman ρ: `0.2914`
- Global R²: `0.0405`
- Mean per-patient Spearman ρ: `0.3913` (computed over `2` val patients)

### Interpretation / caution
- This run’s excellent-only validation set has only `2` patients, so metrics are very noisy and may not reflect true model capability.
- The relative-RR transform removes per-patient RR mean/std, which may also remove predictive absolute scale information (in addition to patient baseline).

