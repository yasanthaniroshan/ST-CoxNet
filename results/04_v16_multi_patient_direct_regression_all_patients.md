## Experiment: v16 multi-patient direct TTE regression (all patients)

**Goal:** Test cross-patient generalization with a single global end-to-end model using direct regression.

**Script:** `cpc_train_v16.py`

### Model
- Encoder → GRU → MLP regression head
- Predicts **TTE in minutes**
- Loss: `SmoothL1Loss` (Huber)

### Data
- Uses SR-only segments (`label == -1`)
- Train/val patient splits come from `CreateTemporalDataset.py` (patient-level split already baked into generated files)
- Train/val patients are **unseen at the patient level**

### Results (run summary from terminal log)
- Global MAE: **18.77 min**
- Global Spearman ρ: **0.1539**
- Global R²: **0.0262**
- Mean per-patient ρ: **0.2182** across **15** validation patients

### Conclusion
Despite strong per-patient signal (v15/v15c), the pooled model **fails to generalize to unseen patients**. This points to a **large patient-to-patient domain shift** in the RR→TTE mapping (or RR distributions), not an absence of temporal structure.

