## Experiment: v15 single-patient sanity check

**Goal:** Verify that the temporal signal needed for time-to-AF onset (TTE) is learnable from RR intervals in a *single patient* using one end-to-end model (no CPC, no Cox).

**Script:** `cpc_train_v15.py`

### Model
- Encoder (multi-scale ResNet) → GRU → MLP regression head
- Predicts **TTE in minutes**
- Loss: Huber / `SmoothL1Loss`

### Data
- Select **one patient** (the one with the most segments)
- Train/val split: random within that patient
- Evaluate on that patient’s SR segments

### Results (from `v15_single_patient_results.png`)
- SR segments: **MAE = 2.38 min**
- SR segments: **Spearman ρ = 0.986**
- SR segments: **R² = 0.971**

### Conclusion
This strongly suggests the RR windows contain a usable temporal signal for TTE prediction. The remaining challenge is **cross-patient generalization**, not signal absence or model incapacity.

