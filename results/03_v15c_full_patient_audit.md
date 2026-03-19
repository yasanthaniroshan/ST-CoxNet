## Experiment: v15c full patient sweep audit

**Goal:** Quantify how consistently the RR→TTE signal is learnable per patient, and identify patient subsets likely to be noisy/corrupted.

**Script:** `cpc_train_v15c.py`

### Protocol
- For **all eligible patients** (>= `20` SR segments), train a **separate** TTE model per patient.
- Predicts **TTE in minutes**, loss: `SmoothL1Loss`.
- Train/val split: random within the patient.
- Max epochs: **400** (with early stopping patience 20).

### Tiering rule (based on per-patient Spearman ρ)
- **excellent:** ρ >= 0.9
- **good:** 0.7 <= ρ < 0.9
- **weak:** 0.3 <= ρ < 0.7
- **poor/noisy:** ρ < 0.3

### Global summary (from the run log + W&B summary)
- Total patients tested: **87**
- Excellent: **56**
- Good: **3**
- Weak: **17**
- Poor: **11**
- Mean **ρ = 0.7687** ± 0.3481
- Mean **MAE = 8.28 min** ± 7.89
- Mean **R² = 0.5941** ± 0.537

### Key insight
The signal is present in **a majority** of patients (56/87 excellent), but **a non-trivial minority** behave like noise/outliers for direct regression.

