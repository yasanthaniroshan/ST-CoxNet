## Experiment: v15b per-patient training on 10 random patients

**Goal:** Determine whether the learnable temporal signal is patient-specific/outlier-driven, or broadly present across different patients.

**Script:** `cpc_train_v15b.py`

### Protocol
- Pick **10 patients randomly** among those with at least `>= 20` SR segments (seeded, reproducible)
- Train a **separate** model per patient (same architecture as v15):
  - Encoder → GRU → MLP regression head
  - Predicts **TTE in minutes**
  - Loss: `SmoothL1Loss`
- Train/val split: random **within each patient**
- Early stopping: patience 25

### Results (run log summary)
Across the 10 trained patients:
- Mean **MAE = 6.12 min** (STD 6.66)
- Mean **Spearman ρ = 0.863** (STD 0.224)
- Mean **R² = 0.709** (STD 0.501)

**Excellent-performing patients** (examples)
- PID 67: ρ = **0.995**
- PID 22: ρ = **0.994**
- PID 12: ρ = **0.992**
- PID 4: ρ = **0.991**
- PID 76: ρ = **0.970**
- etc.

**Poor/noisy patients** (examples)
- PID 26: ρ = **0.460**, R² = **-0.495**, MAE = **20.72 min**
- PID 68: ρ = **0.377**, R² = **-0.045**, MAE = **17.52 min**

### Conclusion
The temporal signal is **learnable for most patients**, but there exists a minority of patients where the mapping is difficult/noisy (for RR→TTE regression with this architecture and training).

