## v17 Probe B (Open-set nearest-centroid, patient-level holdout)

### Setup
- Regressor: `TTEPredictor` trained on excellent patients only (segment-level split within the excellent-only pool).
- Embeddings: GRU hidden state `h` (frozen).
- Probe: nearest-centroid classifier over GRU embeddings for the subset of *seen* patients only.
- Open-set evaluation: decide whether an embedding belongs to an unseen patient based on distance to the nearest seen centroid.

### Splits (excellent-only pool)
- Excellent patients: `56`
- Seen patients for probe: `44`
- Unseen patients for probe: `12`
- Seen segments: train `2186`, val `547`
- Unseen segments (evaluation): `846`

### Regressor validation (TTE)
- Final epoch `120` (from terminal log):
  - `val_MAE = 9.83` minutes
  - `val_rho = 0.762`

### Probe results (centroid)
- Seen closed-set accuracy (nearest centroid):
  - `0.3583`
- Unseen detection:
  - `AUROC = 0.5201` (essentially chance)
  - Threshold at 95th percentile of seen distances:
    - `thr95 = 12.9566`
    - Binary accuracy = `0.3884`
    - `TPR_unseen = 0.0260`
    - `FPR_seen = 0.0512`

### Interpretation
- The nearest-centroid geometry in embedding space does **not** provide a reliable closed-set ID rule (seen accuracy is only ~0.36).
- Open-set detection is at/near chance, which suggests the embeddings are not organized in a simple distance-friendly way for “seen-vs-unseen” discrimination.
- Together with Probe A, this supports the idea that the model may encode patient-specific quirks for memorizing seen IDs, but doesn’t learn a robust, reusable identity structure that generalizes to held-out patients.

