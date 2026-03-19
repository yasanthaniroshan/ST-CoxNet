## v17 Probe A (Seen vs Unseen, patient-level holdout)

### Setup
- Model: `TTEPredictor` (Encoder + GRU + MLP regressor), trained on excellent patients only (segment-level split within the excellent-only pool).
- Probe: patient-ID classification using a linear head trained only on a subset of *seen* excellent patients (segment split within seen patients).
- Holdout: patient-level split of excellent patients into `seen` vs `unseen`.

### Splits (excellent-only pool)
- Excellent patients: `56`
- Seen patients for probe: `44`
- Unseen patients for probe: `12`
- Seen segments: train `2186`, val `547`
- Unseen segments (evaluation): `846`

### Regressor validation (TTE)
- Final epoch `120` (from terminal log):
  - `val_MAE = 9.57` minutes
  - `val_rho = 0.778`

### Probe results
- Seen closed-set classification accuracy (linear probe):
  - `0.6673`
  - Random baseline: `0.0227`
- Unseen patient detection (AUROC, max-softmax confidence):
  - `AUROC = 0.5728`
  - Best threshold AUROC-derived binary accuracy: `0.6167`
  - Best threshold (score threshold on the confidence score used internally): `-0.9179`

### Interpretation
- The probe can strongly separate patients *within the seen set* (far above random), indicating the embeddings contain patient-distinguishing information for patients the probe was trained on.
- However, unseen detection is only marginally above chance (AUROC ~0.57), so that “patient-identity” structure does not transfer cleanly to truly held-out patients.
- This pattern is consistent with **patient memorization / over-specialization** rather than a patient-generalizable identity representation.

