# v22b — Conditional attention context (raw RR) + patient-level split

## Variant definition
- Conditioning: query prediction uses a **query-specific patient context** built from RR **support** segments.
- Context aggregation: **attention-weighted** sum of support embeddings using cosine similarity between query and support embeddings.
- RR preprocessing: **raw RR** (no per-patient baseline removal).
- Episode sizes: `support_k=5`, `query_k=3` (same as v22a).
- Split: patient-level (train/val disjoint) using `tier == "excellent"`.

## Metrics (fill after run)
- Best val MAE:
- Best val Spearman rho:
- Best val R2:
- Centroid diagnostic on embeddings `h`:
  - centroid_seen_acc:
  - centroid_unseen_auroc:

## Run results (from terminal)
- Excellent patients: `56`
- Patient split: `47 train / 9 val` (patient-level)
- RR preprocessing: `relative_rr=False` (raw RR)
- Early stopping: epoch `21` (best val MAE)
- Best validation metrics:
  - Val MAE = `18.45` min
  - Val Spearman rho = `-0.2263`
  - Val R2 = `-0.3717`
- Centroid probe on embeddings `h`:
  - centroid_seen_acc = `0.1207`
  - centroid_unseen_auroc = `0.5511`

