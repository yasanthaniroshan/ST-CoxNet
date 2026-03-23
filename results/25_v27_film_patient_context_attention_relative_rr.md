# v27 — FiLM conditioned (attention support context) one-stage

## Method
- Uses excellent patients only (`v15c_patient_quality.json`)
- Patient-level train/val split
- Episode sampling from the same patient:
  - support_k RR segments
  - query_k RR segments with TTE labels
- RR preprocessing: relative RR

## FiLM conditioning
- Encode support/query segments -> embeddings `h_support` and `h_query`
- Build **query-specific** context with cosine attention over support (v22c-style)
- FiLM generates per-query `(gamma,beta)` from context
- Modulate query embeddings: `h_mod = gamma * h_query + beta`
- Predict TTE from `h_mod` via regression head

## Script / artifacts
- Script: `cpc_train_v27_film_conditional_attention_context_relative_rr.py`
- Checkpoint: `v27_best.pth`
- JSON: `results/v27_results.json`

## Metrics (fill after run)
- Best val MAE:
- Final val Spearman ρ:
- Final val R²:
- Val mean per-patient Spearman ρ:
- Centroid probe:
  - `seen_acc`:
  - `unseen_auroc`:

## Interpretation (fill after run)
``

