# v26 — FiLM conditioning (one-stage)

## Method
- Uses excellent patients only (from `v15c_patient_quality.json`)
- Patient-level disjoint train/val split
- Episodic sampling from the same patient:
  - support_k RR segments
  - query_k RR segments with TTE labels
- FiLM conditioning:
  - `context = mean(h_support)`
  - `gamma,beta = FiLM_MLP(context)`
  - `h_mod = gamma * h_query + beta`
  - regression head predicts TTE

## Script
- `cpc_train_v26_film_conditional_patient_context_mean_relative_rr.py`
- Checkpoint: `v26_best.pth`
- JSON metrics: `results/v26_results.json`

## Metrics (fill after run)
- Train MAE / Val MAE: `11.32` / `18.03` (best checkpoint; early stop at epoch 25)
- Train Spearman ρ / Val Spearman ρ: `0.744` / `0.2124`
- Val R²: `-0.0147`
- Val mean per-patient Spearman ρ: `0.1293`
- Centroid probe (optional interpretability):
  - `seen_acc`:
  - `unseen_auroc`:

## Interpretation (fill after run)
- FiLM conditioning from mean support context improves the **centroid diagnostic** on unseen patients (`unseen_auroc=0.5805` vs `~0.52` for v22a), suggesting the modulation reshapes representation geometry.
- However, **TTE generalization is not better than v22a**:
  - v22a val Spearman rho `~0.2505` with val MAE `~17.58`
  - v26 val Spearman rho `0.2124` with val MAE `18.03`


