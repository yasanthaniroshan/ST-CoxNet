# v21: Patient-Invariant Embedding Experiments

## Overview

Six variants testing whether removing patient-level dependencies from GRU embeddings
improves patient-level TTE generalization (excellent-only patients, patient-level 0.85/0.15 split).

## Variant Matrix

| Variant | Invariance Method | RR Preprocessing | Script |
|---------|-------------------|------------------|--------|
| v21a | GRL adversarial | Relative RR | `cpc_train_v21a_grl_relative_rr.py` |
| v21b | GRL adversarial | Raw RR | `cpc_train_v21b_grl_raw_rr.py` |
| v21c | MMD regularization | Relative RR | `cpc_train_v21c_mmd_relative_rr.py` |
| v21d | MMD regularization | Raw RR | `cpc_train_v21d_mmd_raw_rr.py` |
| v21e | InstanceNorm on h | Relative RR | `cpc_train_v21e_instance_norm_relative_rr.py` |
| v21f | InstanceNorm on h | Raw RR | `cpc_train_v21f_instance_norm_raw_rr.py` |

## Method Descriptions

### GRL (v21a, v21b)
- Gradient Reversal Layer between GRU hidden state and a patient-ID classifier head.
- Loss: `L_tte + lambda_adv * L_patient_ce` (gradients from patient head are negated).
- Lambda ramps from 0 to 1.0 over first 30 epochs.
- Goal: encoder learns to confuse patient classifier while preserving TTE signal.

### MMD (v21c, v21d)
- RBF-kernel Maximum Mean Discrepancy penalty computed across patient groups within each batch.
- Loss: `L_tte + lambda_mmd * MMD`.
- Lambda ramps from 0 to 0.5 over first 30 epochs.
- Goal: push embedding distributions of different patients closer together.

### InstanceNorm (v21e, v21f)
- `nn.InstanceNorm1d(hidden_dim, affine=False)` applied to GRU hidden state.
- Removes per-sample mean/variance from embedding vector.
- No explicit invariance loss; normalization enforces invariance.

## End-of-Run Diagnostics (all variants)

At best checkpoint:
- Extract GRU embeddings for all train and val segments.
- Build centroids from train patients only.
- Report:
  - **Seen closed-set accuracy** (nearest-centroid among train patients)
  - **Unseen detection AUROC** (val patients treated as unseen)

## Results

*(To be filled after running each variant)*

| Variant | Val MAE (min) | Val Spearman rho | Val R2 | Val pp_rho | Centroid Seen Acc | Centroid Unseen AUROC |
|---------|---------------|------------------|--------|------------|-------------------|-----------------------|
| v21a | 18.40 | 0.2709 | 0.0713 | 0.2347 | 0.1228 | 0.4902 |
| v21b | 19.34 | -0.0040 | -0.0006 | -0.2985 | 0.0761 | 0.4403 |
| v21c | 18.24 | 0.3010 | -0.0107 | 0.2517 | 0.1762 | 0.5437 |
| v21d | 18.86 | -0.0930 | 0.0211 | -0.2063 | 0.1033 | 0.4716 |
| v21e | 19.38 | -0.1444 | -0.0043 | -0.1444 | 0.2341 | 0.5710 |
| v21f | 19.38 | 0.0905 | -0.0042 | -0.0882 | 0.3353 | 0.5490 |

## Baseline Comparison

- v20 (relative RR, patient-level split, no invariance): MAE=16.73, rho=0.29, R2=0.04
- Note: v20 was not re-run after the “excellent” patient list update (so v20 numbers may not match this new tier==excellent cohort).
