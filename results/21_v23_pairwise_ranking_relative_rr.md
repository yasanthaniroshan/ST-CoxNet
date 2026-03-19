# v23 — Pairwise Ranking (relative RR)

## Objective
Given two segments `(x_i, x_j)` from the same patient, learn a scalar score `s(x)`
such that `s(x_i) > s(x_j)` means `x_i` is closer to AF onset (smaller TTE).

## Training loss
- Positive label: `y_ij = 1` if `TTE_i < TTE_j` else `0`
- Logit: `s_i - s_j`
- Loss: `BCEWithLogitsLoss(logit, y_ij)` over all valid within-patient ordered pairs

## Model / files
- Script: `cpc_train_v23_pairwise_ranking_relative_rr.py`
- Checkpoint: `results` directory contains `v23_pairwise_ranking_results.json` after run

## Metrics (fill after run)
- Best val Spearman: ``
- Best val pairwise accuracy: ``
- Per-patient mean Spearman: ``

## Run results (from terminal)
- Excellent patients: `56`
- Patient split: `47 train / 9 val` (patient-level)
- RR preprocessing: `relative_rr=True`
- Early stopping: epoch `22` (best val rho)
- Train metrics at best region (~ep 5):
  - train pairwise acc = `0.677`
  - train pairwise loss = `0.5815`
- Best validation metrics:
  - Best val Spearman (score vs -TTE) = `0.187`
  - Val pairwise accuracy (at best rho checkpoint) = `~0.53`
  - Per-patient mean Spearman = `~0.07`
- Final epoch (22) metrics:
  - train pairwise acc = `0.872`
  - train pairwise loss = `0.292`
  - val pairwise acc = `0.511`
  - val pp_rho_mean = `0.034`
  - val Spearman = `0.045`

## Interpretation
- The model learns within-patient ordering well on training data (train pair_acc ~87%) but **does not generalize** to held-out patients (val pair_acc ~51–54%, barely above chance 50%).
- Best val Spearman of `0.187` is positive but weak and unstable (drops to near zero by epoch 22).
- This confirms the same pattern seen across v20–v22: the model captures patient-specific temporal structure but **cannot transfer ordering knowledge across patients**.

