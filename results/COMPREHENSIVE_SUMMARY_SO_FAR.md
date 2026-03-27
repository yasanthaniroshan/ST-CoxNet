# ST-CoxNet Comprehensive Summary So Far

Date: 2026-03-27

## 1) Project objective

The project goal is to predict temporal proximity to AFib onset from SR RR-interval segments, with strict emphasis on patient-level generalization (no leakage across patients).

---

## 2) Data and evaluation setup used across experiments

- Dataset source: IRIDIA AFib records (RR interval series around AFib episodes).
- Typical SR window in baseline phase: 1.5h before AFib onset.
- Segment construction: 10 windows x 100 RR = 1000 RR values per segment.
- Overlap in early pipelines was high (stride 20 or 100), later reduced in planning to stride 200.
- Core evaluation principle after diagnostics: patient-level disjoint train/validation/testing (GroupKFold or explicit patient holdout).
- Main tasks explored:
  - AFib vs SR binary classification
  - time-to-event regression
  - 3-class time-bin classification (Near/Mid/Far)
  - pairwise ranking and patient-invariant representation learning.

---

## 3) What was done, chronologically

## Phase A: Foundation and first supervised tasks (v4-v11 + diagnostics)

1. CPC self-supervised pretraining was developed and debugged (v4-v8).
   - Critical fix: switched from within-batch negatives to cross-batch InfoNCE.
   - Result: CPC training stabilized; next-step prediction reached strong train/val accuracy.

2. AFib vs SR downstream classification was tested with CPC features.
   - Strong result: macro F1 around 92.7% on held-out patients.

3. Time-to-event tasks were attempted (regression and 3-class bins).
   - Regression and basic CPC-only bin models underperformed.
   - Diagnostics revealed key leakage and generalization issues.

4. Diagnostic phase identified root causes.
   - Segment-level CV gave inflated performance due to patient leakage.
   - Patient-grouped evaluation caused major metric drop.
   - Per-patient normalization consistently improved results.

5. Multi-modal and sequence-level models were tested.
   - Added HRV + CPC features and trajectory models (Transformer/GRU).
   - Gains were small, with performance plateauing in low-40% macro-F1 range.

## Phase B: Temporal-signal validation and patient-invariance line (v13-v23)

1. Temporal ranking objectives were introduced (v13/v14).
   - Within-patient ranking improved correlation signals compared to cross-patient ranking.
   - However, true cross-patient transfer remained weak.

2. Per-patient audits and split-stress tests (v15-v17).
   - Single-patient modeling can be excellent.
   - Full pooled patient-unseen generalization is poor.
   - Segment-level leakage recreates strong numbers, confirming shortcut learning.

3. Excellent-tier filtering and relative-RR preprocessing (v20-v21).
   - Best variants used relative-RR and gave modest positive Spearman.
   - GRL/MMD/InstanceNorm did not produce robust patient-invariant embeddings.
   - Probe AUROCs for unseen patient detection stayed near chance/inconsistent.

4. Pairwise ranking with patient-level split (v23).
   - Training pairwise accuracy became high.
   - Validation pairwise accuracy stayed around chance to weakly above chance.
   - Conclusion: within-patient order learned, cross-patient transfer still weak.

## Phase C: Conditional/adaptive modeling and FiLM variants (v22-v27)

1. Conditional support-query formulations were tried (v22a/v22b/v22c).
   - v22a (mean context + relative RR) was the most stable among v22.
   - Attention variants were not consistently better.

2. Two-stage and FiLM-conditioned variants explored (v25-v27).
   - v25 (two-stage GRL + contrastive then regression) produced modest rho but negative R2 and weak invariance probe.
   - v26 FiLM one-stage showed some representation-geometry improvement in probe AUROC, but no clear TTE metric breakthrough.
   - v27 template/report exists but metrics were not finalized in the checked report.

3. Dataset redesign plan was drafted (v18/v19 planning docs).
   - Move from 1.5h to 2.5h then 6h SR window.
   - Relax AF duration requirement to keep useful patients.
   - Increase stride to reduce extreme overlap/redundancy.

---

## 4) Key numeric results and outcomes

## A) Strongest positive result

- AFib vs SR classification: ~92.7% macro F1 (CPC features), indicating RR-based representations are strong for rhythm discrimination.

## B) Time-proximity prediction results (patient-held-out context)

- CPC-only 3-class time bins (v9): ~37.8% macro F1.
- CPC+HRV (global normalization, v10): ~34.0% macro F1.
- CPC+HRV (per-patient normalization, v10): ~42.8% macro F1.
- Sequence Transformer (v11): ~43.5% macro F1 (best in this family).
- Sequence GRU (v11 revised): ~41.0% macro F1.

Interpretation: improvements exist, but absolute performance stays only modestly above 3-class chance (33.3%).

## C) Leakage and patient-dependency evidence

- Random Forest segment-level CV (leaky): ~82.2%.
- Random Forest patient-grouped CV: ~34.6%.
- This large drop confirms patient-identity shortcuts dominate when leakage is not controlled.

## D) CPC temporal signal verification vs HRV baseline

From explicit CPC-vs-HRV temporal analysis:
- Best HRV feature (pNN50) within-patient temporal monotonicity outperformed all CPC dims.
- No CPC encoder/context dimensions exceeded best HRV in that run.
- Linear probes for TTE with patient-level splits had near-zero or negative R2 for CPC representations.
- C-index was slightly better for HRV than CPC combined features.

Interpretation: the selected CPC checkpoint did not surpass simple HRV statistics for temporal ordering.

## E) Invariance and conditional families (selected values)

- v21 best ranking among variants (rerun): v21c rho ~0.3010, v21a rho ~0.2709, but invariance probes remained unstable/near-chance.
- v22 summary:
  - v22a: MAE 17.58, rho 0.2505, R2 0.0558.
  - v22b/v22c did not show stable improvements.
- v25 two-stage: MAE 19.08, rho 0.1991, R2 -0.0817, unseen AUROC 0.4851.
- v26 FiLM mean-context: MAE 18.03, rho 0.2124, R2 -0.0147, per-patient rho mean 0.1293.

Interpretation: many variants produce weak-to-moderate positive rho at best, but none convincingly solve cross-patient generalization.

---

## 5) Main conclusions

1. RR-based models are good at rhythm-state discrimination (AFib vs SR) but not yet reliable for patient-unseen time-to-onset prediction.
2. The dominant bottleneck is cross-patient domain shift and baseline variability, not lack of model capacity.
3. Leakage control is non-negotiable; segment-level splits can be highly misleading.
4. Per-patient normalization is consistently helpful and should remain default in temporal-proximity experiments.
5. Patient-invariance mechanisms tested so far (GRL/MMD/InstanceNorm/contrastive variants) have not yielded robust, transferable invariance.
6. Conditional and FiLM-style adaptation improve some diagnostics but have not delivered decisive TTE gains.
7. The short-context/high-overlap data recipe is likely limiting the learnable temporal signal.

---

## 6) Practical interpretation of "where we are now"

- We have good evidence that the current setting contains weak, noisy temporal signal across unseen patients.
- We also have evidence that some signal exists within-patient and can be exploited when patient structure leaks into train/test.
- Therefore, the next meaningful gains are expected to come more from data design and adaptation strategy than from swapping model architectures.

---

## 7) Recommended next moves (based on completed work)

1. Execute the planned longer-context dataset runs (2.5h and especially 6h), with stride 200 and strict patient-level evaluation.
2. Re-run a compact baseline suite on the new dataset first (v10-style patient-norm CPC+HRV; binary Near/Far and 3-class).
3. Keep a standardized leakage-safe benchmark table to compare new runs directly to current ceiling (~43.5% macro-F1 for 3-class).
4. If longer-context still fails, prioritize patient-adaptive inference and/or additional modalities (e.g., ECG morphology) over larger RR-only architectures.

---

## 8) Bottom line

So far, the work has been thorough and has identified the central challenge clearly: robust cross-patient temporal prediction from short RR-only segments is weak under strict evaluation. The project has strong foundations (data diagnostics, leakage controls, CPC baseline quality, broad model exploration), and the highest-value path forward is now data-regime change plus patient-adaptive strategy rather than incremental architecture complexity.
