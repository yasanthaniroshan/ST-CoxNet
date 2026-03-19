# v21 Latest Summary (excellent-only tier list updated)

## What changed
- You updated `v15c_patient_quality.json`, and now `tier == "excellent"` contains **56 patients** (instead of 10 previously).
- I re-ran the six v21 variants (v21a..v21f) using the same scripts, which apply:
  - patient-level split: **0.85 train / 0.15 val** (patient-disjoint)
  - RR preprocessing: **relative RR** for a/c/e and **raw RR** for b/d/f
  - patient-invariance diagnostics: nearest-centroid probe using train vs val patient IDs

## Final metrics (from your terminal)
- v21a (GRL + relative RR): MAE=18.40, rho=0.2709, R2=0.0713, pp_rho=0.2347; centroid unseen AUROC=0.4902
- v21b (GRL + raw RR): MAE=19.34, rho=-0.0040, R2=-0.0006, pp_rho=-0.2985; centroid unseen AUROC=0.4403
- v21c (MMD + relative RR): MAE=18.24, rho=0.3010, R2=-0.0107, pp_rho=0.2517; centroid unseen AUROC=0.5437
- v21d (MMD + raw RR): MAE=18.86, rho=-0.0930, R2=0.0211, pp_rho=-0.2063; centroid unseen AUROC=0.4716
- v21e (InstanceNorm + relative RR): MAE=19.38, rho=-0.1444, R2=-0.0043, pp_rho=-0.1444; centroid unseen AUROC=0.5710
- v21f (InstanceNorm + raw RR): MAE=19.38, rho=0.0905, R2=-0.0042, pp_rho=-0.0882; centroid unseen AUROC=0.5490

## Conclusion
1. **No method consistently achieves strong patient-invariant embeddings while also improving TTE generalization.**
   - The patient-ID centroid “unseen” AUROC stays close to chance for some variants (e.g. v21a ~0.49, v21b ~0.44, v21d ~0.47), but can drift above chance for others (v21c/v21e/v21f ~0.54–0.57).
   - This suggests the embedding geometry is not reliably de-patientized across variants.

2. **Relative-RR variants (a/c/e) are more stable for TTE ranking** than raw-RR variants (b/d/f):
   - Best rho values among these runs are v21c (rho=0.3010) and v21a (rho=0.2709).
   - Raw-RR variants b and d have near-zero or negative rho.

3. **MMD (v21c) is the strongest TTE-ranking improvement in this rerun**, but its invariance diagnostic (unseen AUROC=0.5437) is not convincingly better than chance.

## Practical next step
- If the goal is “patient-invariant embeddings”, the next experiment should tighten the invariance objective (e.g., stronger/effective adversarial training) and/or re-run the probe using the same embedding extraction protocol as the earlier v17 probes (so the diagnostic is apples-to-apples).

