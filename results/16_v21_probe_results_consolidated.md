# v21 Probe Results Consolidated (from `results/*.json`)

Probe definitions (consistent with v17 probes):
- Probe A: linear patient-ID classifier trained on *seen* patients, then evaluated for:
  - `seen_cls_acc` (closed-set over seen patients)
  - `unseen_detection_auroc` (open-set seen vs unseen)
- Probe B: open-set nearest-centroid using *seen* patients’ embeddings; evaluates:
  - `seen_closed_set_accuracy`
  - `unseen_detection_auroc`

Patient pool used in the probes:
- `tier == "excellent"` from `v15c_patient_quality.json` (56 patients total)
- Probe split: seen/unseen patients = `0.8 / 0.2` (44 seen, 12 unseen)

## Consolidated table

| Variant | ProbeA seen_cls_acc | ProbeA unseen_AUROC | ProbeB seen_acc | ProbeB unseen_AUROC |
|---------|----------------------|----------------------|-----------------|---------------------|
| v21a    | 0.3071               | 0.4055               | 0.2413          | 0.4639              |
| v21b    | 0.4004               | 0.4744               | 0.2888          | 0.4655              |
| v21c    | 0.4845               | 0.4403               | 0.3272          | 0.4815              |
| v21d    | 0.1280               | 0.4509               | 0.1938          | 0.4769              |
| v21e    | 0.2267               | 0.4075               | 0.2395          | 0.5249              |
| v21f    | 0.0987               | 0.4271               | 0.1408          | 0.4424              |

## Conclusion
- Embeddings still strongly encode patient identity on the *seen* patients (all `seen_*` accuracies are far above the random baseline `~1/44 ≈ 0.0227`).
- The *unseen* open-set separation is mostly at or below chance:
  - Probe A `unseen_detection_auroc` ranges `0.405–0.474`.
  - Probe B `unseen_detection_auroc` ranges `0.442–0.525` (only `v21e` is modestly above 0.5).
- Therefore, across GRL, MMD, and InstanceNorm variants, we did **not** achieve robust patient-invariant embeddings (as measured by held-out patient detection).

## Most important cross-check with TTE metrics
Even when TTE ranking improves slightly (best rho in your rerun was `v21c` and `v21a`), the probe results indicate that the improvement is not coming from learning a clean, transferable patient-invariant representation.

