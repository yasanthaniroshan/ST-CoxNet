# Experiment v18: Time-Proximity with 2.5h SR Window

## Goal
Check whether increasing the SR context length from **1.5 hours** to **2.5 hours** improves prediction of temporal proximity to AFib onset under the same **strict patient-unseen** evaluation setup used in v10/v11.

This is the first “extended SR window” sanity check before going to 6h+.

## Why this change
All prior experiments indicate the learned signal for time-to-onset is weak at short RR context lengths and does not generalize well across patients. If the temporal substrate evolves on a slower timescale (hours rather than minutes), a 2.5h window should be the smallest meaningful step up from 1.5h.

## Dataset change (CreateTemporalDataset.py)

### Parameters to set
Use the same dataset recipe as v17/v14-style temporal datasets, but with SR length bumped to 2.5h.

```python
create_temporal_dataset(
    dataset_path="/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1",
    afib_length=60,          # was 3600s; keep onset requirement minimal
    sr_length=int(2.5*60*60),# 2.5h SR window
    number_of_windows_in_segment=10,
    stride=200,              # was 100; reduces extreme redundancy
    window_size=100,         # RR intervals per window (unchanged)
)
```

### Derived constants
- `window_size = 100` RR
- `number_of_windows_in_segment = 10`
- `segment_size = 10 * 100 = 1000` RR intervals (same segment length as current pipeline)
- Temporal overlap implied by stride:
  - stride=200 => ~80% overlap (previous: stride=100 => ~90% overlap)

### Expected cohort size (from raw metadata, same-file episodes)
Computed with the current “same_file” constraint and the CreateTemporalDataset-style SR requirement of `nsr_before_duration >= sr_length + 6000`.

- Current baseline (as in CreateTemporalDataset.py):
  - `afib_length >= 3600s`, `sr_length = 1.5h` => `nsr_before >= 11400s`
  - **128 episodes**, **102 patients**
- v18 proposed:
  - `afib_length >= 60s`, `sr_length = 2.5h` => `nsr_before >= 15000s`
  - **164 episodes**, **127 patients**

## Training tasks (run on the new dataset)

### Task A: 3-class time bins (Near / Mid / Far)
Use time bins computed as position within the new SR window (normalized time-to-onset using the dataset’s `max_tte_seconds`).

Model:
- v10-style **CPC + HRV MLP** with **per-patient normalization**.

### Task B: 2-class time bins (Near vs Far; drop Mid)
Same labels as Task A but merge Near/Mid into Near (or recompute bins as Near/Far only), and evaluate as 2-class to increase SNR.

Model:
- same v10-style patient-normalized CPC+HRV MLP.

### Task C (optional / compute-controlled): 3-class sequence baseline
- v11-revised **light GRU trajectory** with patient normalization.

Rationale: sequence models only helped marginally at 1.5h, but this test checks whether longer context makes the trajectory signal more learnable.

## Strict evaluation protocol (must match existing work)
- Patient-unseen splits only (`GroupKFold` / patient-level holdout).
- No segment-level leakage across folds.
- Report:
  - Task A/B: **macro-F1** (and balanced accuracy for Task B if available).
  - Within-patient temporal diagnostic (pre-training sanity check): re-run within-patient HRV-vs-time monotonicity analysis using the new SR window.

## Success criteria
Declare v18 successful if *any* of the following happen:
- Task B (Near vs Far) macro-F1 shows a clear improvement over the prior v10/v11 ceilings (previous 3-class best ~43.5% macro-F1; binary is expected to be higher but must be patient-unseen).
- Task A (3-class) macro-F1 improves materially beyond the current low-40% ceiling.
- Within-patient temporal monotonicity (diagnostic Spearman |rho|) becomes stronger at the longer SR timescale (not just statistically significant, but meaningfully larger magnitude).

## If it doesn’t move the needle
If v18 fails to improve under strict patient-unseen evaluation, then the bottleneck likely isn’t just SR truncation. Next pivot would be:
- a further jump to 6h+,
- or switching to patient-adaptive inference / conditioning,
- and/or adding additional modalities (ECG morphology) if available.

## Confirmed Settings
- `stride=200` (consistent with the stride-reduction plan)

