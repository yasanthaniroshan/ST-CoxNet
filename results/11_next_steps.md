# Next Steps: Improving "Time-to-AFib Proximity" Prediction

## Executive Decision

**Yes, the dataset needs modification.** The current configuration is the single biggest bottleneck, and no architecture change will fix it. We need to change three things simultaneously:

1. **Run SR=2.5h first (v18)**, then extend further to **6h+** if needed (v19)
2. **Relax the AFib duration filter** from 3600s to **60s** (we only need the onset timestamp)
3. **Increase stride** from 20 to **200** (reduce the 98% overlap / 717x inflation to a healthier 80% / 5x)

These changes are all inside `CreateTemporalDataset.py` and require **zero model changes**.

---

## Current Dataset: What's Wrong

| Property | Current Value | Problem |
|----------|--------------|---------|
| SR window | **1.5 hours** | Only ~6,300 RR intervals per episode; each segment sees ~2 min of context — too short for slow-evolving pre-AFib HRV changes |
| AF filter | **af_duration >= 3,600s (1h)** | Unnecessarily strict — we only use SR segments for time-proximity; the AF window just marks onset time |
| Stride | **20** (CPC pipeline) / **100** (temporal) | 98% / 90% overlap creates 717x / 10x inflation; gives illusion of large dataset while effective N = 102 patients |
| Patients | **102** | Constrained by the combined AF + SR filters; relaxing AF filter alone adds patients |
| Segments per patient | **~250–700** | Massively correlated due to overlap; true independent information is far less |

### Why the overlap matters

With stride=20 on segment_size=1000, consecutive segments share 980/1000 RR values (98%). The model sees 26,964 "train segments" but the true information content is closer to **~500 independent observations** spread across 102 patients. This is why every model overfits catastrophically (train F1 > 90%, val F1 ~42%).

---

## Proposed New Dataset Configuration

| Property | Current | Proposed | Rationale |
|----------|---------|----------|-----------|
| SR window | 1.5h (5,400s) | **6h (v19: 21,600s)** | 4x more temporal context; captures slower pre-AFib autonomic changes |
| AF filter | >= 3,600s | **>= 60s** | Only need onset timestamp; relaxing gains +5 patients even with longer SR requirement |
| Stride | 20 / 100 | **200** | 80% overlap (5x inflation) — still smooths noise but reduces correlated duplicates by 10x |
| Segment size | 1,000 RR | **1,000 RR** | Keep unchanged — individual segment is ~2 min, proven effective for AFib/SR and CPC |
| Window size | 100 | **100** | Keep unchanged |
| nsr_before filter | >= 11,400s | **>= 27,600s** (6h + 6000s buffer) | Matches new SR window + safety margin |

### What this gives us (verified against the raw IRIDIA metadata)

| Config | Episodes | Unique Patients |
|--------|----------|-----------------|
| **Current** (af>=3600s, sr>=1.5h+buf) | 128 | 102 |
| **Proposed** (af>=60s, sr>=6h+buf) | 120 | 107 |
| Aggressive (af>=60s, sr>=8h+buf) | 109 | 98 |
| Maximum SR (af>=60s, sr>=12h+buf) | 87 | 78 |

The proposed 6h config retains **107 patients** (actually 5 more than current) while giving 4x more temporal context per episode. We lose a handful of episodes from patients whose SR recordings are shorter than 6h, but gain patients who had short AF episodes that were previously filtered out.

### Estimated segment counts

| Config | SR segments per episode | Total SR segments (est.) |
|--------|------------------------|--------------------------|
| Current (1.5h, stride=100) | ~54 | ~6,900 |
| **Proposed (6h, stride=200)** | ~122 | ~14,600 |
| Aggressive (8h, stride=200) | ~164 | ~17,900 |

More segments per patient **and** better-separated segments (80% vs 98% overlap).

---

## What Worked (keep these)

- **CPC representations for rhythm discrimination**: AFib vs SR reaches ~**92.7% macro F1** on held-out patients — strong encoder.
- **Patient-aware evaluation is mandatory**: segment-level CV inflates results catastrophically (RF: 82% leaky vs 34.6% patient-grouped).
- **Per-patient normalization unlocks most of the available signal**: lifts NN/HRV variants from near-chance (~34%) to ~**42–43%**.

## What Didn't Work (root causes)

- **Time-to-onset signal is extremely weak at 1.5h scale**:
  - HRV feature vs time correlations: |rho| < ~0.1 globally
  - CPC temporal probing shows no advantage over best HRV statistic (pNN50)
- **Cross-patient generalization is the bottleneck**:
  - Single-patient RR->TTE regression: near-perfect (rho ~0.99)
  - Global pooled regression on unseen patients: collapses (rho ~0.15)
- **Sequence models help only marginally**: Transformer/GRU trajectory +1pp at best, still low-40% on patient-held-out splits
- **Architecture is not the bottleneck**: 88K GRU vs 500K Transformer plateau at the same ceiling

---

## Follow-on Experiment Plan (v19: 6h SR Window)

### Step 1: Regenerate the dataset

Modify `CreateTemporalDataset.py` call parameters:

```python
create_temporal_dataset(
    dataset_path="...",
    afib_length=60,        # was 3600 — only need onset marker
    sr_length=6 * 3600,    # was 1.5 * 3600 — 4x more context
    stride=200,            # was 100 — healthier overlap
    window_size=100,       # unchanged
    number_of_windows_in_segment=10,  # unchanged
)
```

### Step 2: Re-run baselines with the new dataset

Run these under strict patient-unseen evaluation, keeping models identical:

| Task | Model | What we're measuring |
|------|-------|---------------------|
| **A: 3-class** (Near/Mid/Far) | Patient-norm CPC+HRV MLP (v10) | Does longer context improve time bins? |
| **B: 2-class** (Near vs Far, drop Mid) | Patient-norm CPC+HRV MLP (v10) | Does binary task improve SNR? |
| **C: 3-class trajectory** | Light GRU (v11-revised) | Does sequence modeling benefit more with longer windows? |

### Step 3: Diagnostic check on the new data

Before training, re-run the within-patient Spearman analysis (`raw_dataset_analysis.py` style) on the new 6h dataset to verify whether the temporal monotonicity signal is actually stronger at the longer timescale.

## Evaluation Protocol (unchanged)

- **Patient-unseen** splits (GroupKFold / GroupHoldout)
- **No segment-level leakage** across folds
- **Macro-F1** for classification tasks
- **Within-patient Spearman** on the new data as a diagnostic sanity check

## Success Criteria

- **Binary (Near vs Far)** macro-F1 clearly above the current 3-class ceiling (~43.5%)
- **3-class macro-F1** improves materially beyond **43.5%**
- Within-patient temporal Spearman (median |rho|) improves over the current pNN50 baseline (|rho| = 0.56)

If none of these move, the problem is not "more temporal context" — pivot to patient-adaptive inference or new modalities (ECG morphology).

## Secondary Options (only if primary doesn't move)

1. **Ordinal regression / ranking losses** instead of fixed-bin cross-entropy
2. **Patient-adaptive inference**: FiLM-style conditioning from a short "reference" SR subset
3. **Multi-modal**: integrate raw ECG morphology features (P-wave, T-wave changes)
4. **Hard negative mining** for temporal contrastive learning within patient sequences

## Concrete Implementation Checklist

1. [ ] Update `CreateTemporalDataset.py` call: `sr_length=21600`, `afib_length=60`, `stride=200`
2. [ ] Regenerate processed HDF5 files
3. [ ] Run within-patient Spearman analysis on the new 6h segments (diagnostic)
4. [ ] Re-run v10-style CPC+HRV (patient-norm) for 3-class and 2-class tasks
5. [ ] Re-run v11-GRU for 3-class trajectory task
6. [ ] Compare against current ceiling; inspect confusion matrices and per-patient performance
7. [ ] Write up results in `results/12_v19_sr_6h_experiment.md`
