"""
Raw Dataset Quality Analysis for IRIDIA AFIB Dataset
=====================================================
Loads the raw RR interval data (same pipeline as CreateDataset.py) and produces
an extensive report covering:
  1. Patient & record-level statistics
  2. RR interval quality & physiological plausibility
  3. Outlier detection (misdetected R-peaks)
  4. Distributions before / after RobustScaler
  5. Segment-level analysis (label balance, time-to-event)
  6. HRV feature comparison (SR vs AFib)
  7. Temporal dynamics approaching AFib onset
  8. Dimensionality reduction (PCA, t-SNE)
  9. ML-readiness summary

All plots are saved to  plots/dataset_analysis/
A text report is saved to plots/dataset_analysis/report.txt
"""

import os
import sys
import warnings
import json
import textwrap
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats as sp_stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — matches CreateDataset.py defaults
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
AFIB_LENGTH = 60 * 60          # 1 hour of AFib in seconds
SR_LENGTH = int(1.5 * 60 * 60) # 1.5 hours of SR in seconds
WINDOW_SIZE = 100
NUMBER_OF_WINDOWS = 10
STRIDE = 20
SEGMENT_SIZE = NUMBER_OF_WINDOWS * WINDOW_SIZE
VALIDATION_SPLIT = 0.15

PHYSIO_LOW = 200    # ms — shortest plausible RR (300 bpm)
PHYSIO_HIGH = 2500  # ms — longest plausible RR (24 bpm)
IQR_FACTOR = 3.0    # for IQR-based outlier detection

REPORT_DIR = os.path.join(os.getcwd(), "plots", "dataset_analysis")
os.makedirs(REPORT_DIR, exist_ok=True)

np.random.seed(42)
report_lines = []


def log(msg: str = ""):
    print(msg)
    report_lines.append(msg)


def save_fig(fig, name: str):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Data Loading  (mirrors CreateDataset.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_record_details(patient_list, dataset_path, afib_length, sr_length):
    records = []
    skipped = defaultdict(list)
    for patient in patient_list:
        record_dir = os.path.join(dataset_path, patient)
        ecg_csv = os.path.join(record_dir, f"{patient}_ecg_labels.csv")
        if not os.path.exists(ecg_csv):
            skipped["missing_csv"].append(patient)
            continue
        df = pd.read_csv(ecg_csv)
        for idx, row in df.iterrows():
            if row["start_file_index"] != row["end_file_index"]:
                skipped["cross_file"].append((patient, idx))
                continue
            if row["af_duration"] < afib_length:
                skipped["short_afib"].append((patient, idx))
                continue
            if row["nsr_before_duration"] < sr_length + 6000:
                skipped["short_sr"].append((patient, idx))
                continue
            records.append({
                "patient": patient,
                "record_index": idx,
                "start_file_index": row["start_file_index"],
                "end_file_index": row["end_file_index"],
                "af_duration": row["af_duration"],
                "nsr_before_duration": row["nsr_before_duration"],
            })
    return records, skipped


def load_rr_data(patient_data, dataset_path, afib_length, sr_length):
    patient = patient_data["patient"]
    record_index = patient_data["record_index"]
    record_dir = os.path.join(dataset_path, patient)
    rr_df = pd.read_csv(os.path.join(record_dir, f"{patient}_rr_labels.csv"))
    row = rr_df.loc[record_index]
    rr_start = int(row["start_rr_index"])
    rr_end = int(row["end_rr_index"])
    file_idx = int(row["start_file_index"])

    with h5py.File(os.path.join(record_dir, f"{patient}_rr_{file_idx:02d}.h5"), "r") as f:
        rr_data = f["rr"][:]

    afib_seg = rr_data[rr_start:rr_end]
    afib_cum = np.cumsum(afib_seg) / 1000
    af_idx = np.searchsorted(afib_cum, afib_length)
    afib_seg = afib_seg[:af_idx]

    sr_seg = rr_data[:rr_start][::-1]
    sr_cum = np.cumsum(sr_seg) / 1000
    sr_idx = np.searchsorted(sr_cum, sr_length)
    sr_seg = sr_seg[:sr_idx][::-1]

    combined = np.concatenate([sr_seg, afib_seg])
    afib_start_index = len(sr_seg)
    return combined, afib_start_index, sr_seg, afib_seg


def segment_rr_data(rr_data, afib_start_index, segment_size, stride, window_size):
    segments, labels, times = [], [], []
    is_time_calculated = False
    total_length = len(rr_data)
    for start in range(0, total_length - segment_size + 1, stride):
        end = start + segment_size
        segment = rr_data[start:end].reshape(-1, window_size)
        if start < afib_start_index and end <= afib_start_index:
            label = -1
        elif start < afib_start_index and end > afib_start_index:
            label = 0
        else:
            label = 1
        segments.append(segment)
        labels.append(label)
        if is_time_calculated:
            times.append(0)
        elif end >= afib_start_index and not is_time_calculated:
            time_so_far = 0
            tte = [time_so_far]
            for seg in reversed(segments[:-1]):
                time_so_far += sum(seg.flatten()[:stride])
                tte.append(time_so_far)
            times.extend(tte[::-1])
            is_time_calculated = True
    return np.array(segments), np.array(labels), np.array(times)


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Helpers
# ══════════════════════════════════════════════════════════════════════════════
def compute_hrv_features(rr_ms):
    """Lightweight HRV feature set computed on raw RR (ms)."""
    if len(rr_ms) < 4:
        return {}
    diff = np.diff(rr_ms)
    mean_nn = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else 0
    pnn50 = np.mean(np.abs(diff) > 50) * 100 if len(diff) else 0
    pnn20 = np.mean(np.abs(diff) > 20) * 100 if len(diff) else 0
    cv = sdnn / mean_nn if mean_nn > 0 else 0
    median_nn = np.median(rr_ms)
    iqr_nn = np.subtract(*np.percentile(rr_ms, [75, 25]))
    mean_hr = 60000.0 / mean_nn if mean_nn > 0 else 0
    return {
        "MeanNN": mean_nn, "SDNN": sdnn, "RMSSD": rmssd,
        "pNN50": pnn50, "pNN20": pnn20, "CVNN": cv,
        "MedianNN": median_nn, "IQRNN": iqr_nn, "MeanHR": mean_hr,
    }


def detect_outliers_iqr(rr, factor=IQR_FACTOR):
    q1, q3 = np.percentile(rr, [25, 75])
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return (rr < low) | (rr > high)


def detect_outliers_physio(rr):
    return (rr < PHYSIO_LOW) | (rr > PHYSIO_HIGH)


def detect_outliers_successive(rr, threshold_factor=0.20):
    """Flag beats where successive difference exceeds 20% of local median."""
    if len(rr) < 3:
        return np.zeros(len(rr), dtype=bool)
    diff = np.abs(np.diff(rr))
    local_med = np.median(rr)
    flags = diff > (threshold_factor * local_med)
    mask = np.zeros(len(rr), dtype=bool)
    mask[:-1] |= flags
    mask[1:] |= flags
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("  IRIDIA AFIB DATASET — RAW DATA QUALITY ANALYSIS")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    patient_list = sorted(os.listdir(DATASET_PATH))
    log(f"\nDataset path : {DATASET_PATH}")
    log(f"Total patient directories found: {len(patient_list)}")

    # ── 1  Record details ────────────────────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 1: RECORD FILTERING & PATIENT STATISTICS")
    log("─" * 80)

    records, skipped = load_record_details(
        patient_list, DATASET_PATH, AFIB_LENGTH, SR_LENGTH
    )
    unique_patients = set(r["patient"] for r in records)

    log(f"  Records that pass all filters : {len(records)}")
    log(f"  Unique patients with records  : {len(unique_patients)}")
    log(f"  Skipped — missing CSV         : {len(skipped.get('missing_csv', []))}")
    log(f"  Skipped — cross-file episode  : {len(skipped.get('cross_file', []))}")
    log(f"  Skipped — AFib too short      : {len(skipped.get('short_afib', []))}")
    log(f"  Skipped — SR too short        : {len(skipped.get('short_sr', []))}")

    records_per_patient = defaultdict(int)
    for r in records:
        records_per_patient[r["patient"]] += 1
    counts = list(records_per_patient.values())
    log(f"\n  Records per patient — min: {min(counts)}, max: {max(counts)}, "
        f"mean: {np.mean(counts):.1f}, median: {np.median(counts):.0f}")

    af_durs = [r["af_duration"] / 1000 for r in records]
    sr_durs = [r["nsr_before_duration"] / 1000 for r in records]
    log(f"\n  AF duration (s)  — min: {min(af_durs):.0f}, max: {max(af_durs):.0f}, "
        f"mean: {np.mean(af_durs):.0f}, median: {np.median(af_durs):.0f}")
    log(f"  SR duration (s)  — min: {min(sr_durs):.0f}, max: {max(sr_durs):.0f}, "
        f"mean: {np.mean(sr_durs):.0f}, median: {np.median(sr_durs):.0f}")

    # ── 2  Load raw RR data ──────────────────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 2: RAW RR INTERVAL LOADING")
    log("─" * 80)

    all_rr_raw = []           # every raw RR value across all records
    all_sr_rr = []            # SR portions only
    all_afib_rr = []          # AFib portions only
    per_record_stats = []     # per-record summary
    per_record_rr = []        # (combined, afib_start_index) per record
    per_record_sr_rr = []
    per_record_afib_rr = []

    pbar = tqdm(records, desc="Loading RR data")
    for rec in pbar:
        pbar.set_description(f"Loading {rec['patient']}")
        try:
            combined, afib_start, sr_seg, afib_seg = load_rr_data(
                rec, DATASET_PATH, AFIB_LENGTH, SR_LENGTH
            )
        except Exception as e:
            log(f"  [WARN] Failed to load {rec['patient']} idx={rec['record_index']}: {e}")
            continue

        all_rr_raw.append(combined)
        all_sr_rr.append(sr_seg)
        all_afib_rr.append(afib_seg)
        per_record_rr.append((combined, afib_start))
        per_record_sr_rr.append(sr_seg)
        per_record_afib_rr.append(afib_seg)

        per_record_stats.append({
            "patient": rec["patient"],
            "record_index": rec["record_index"],
            "n_rr_total": len(combined),
            "n_sr": len(sr_seg),
            "n_afib": len(afib_seg),
            "sr_dur_s": np.sum(sr_seg) / 1000,
            "afib_dur_s": np.sum(afib_seg) / 1000,
            "mean_rr": np.mean(combined),
            "std_rr": np.std(combined),
            "min_rr": np.min(combined),
            "max_rr": np.max(combined),
        })

    all_rr_flat = np.concatenate(all_rr_raw)
    all_sr_flat = np.concatenate(all_sr_rr)
    all_afib_flat = np.concatenate(all_afib_rr)

    log(f"\n  Successfully loaded records       : {len(per_record_stats)}")
    log(f"  Total RR intervals (all records)  : {len(all_rr_flat):,}")
    log(f"  Total SR intervals                : {len(all_sr_flat):,}")
    log(f"  Total AFib intervals              : {len(all_afib_flat):,}")
    log(f"\n  Global RR stats (ms):")
    log(f"    Mean: {np.mean(all_rr_flat):.2f}  Std: {np.std(all_rr_flat):.2f}")
    log(f"    Min : {np.min(all_rr_flat):.2f}  Max: {np.max(all_rr_flat):.2f}")
    log(f"    Median: {np.median(all_rr_flat):.2f}  IQR: {np.subtract(*np.percentile(all_rr_flat, [75, 25])):.2f}")

    # ── 3  RR quality & outlier analysis ─────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 3: RR INTERVAL QUALITY & OUTLIER DETECTION")
    log("─" * 80)

    physio_outliers = detect_outliers_physio(all_rr_flat)
    iqr_outliers = detect_outliers_iqr(all_rr_flat)
    combined_outliers = physio_outliers | iqr_outliers

    log(f"\n  Physiological-range outliers (RR<{PHYSIO_LOW} or >{PHYSIO_HIGH} ms):")
    log(f"    Count : {physio_outliers.sum():,}  ({100*physio_outliers.mean():.3f}%)")
    log(f"    Too short (<{PHYSIO_LOW} ms) : {(all_rr_flat < PHYSIO_LOW).sum():,}")
    log(f"    Too long  (>{PHYSIO_HIGH} ms): {(all_rr_flat > PHYSIO_HIGH).sum():,}")

    log(f"\n  IQR-based outliers (factor={IQR_FACTOR}):")
    q1, q3 = np.percentile(all_rr_flat, [25, 75])
    iqr = q3 - q1
    log(f"    Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    log(f"    Lower bound: {q1 - IQR_FACTOR * iqr:.1f}, Upper bound: {q3 + IQR_FACTOR * iqr:.1f}")
    log(f"    Count : {iqr_outliers.sum():,}  ({100*iqr_outliers.mean():.3f}%)")

    log(f"\n  Combined outliers (union):")
    log(f"    Count : {combined_outliers.sum():,}  ({100*combined_outliers.mean():.3f}%)")

    # Successive-difference outliers per record
    succ_outlier_counts = []
    succ_outlier_pcts = []
    for rr_arr in all_rr_raw:
        mask = detect_outliers_successive(rr_arr)
        succ_outlier_counts.append(mask.sum())
        succ_outlier_pcts.append(100 * mask.mean())
    log(f"\n  Successive-difference outliers (>20% of local median):")
    log(f"    Per-record mean count : {np.mean(succ_outlier_counts):.1f}")
    log(f"    Per-record mean %     : {np.mean(succ_outlier_pcts):.2f}%")
    log(f"    Per-record max  %     : {np.max(succ_outlier_pcts):.2f}%")

    # SR vs AFib outlier comparison
    sr_physio = detect_outliers_physio(all_sr_flat)
    af_physio = detect_outliers_physio(all_afib_flat)
    sr_iqr = detect_outliers_iqr(all_sr_flat)
    af_iqr = detect_outliers_iqr(all_afib_flat)
    log(f"\n  Outlier comparison — SR vs AFib:")
    log(f"    SR  physio outliers : {sr_physio.sum():,} ({100*sr_physio.mean():.3f}%)")
    log(f"    AFib physio outliers: {af_physio.sum():,} ({100*af_physio.mean():.3f}%)")
    log(f"    SR  IQR outliers   : {sr_iqr.sum():,} ({100*sr_iqr.mean():.3f}%)")
    log(f"    AFib IQR outliers  : {af_iqr.sum():,} ({100*af_iqr.mean():.3f}%)")

    # ── Plot 1: Outlier summary ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("RR Interval Quality & Outlier Analysis", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(all_rr_flat, bins=200, color="#3498db", alpha=0.8, edgecolor="none")
    ax.axvline(PHYSIO_LOW, color="red", ls="--", lw=1.5, label=f"Physio low ({PHYSIO_LOW}ms)")
    ax.axvline(PHYSIO_HIGH, color="red", ls="--", lw=1.5, label=f"Physio high ({PHYSIO_HIGH}ms)")
    ax.axvline(q1 - IQR_FACTOR * iqr, color="orange", ls=":", lw=1.5, label=f"IQR low")
    ax.axvline(q3 + IQR_FACTOR * iqr, color="orange", ls=":", lw=1.5, label=f"IQR high")
    ax.set_xlabel("RR Interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title("All RR Intervals — Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(3000, np.percentile(all_rr_flat, 99.9) * 1.2))

    ax = axes[0, 1]
    ax.hist(all_sr_flat, bins=150, color="#2ecc71", alpha=0.7, label="SR", density=True, edgecolor="none")
    ax.hist(all_afib_flat, bins=150, color="#e74c3c", alpha=0.7, label="AFib", density=True, edgecolor="none")
    ax.set_xlabel("RR Interval (ms)")
    ax.set_ylabel("Density")
    ax.set_title("SR vs AFib — RR Distribution")
    ax.legend()
    ax.set_xlim(0, min(2500, np.percentile(all_rr_flat, 99.5) * 1.2))

    ax = axes[0, 2]
    categories = ["Physiological", "IQR-based", "Successive\nDifference", "Combined"]
    counts_bar = [
        physio_outliers.sum(),
        iqr_outliers.sum(),
        sum(succ_outlier_counts),
        combined_outliers.sum(),
    ]
    pcts = [100 * c / len(all_rr_flat) for c in counts_bar]
    bars = ax.bar(categories, pcts, color=["#e74c3c", "#f39c12", "#9b59b6", "#2c3e50"], alpha=0.85)
    for bar, pct, cnt in zip(bars, pcts, counts_bar):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{pct:.2f}%\n({cnt:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("% of Total RR Intervals")
    ax.set_title("Outlier Rates by Method")

    ax = axes[1, 0]
    sr_diff = np.abs(np.diff(all_sr_flat))
    af_diff = np.abs(np.diff(all_afib_flat))
    clip_val = np.percentile(np.concatenate([sr_diff, af_diff]), 99)
    ax.hist(np.clip(sr_diff, 0, clip_val), bins=150, color="#2ecc71", alpha=0.7,
            label="SR", density=True, edgecolor="none")
    ax.hist(np.clip(af_diff, 0, clip_val), bins=150, color="#e74c3c", alpha=0.7,
            label="AFib", density=True, edgecolor="none")
    ax.set_xlabel("|ΔRR| (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Successive RR Differences — SR vs AFib")
    ax.legend()

    ax = axes[1, 1]
    per_rec_out_pct = []
    for rr_arr in all_rr_raw:
        mask_p = detect_outliers_physio(rr_arr)
        mask_i = detect_outliers_iqr(rr_arr)
        per_rec_out_pct.append(100 * (mask_p | mask_i).mean())
    ax.hist(per_rec_out_pct, bins=30, color="#8e44ad", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Outlier % per Record")
    ax.set_ylabel("Number of Records")
    ax.set_title("Per-Record Outlier Rate Distribution")
    ax.axvline(np.mean(per_rec_out_pct), color="red", ls="--", label=f"Mean: {np.mean(per_rec_out_pct):.2f}%")
    ax.legend()

    ax = axes[1, 2]
    box_data = [all_sr_flat[~detect_outliers_physio(all_sr_flat)],
                all_afib_flat[~detect_outliers_physio(all_afib_flat)]]
    bp = ax.boxplot(box_data, tick_labels=["SR", "AFib"], patch_artist=True,
                    widths=0.5, showfliers=False)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    ax.set_ylabel("RR Interval (ms)")
    ax.set_title("RR Box Plots (outliers removed for clarity)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "01_rr_quality_outliers.png")

    # ── 4  Segmentation & label analysis ─────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 4: SEGMENTATION & LABEL BALANCE")
    log("─" * 80)

    all_segments = []
    all_labels = []
    all_times = []
    all_patient_ids = []

    pbar = tqdm(per_record_rr, desc="Segmenting records")
    for rec_idx, (combined, afib_start) in enumerate(pbar):
        segs, labs, tms = segment_rr_data(combined, afib_start, SEGMENT_SIZE, STRIDE, WINDOW_SIZE)
        if len(segs) == 0:
            continue
        all_segments.append(segs)
        all_labels.append(labs)
        all_times.append(tms)
        all_patient_ids.append(
            np.full(len(segs), per_record_stats[rec_idx]["patient"], dtype=object)
        )

    all_segments = np.concatenate(all_segments, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_times = np.concatenate(all_times, axis=0)
    all_patient_ids = np.concatenate(all_patient_ids, axis=0)

    n_sr = (all_labels == -1).sum()
    n_mixed = (all_labels == 0).sum()
    n_afib = (all_labels == 1).sum()
    total_segs = len(all_labels)
    unique_seg_patients = np.unique(all_patient_ids)

    log(f"\n  Segment size : {SEGMENT_SIZE} RR intervals ({NUMBER_OF_WINDOWS} windows × {WINDOW_SIZE})")
    log(f"  Stride       : {STRIDE}")
    log(f"\n  Total segments : {total_segs:,}")
    log(f"    SR   (label=-1) : {n_sr:,}  ({100*n_sr/total_segs:.1f}%)")
    log(f"    Mixed (label=0) : {n_mixed:,}  ({100*n_mixed/total_segs:.1f}%)")
    log(f"    AFib (label=1)  : {n_afib:,}  ({100*n_afib/total_segs:.1f}%)")
    log(f"    Unique patients : {len(unique_seg_patients)}")

    overlap_pct = (SEGMENT_SIZE - STRIDE) / SEGMENT_SIZE * 100
    segs_per_patient = pd.Series(all_patient_ids).value_counts()
    log(f"\n  Segment overlap  : {overlap_pct:.1f}% "
        f"(stride {STRIDE} / segment {SEGMENT_SIZE})")
    log(f"  Segments per patient — min: {segs_per_patient.min()}, "
        f"max: {segs_per_patient.max()}, mean: {segs_per_patient.mean():.0f}, "
        f"median: {segs_per_patient.median():.0f}")
    effective_n = len(unique_seg_patients)
    log(f"  Effective independent N (patient-level): {effective_n}")
    log(f"  Inflation factor vs segment count: {total_segs / effective_n:.1f}x")

    max_time = np.max(all_times) if np.max(all_times) > 0 else 1
    times_norm = all_times / max_time
    sr_times_ms = all_times[all_labels == -1]
    sr_times_s = sr_times_ms / 1000

    log(f"\n  Time-to-event (SR segments only, seconds):")
    log(f"    Min : {np.min(sr_times_s):.1f}")
    log(f"    Max : {np.max(sr_times_s):.1f}")
    log(f"    Mean: {np.mean(sr_times_s):.1f}")
    log(f"    Median: {np.median(sr_times_s):.1f}")

    # ── Plot 2: Label balance & time-to-event ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Segment Analysis — Label Balance & Time-to-Event", fontsize=16, fontweight="bold")

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        [n_sr, n_mixed, n_afib],
        labels=["SR", "Mixed", "AFib"],
        colors=["#2ecc71", "#f1c40f", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    ax.set_title(f"Label Distribution (n={total_segs:,})")

    ax = axes[1]
    ax.hist(sr_times_s, bins=80, color="#3498db", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Time to AFib Onset (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Time-to-Event Distribution (SR segments)")
    ax.axvline(np.median(sr_times_s), color="red", ls="--", label=f"Median: {np.median(sr_times_s):.0f}s")
    ax.legend()

    ax = axes[2]
    sr_times_min = sr_times_s / 60
    ax.hist(sr_times_min, bins=80, color="#1abc9c", alpha=0.85, edgecolor="none", cumulative=True, density=True)
    ax.set_xlabel("Time to AFib Onset (minutes)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("CDF of Time-to-Event (SR segments)")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.7)
    ax.axhline(0.9, color="gray", ls=":", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "02_label_balance_tte.png")

    # ── 5  Distributions before / after RobustScaler ─────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 5: DISTRIBUTIONS BEFORE & AFTER ROBUSTSCALER")
    log("─" * 80)

    scaler = RobustScaler()
    segments_flat = all_segments.reshape(-1, WINDOW_SIZE)
    segments_scaled = scaler.fit_transform(segments_flat)
    segments_scaled_3d = segments_scaled.reshape(all_segments.shape)

    raw_flat = segments_flat.flatten()
    scaled_flat = segments_scaled.flatten()

    log(f"\n  Before scaling:")
    log(f"    Mean: {np.mean(raw_flat):.2f}  Std: {np.std(raw_flat):.2f}")
    log(f"    Min : {np.min(raw_flat):.2f}  Max: {np.max(raw_flat):.2f}")
    log(f"    Median: {np.median(raw_flat):.2f}")

    log(f"\n  After RobustScaler:")
    log(f"    Mean: {np.mean(scaled_flat):.4f}  Std: {np.std(scaled_flat):.4f}")
    log(f"    Min : {np.min(scaled_flat):.4f}  Max: {np.max(scaled_flat):.4f}")
    log(f"    Median: {np.median(scaled_flat):.4f}")
    log(f"    Scaler center (median per feature): mean={np.mean(scaler.center_):.2f}")
    log(f"    Scaler scale  (IQR per feature)   : mean={np.mean(scaler.scale_):.2f}")

    # Per-label distributions
    for lab, lab_name, col in [(-1, "SR", "#2ecc71"), (0, "Mixed", "#f1c40f"), (1, "AFib", "#e74c3c")]:
        mask = all_labels == lab
        if mask.sum() == 0:
            continue
        vals_raw = all_segments[mask].flatten()
        vals_scl = segments_scaled_3d[mask].flatten()
        log(f"\n  {lab_name} (n={mask.sum():,} segments):")
        log(f"    Raw   — mean: {np.mean(vals_raw):.2f}, std: {np.std(vals_raw):.2f}, "
            f"median: {np.median(vals_raw):.2f}")
        log(f"    Scaled— mean: {np.mean(vals_scl):.4f}, std: {np.std(vals_scl):.4f}, "
            f"median: {np.median(vals_scl):.4f}")

    # ── Plot 3: Before / after scaling ───────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Distributions Before & After RobustScaler", fontsize=16, fontweight="bold")

    clip_raw = np.percentile(raw_flat, [0.5, 99.5])
    clip_scl = np.percentile(scaled_flat, [0.5, 99.5])

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(np.clip(raw_flat, *clip_raw), bins=200, color="#3498db", alpha=0.8, edgecolor="none")
    ax.set_title("All RR — Raw")
    ax.set_xlabel("RR (ms)")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(np.clip(scaled_flat, *clip_scl), bins=200, color="#e67e22", alpha=0.8, edgecolor="none")
    ax.set_title("All RR — Scaled")
    ax.set_xlabel("Scaled Value")

    ax = fig.add_subplot(gs[0, 2])
    sample_idx = np.random.choice(len(segments_flat), min(5000, len(segments_flat)), replace=False)
    ax.scatter(segments_flat[sample_idx].mean(axis=1),
               segments_scaled[sample_idx].mean(axis=1),
               alpha=0.3, s=5, c="#2c3e50")
    ax.set_xlabel("Raw Window Mean (ms)")
    ax.set_ylabel("Scaled Window Mean")
    ax.set_title("Raw vs Scaled (window means)")

    ax = fig.add_subplot(gs[0, 3])
    positions_raw = [np.clip(raw_flat, *clip_raw)]
    positions_scl = [np.clip(scaled_flat, *clip_scl)]
    ax.boxplot([np.random.choice(raw_flat, 50000, replace=False)], positions=[0],
               widths=0.3, patch_artist=True,
               boxprops=dict(facecolor="#3498db", alpha=0.7), showfliers=False)
    ax2 = ax.twinx()
    ax2.boxplot([np.random.choice(scaled_flat, 50000, replace=False)], positions=[1],
                widths=0.3, patch_artist=True,
                boxprops=dict(facecolor="#e67e22", alpha=0.7), showfliers=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw", "Scaled"])
    ax.set_title("Box Plots: Raw vs Scaled")
    ax.set_ylabel("Raw (ms)")
    ax2.set_ylabel("Scaled")

    label_info = [(-1, "SR", "#2ecc71"), (0, "Mixed", "#f1c40f"), (1, "AFib", "#e74c3c")]
    for i, (lab, lab_name, col) in enumerate(label_info):
        mask = all_labels == lab
        if mask.sum() == 0:
            continue
        vals_r = all_segments[mask].flatten()
        vals_s = segments_scaled_3d[mask].flatten()

        ax = fig.add_subplot(gs[1, i])
        ax.hist(np.clip(vals_r, *clip_raw), bins=150, color=col, alpha=0.8, edgecolor="none")
        ax.set_title(f"{lab_name} — Raw")
        ax.set_xlabel("RR (ms)")

        ax = fig.add_subplot(gs[2, i])
        ax.hist(np.clip(vals_s, *clip_scl), bins=150, color=col, alpha=0.8, edgecolor="none")
        ax.set_title(f"{lab_name} — Scaled")
        ax.set_xlabel("Scaled Value")

    # Q-Q plots
    ax = fig.add_subplot(gs[1, 3])
    subsample = np.random.choice(raw_flat, min(10000, len(raw_flat)), replace=False)
    sp_stats.probplot(subsample, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot — Raw RR")
    ax.get_lines()[0].set(markersize=2, alpha=0.5)

    ax = fig.add_subplot(gs[2, 3])
    subsample_s = np.random.choice(scaled_flat, min(10000, len(scaled_flat)), replace=False)
    sp_stats.probplot(subsample_s, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot — Scaled RR")
    ax.get_lines()[0].set(markersize=2, alpha=0.5)

    save_fig(fig, "03_before_after_scaling.png")

    # ── 6  HRV features — SR vs AFib (patient-level) ────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 6: HRV FEATURE COMPARISON — SR vs AFIB (PATIENT-LEVEL)")
    log("─" * 80)

    sr_indices = np.where(all_labels == -1)[0]
    af_indices = np.where(all_labels == 1)[0]

    # --- Patient-level aggregation: one representative per patient ----------
    # For each patient, sample one SR and one AFib segment to avoid overlap
    # inflation, then also compute patient-level means for comparison.
    sr_patient_ids_unique = np.unique(all_patient_ids[sr_indices])
    af_patient_ids_unique = np.unique(all_patient_ids[af_indices])
    log(f"\n  Patients contributing SR segments  : {len(sr_patient_ids_unique)}")
    log(f"  Patients contributing AFib segments: {len(af_patient_ids_unique)}")

    patient_sr_hrv = []
    for pid in tqdm(sr_patient_ids_unique, desc="Patient-level SR HRV"):
        pmask = (all_patient_ids == pid) & (all_labels == -1)
        p_indices = np.where(pmask)[0]
        n_segs = len(p_indices)
        sample_size = min(5, n_segs)
        sampled = np.random.choice(p_indices, sample_size, replace=False)
        feats_list = [compute_hrv_features(all_segments[s].flatten()) for s in sampled]
        mean_feats = pd.DataFrame(feats_list).mean().to_dict()
        mean_feats["patient"] = pid
        mean_feats["n_segments"] = n_segs
        patient_sr_hrv.append(mean_feats)

    patient_af_hrv = []
    for pid in tqdm(af_patient_ids_unique, desc="Patient-level AFib HRV"):
        pmask = (all_patient_ids == pid) & (all_labels == 1)
        p_indices = np.where(pmask)[0]
        n_segs = len(p_indices)
        sample_size = min(5, n_segs)
        sampled = np.random.choice(p_indices, sample_size, replace=False)
        feats_list = [compute_hrv_features(all_segments[s].flatten()) for s in sampled]
        mean_feats = pd.DataFrame(feats_list).mean().to_dict()
        mean_feats["patient"] = pid
        mean_feats["n_segments"] = n_segs
        patient_af_hrv.append(mean_feats)

    sr_hrv_df = pd.DataFrame(patient_sr_hrv)
    af_hrv_df = pd.DataFrame(patient_af_hrv)
    feature_names = [c for c in sr_hrv_df.columns if c not in ("patient", "n_segments")]

    log(f"\n  Patient-level HRV (one mean per patient):")
    log(f"  SR patients: {len(sr_hrv_df)}, AFib patients: {len(af_hrv_df)}")
    log(f"  Features: {', '.join(feature_names)}")

    log(f"\n  {'Feature':<12} {'SR mean':>10} {'SR std':>10} {'AF mean':>10} {'AF std':>10} "
        f"{'Patient p':>12} {'Seg p':>12} {'d':>6}")
    log(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*6}")
    effect_sizes = {}
    patient_pvals = {}
    for feat in feature_names:
        sr_vals = sr_hrv_df[feat].dropna().values
        af_vals = af_hrv_df[feat].dropna().values
        if len(sr_vals) < 2 or len(af_vals) < 2:
            continue
        _, pat_pval = sp_stats.mannwhitneyu(sr_vals, af_vals, alternative="two-sided")
        patient_pvals[feat] = pat_pval
        # Segment-level p-value for comparison (sample 1 per patient)
        sr_seg_sample = []
        for pid in sr_patient_ids_unique:
            idxs = np.where((all_patient_ids == pid) & (all_labels == -1))[0]
            sr_seg_sample.append(all_segments[np.random.choice(idxs)].flatten())
        af_seg_sample = []
        for pid in af_patient_ids_unique:
            idxs = np.where((all_patient_ids == pid) & (all_labels == 1))[0]
            af_seg_sample.append(all_segments[np.random.choice(idxs)].flatten())
        sr_seg_feat = [compute_hrv_features(s)[feat] for s in sr_seg_sample]
        af_seg_feat = [compute_hrv_features(s)[feat] for s in af_seg_sample]
        _, seg_pval = sp_stats.mannwhitneyu(sr_seg_feat, af_seg_feat, alternative="two-sided")

        pooled_std = np.sqrt((np.var(sr_vals) + np.var(af_vals)) / 2)
        cohens_d = abs(np.mean(sr_vals) - np.mean(af_vals)) / pooled_std if pooled_std > 0 else 0
        effect_sizes[feat] = cohens_d
        sig = "***" if pat_pval < 0.001 else "**" if pat_pval < 0.01 else "*" if pat_pval < 0.05 else "ns"
        log(f"  {feat:<12} {np.mean(sr_vals):>10.2f} {np.std(sr_vals):>10.2f} "
            f"{np.mean(af_vals):>10.2f} {np.std(af_vals):>10.2f} "
            f"{pat_pval:>12.2e} {seg_pval:>12.2e} {cohens_d:>6.2f} {sig}")

    # ── Plot 4: HRV violin plots (patient-level) ─────────────────────────────
    n_feats = len(feature_names)
    fig, axes = plt.subplots(2, (n_feats + 1) // 2, figsize=(22, 10))
    fig.suptitle("HRV Feature Distributions — SR vs AFib (Patient-Level Means)",
                 fontsize=16, fontweight="bold")
    axes_flat = axes.flatten()

    for i, feat in enumerate(feature_names):
        ax = axes_flat[i]
        sr_v = sr_hrv_df[feat].dropna().values
        af_v = af_hrv_df[feat].dropna().values
        parts = ax.violinplot([sr_v, af_v], positions=[0, 1], showmeans=True, showmedians=True)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#2ecc71", "#e74c3c"][j])
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SR", "AFib"])
        d_str = f"d={effect_sizes.get(feat, 0):.2f}"
        p_str = f"p={patient_pvals.get(feat, 1):.1e}"
        ax.set_title(f"{feat}\n({d_str}, {p_str})", fontsize=10)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "04_hrv_violin_sr_vs_afib.png")

    # ── 7  Temporal dynamics — within-patient correlations ───────────────────
    log("\n" + "─" * 80)
    log("SECTION 7: TEMPORAL DYNAMICS — WITHIN-PATIENT CORRELATIONS")
    log("─" * 80)

    sr_mask = all_labels == -1
    sr_segs = all_segments[sr_mask]
    sr_tms = all_times[sr_mask] / 1000
    sr_pids = all_patient_ids[sr_mask]

    MIN_SEGS_FOR_CORR = 10
    within_patient_rhos = {feat: [] for feat in feature_names}
    within_patient_pvals = {feat: [] for feat in feature_names}
    patients_analyzed = 0

    for pid in tqdm(np.unique(sr_pids), desc="Within-patient Spearman"):
        pmask = sr_pids == pid
        p_segs = sr_segs[pmask]
        p_times = sr_tms[pmask]
        if len(p_segs) < MIN_SEGS_FOR_CORR:
            continue
        # Sub-sample to reduce overlap: take every Nth segment
        step = max(1, SEGMENT_SIZE // STRIDE // 2)
        sub_idx = np.arange(0, len(p_segs), step)
        p_segs_sub = p_segs[sub_idx]
        p_times_sub = p_times[sub_idx]
        if len(p_segs_sub) < MIN_SEGS_FOR_CORR:
            continue
        patients_analyzed += 1
        for feat in feature_names:
            feat_vals = np.array([compute_hrv_features(s.flatten()).get(feat, np.nan) for s in p_segs_sub])
            valid = ~(np.isnan(feat_vals) | np.isnan(p_times_sub))
            if valid.sum() < MIN_SEGS_FOR_CORR:
                continue
            rho, pval = sp_stats.spearmanr(feat_vals[valid], p_times_sub[valid])
            within_patient_rhos[feat].append(rho)
            within_patient_pvals[feat].append(pval)

    log(f"\n  Within-patient Spearman correlations (feature vs time-to-event)")
    log(f"  Patients analyzed: {patients_analyzed} (min {MIN_SEGS_FOR_CORR} SR segments required)")
    log(f"  Sub-sampling step: every {max(1, SEGMENT_SIZE // STRIDE // 2)} segments to reduce overlap")
    log(f"\n  {'Feature':<12} {'median ρ':>10} {'mean ρ':>10} {'% sig':>8} {'n_pts':>7}")
    log(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*7}")
    spearman_results = {}
    for feat in feature_names:
        rhos = np.array(within_patient_rhos[feat])
        pvals = np.array(within_patient_pvals[feat])
        if len(rhos) == 0:
            spearman_results[feat] = (0, 1)
            continue
        median_rho = np.nanmedian(rhos)
        mean_rho = np.nanmean(rhos)
        pct_sig = 100 * np.mean(pvals < 0.05)
        spearman_results[feat] = (median_rho, np.nan)
        log(f"  {feat:<12} {median_rho:>10.4f} {mean_rho:>10.4f} {pct_sig:>7.1f}% {len(rhos):>7}")

    # ── Plot 5: Within-patient temporal trends ────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle("Within-Patient HRV Temporal Dynamics\n"
                 "(per-patient Spearman ρ distributions & per-patient trend lines)",
                 fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for i, feat in enumerate(feature_names):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        rhos = np.array(within_patient_rhos[feat])
        if len(rhos) == 0:
            ax.set_visible(False)
            continue

        ax.hist(rhos, bins=25, color="#3498db", alpha=0.8, edgecolor="none", density=True)
        ax.axvline(0, color="gray", ls=":", lw=1)
        median_rho = np.nanmedian(rhos)
        ax.axvline(median_rho, color="red", ls="--", lw=2,
                   label=f"median ρ={median_rho:.3f}")
        pct_neg = 100 * np.mean(rhos < 0)
        pct_sig = 100 * np.mean(np.array(within_patient_pvals[feat]) < 0.05)
        ax.set_title(f"{feat}\n({pct_sig:.0f}% significant, {pct_neg:.0f}% negative)", fontsize=10)
        ax.set_xlabel("Within-patient Spearman ρ")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, "05_temporal_dynamics.png")

    # ── 8  Per-record outlier deep dive & example traces ─────────────────────
    log("\n" + "─" * 80)
    log("SECTION 8: PER-RECORD OUTLIER DEEP DIVE & EXAMPLE RR TRACES")
    log("─" * 80)

    worst_records = sorted(range(len(per_record_stats)),
                           key=lambda i: per_rec_out_pct[i], reverse=True)[:5]
    best_records = sorted(range(len(per_record_stats)),
                          key=lambda i: per_rec_out_pct[i])[:5]

    log(f"\n  Top 5 worst records (highest outlier %):")
    for rank, idx in enumerate(worst_records):
        s = per_record_stats[idx]
        log(f"    {rank+1}. {s['patient']} rec#{s['record_index']} — "
            f"outlier: {per_rec_out_pct[idx]:.2f}%, n_rr: {s['n_rr_total']:,}, "
            f"mean: {s['mean_rr']:.1f}, std: {s['std_rr']:.1f}")

    log(f"\n  Top 5 cleanest records (lowest outlier %):")
    for rank, idx in enumerate(best_records):
        s = per_record_stats[idx]
        log(f"    {rank+1}. {s['patient']} rec#{s['record_index']} — "
            f"outlier: {per_rec_out_pct[idx]:.2f}%, n_rr: {s['n_rr_total']:,}, "
            f"mean: {s['mean_rr']:.1f}, std: {s['std_rr']:.1f}")

    # ── Plot 6: Example RR traces (worst + best) ────────────────────────────
    n_examples = min(3, len(worst_records), len(best_records))
    fig, axes = plt.subplots(n_examples, 2, figsize=(22, 4 * n_examples))
    fig.suptitle("Example RR Traces — Worst vs Best Quality Records", fontsize=16, fontweight="bold")
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_examples):
        for col, rec_list, title_prefix in [(0, worst_records, "WORST"), (1, best_records, "BEST")]:
            idx = rec_list[row]
            rr = all_rr_raw[idx]
            s = per_record_stats[idx]
            ax = axes[row, col]

            show_len = min(3000, len(rr))
            rr_show = rr[:show_len]
            x = np.arange(show_len)
            outlier_mask = detect_outliers_physio(rr_show) | detect_outliers_iqr(rr_show)

            ax.plot(x[~outlier_mask], rr_show[~outlier_mask], ".", color="#3498db",
                    markersize=1.5, alpha=0.6, label="Normal")
            ax.plot(x[outlier_mask], rr_show[outlier_mask], ".", color="#e74c3c",
                    markersize=4, alpha=0.9, label="Outlier")
            ax.set_title(f"{title_prefix} #{row+1}: {s['patient']} "
                         f"(outlier: {per_rec_out_pct[idx]:.2f}%)", fontsize=10)
            ax.set_xlabel("Beat index")
            ax.set_ylabel("RR (ms)")
            ax.legend(fontsize=8, markerscale=3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "06_example_rr_traces.png")

    # ── 9  Dimensionality reduction (PCA + t-SNE, label & patient colored) ──
    log("\n" + "─" * 80)
    log("SECTION 9: DIMENSIONALITY REDUCTION (PCA & t-SNE)")
    log("─" * 80)

    n_dr = min(4000, len(all_segments))
    dr_idx = np.random.choice(len(all_segments), n_dr, replace=False)
    dr_feats = []
    for idx in tqdm(dr_idx, desc="Computing features for DR"):
        seg = all_segments[idx].flatten()
        f = compute_hrv_features(seg)
        dr_feats.append(list(f.values()))
    dr_matrix = np.array(dr_feats)
    dr_labels = all_labels[dr_idx]
    dr_pids = all_patient_ids[dr_idx]

    valid_mask = ~(np.isnan(dr_matrix).any(axis=1) | np.isinf(dr_matrix).any(axis=1))
    dr_matrix = dr_matrix[valid_mask]
    dr_labels = dr_labels[valid_mask]
    dr_pids = dr_pids[valid_mask]

    dr_scaler = RobustScaler()
    dr_scaled = dr_scaler.fit_transform(dr_matrix)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(dr_scaled)

    log(f"\n  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
        f"total={sum(pca.explained_variance_ratio_[:2]):.3f}")

    pca_full = PCA(random_state=42)
    pca_full.fit(dr_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = np.searchsorted(cum_var, 0.95) + 1
    log(f"  Components for 95% variance: {n_95} / {dr_scaled.shape[1]}")

    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(dr_scaled) // 4))
        tsne_coords = tsne.fit_transform(dr_scaled)
        has_tsne = True
        log(f"  t-SNE converged (KL divergence: {tsne.kl_divergence_:.4f})")
    except Exception as e:
        has_tsne = False
        log(f"  [WARN] t-SNE failed: {e}")

    # Assign numeric patient IDs for coloring
    unique_dr_pids = np.unique(dr_pids)
    pid_to_num = {pid: i for i, pid in enumerate(unique_dr_pids)}
    dr_pid_nums = np.array([pid_to_num[p] for p in dr_pids])
    log(f"  Unique patients in DR sample: {len(unique_dr_pids)}")

    color_map = {-1: "#2ecc71", 0: "#f1c40f", 1: "#e74c3c"}
    label_names = {-1: "SR", 0: "Mixed", 1: "AFib"}

    # --- Plot 7a: Label-colored -----------------------------------------------
    ncols = 3 if has_tsne else 2
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    fig.suptitle("Dimensionality Reduction — Colored by Label", fontsize=16, fontweight="bold")

    ax = axes[0]
    for lab in [-1, 0, 1]:
        mask = dr_labels == lab
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], c=color_map[lab],
                   label=label_names[lab], alpha=0.5, s=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — by Label")
    ax.legend()

    if has_tsne:
        ax = axes[1]
        for lab in [-1, 0, 1]:
            mask = dr_labels == lab
            ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], c=color_map[lab],
                       label=label_names[lab], alpha=0.5, s=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE — by Label")
        ax.legend()

    ax = axes[-1]
    ax.bar(range(len(pca_full.explained_variance_ratio_)),
           pca_full.explained_variance_ratio_, color="#3498db", alpha=0.7, label="Individual")
    ax.plot(range(len(cum_var)), cum_var, "r-o", markersize=4, label="Cumulative")
    ax.axhline(0.95, color="gray", ls="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "07a_dr_by_label.png")

    # --- Plot 7b: Patient-colored ---------------------------------------------
    patient_cmap = plt.colormaps.get_cmap("tab20").resampled(min(20, len(unique_dr_pids)))
    ncols_p = 2 if has_tsne else 1
    fig, axes = plt.subplots(1, ncols_p, figsize=(10 * ncols_p, 8))
    if ncols_p == 1:
        axes = [axes]
    fig.suptitle("Dimensionality Reduction — Colored by Patient\n"
                 "(tight patient clusters = patient dependency dominates)",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                    c=dr_pid_nums, cmap=patient_cmap, alpha=0.5, s=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — by Patient")

    if has_tsne:
        ax = axes[1]
        sc = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                        c=dr_pid_nums, cmap=patient_cmap, alpha=0.5, s=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE — by Patient")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "07b_dr_by_patient.png")

    # ── 10  Between-patient vs within-patient variance (ICC) ────────────────
    log("\n" + "─" * 80)
    log("SECTION 10: PATIENT DEPENDENCY — ICC & VARIANCE DECOMPOSITION")
    log("─" * 80)

    # Compute ICC(1) for each HRV feature using the DR sample
    # ICC(1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    # where k = average group size
    icc_values = {}
    dr_pid_series = pd.Series(dr_pids)
    for fi, feat in enumerate(feature_names):
        feat_vals = dr_matrix[:, fi]
        groups = dr_pid_series.values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        if n_groups < 2:
            icc_values[feat] = 0.0
            continue
        grand_mean = np.mean(feat_vals)
        group_means = []
        group_sizes = []
        ss_within = 0.0
        for g in unique_groups:
            g_mask = groups == g
            g_vals = feat_vals[g_mask]
            g_mean = np.mean(g_vals)
            group_means.append(g_mean)
            group_sizes.append(len(g_vals))
            ss_within += np.sum((g_vals - g_mean) ** 2)
        group_means = np.array(group_means)
        group_sizes = np.array(group_sizes)
        ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
        df_between = n_groups - 1
        df_within = len(feat_vals) - n_groups
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 1e-10
        k = np.mean(group_sizes)
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        icc = max(0.0, min(1.0, icc))
        icc_values[feat] = icc

    log(f"\n  Intraclass Correlation Coefficient (ICC) per HRV feature")
    log(f"  ICC near 1.0 = variance dominated by patient identity")
    log(f"  ICC near 0.0 = variance dominated by within-patient differences")
    log(f"\n  {'Feature':<12} {'ICC':>8} {'Interpretation'}")
    log(f"  {'─'*12} {'─'*8} {'─'*20}")
    for feat in feature_names:
        icc = icc_values[feat]
        interp = ("LOW — rhythm-driven" if icc < 0.4
                  else "MODERATE" if icc < 0.7
                  else "HIGH — patient-driven")
        log(f"  {feat:<12} {icc:>8.3f} {interp}")

    mean_icc = np.mean(list(icc_values.values()))
    log(f"\n  Mean ICC across features: {mean_icc:.3f}")
    if mean_icc > 0.5:
        log(f"  WARNING: Patient identity explains >{mean_icc*100:.0f}% of HRV variance on average.")
        log(f"           Strict patient-level train/val/test splits are essential.")
    else:
        log(f"  Good: Within-patient variance dominates — features capture rhythm, not patient identity.")

    # Segments per patient distribution
    seg_counts = pd.Series(all_patient_ids).value_counts()
    log(f"\n  Segments per patient:")
    log(f"    min: {seg_counts.min()}, max: {seg_counts.max()}, "
        f"mean: {seg_counts.mean():.0f}, median: {seg_counts.median():.0f}")
    log(f"    Top-5 patients by segment count:")
    for pid, cnt in seg_counts.head(5).items():
        log(f"      {pid}: {cnt:,} segments")

    # ── Plot 9: ICC bar chart and segment distribution ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Patient Dependency Analysis — ICC & Segment Distribution",
                 fontsize=16, fontweight="bold")

    ax = axes[0]
    feat_list = list(icc_values.keys())
    icc_vals = [icc_values[f] for f in feat_list]
    bar_colors = ["#e74c3c" if v > 0.7 else "#f39c12" if v > 0.4 else "#2ecc71" for v in icc_vals]
    bars = ax.barh(range(len(feat_list)), icc_vals, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(feat_list)))
    ax.set_yticklabels(feat_list, fontsize=10)
    ax.set_xlabel("ICC (Intraclass Correlation)")
    ax.set_title("ICC per HRV Feature")
    ax.axvline(0.4, color="gray", ls=":", alpha=0.6, label="Low/Moderate boundary")
    ax.axvline(0.7, color="gray", ls="--", alpha=0.6, label="Moderate/High boundary")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    ax = axes[1]
    ax.hist(seg_counts.values, bins=30, color="#8e44ad", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Segments per Patient")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Patient Segment Count Distribution")
    ax.axvline(seg_counts.mean(), color="red", ls="--",
               label=f"Mean: {seg_counts.mean():.0f}")
    ax.legend()

    ax = axes[2]
    var_between = np.array([icc_values[f] for f in feat_list])
    var_within = 1 - var_between
    x = np.arange(len(feat_list))
    ax.bar(x, var_between, color="#e74c3c", alpha=0.8, label="Between-patient")
    ax.bar(x, var_within, bottom=var_between, color="#2ecc71", alpha=0.8, label="Within-patient")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Proportion of Variance")
    ax.set_title("Variance Decomposition")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "09_patient_dependency_icc.png")

    # ── 11  Additional ML-readiness checks ───────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 11: ML-READINESS CHECKS")
    log("─" * 80)

    nan_count = np.isnan(all_segments).sum()
    inf_count = np.isinf(all_segments).sum()
    log(f"\n  NaN values in segments  : {nan_count}")
    log(f"  Inf values in segments  : {inf_count}")

    neg_count = (all_segments < 0).sum()
    zero_count = (all_segments == 0).sum()
    log(f"  Negative RR values      : {neg_count}")
    log(f"  Zero RR values          : {zero_count}")

    per_window_std = all_segments.reshape(-1, WINDOW_SIZE).std(axis=1)
    zero_var_windows = (per_window_std < 1e-6).sum()
    log(f"  Zero-variance windows   : {zero_var_windows} / {len(per_window_std):,}")

    per_seg_mean = all_segments.reshape(len(all_segments), -1).mean(axis=1)
    per_seg_std = all_segments.reshape(len(all_segments), -1).std(axis=1)
    log(f"\n  Per-segment mean  — mean: {np.mean(per_seg_mean):.2f}, std: {np.std(per_seg_mean):.2f}")
    log(f"  Per-segment std   — mean: {np.mean(per_seg_std):.2f}, std: {np.std(per_seg_std):.2f}")

    if total_segs > 0:
        class_weights = {
            "SR": total_segs / (3 * n_sr) if n_sr > 0 else 0,
            "Mixed": total_segs / (3 * n_mixed) if n_mixed > 0 else 0,
            "AFib": total_segs / (3 * n_afib) if n_afib > 0 else 0,
        }
        log(f"\n  Suggested class weights (inverse frequency):")
        for k, v in class_weights.items():
            log(f"    {k}: {v:.3f}")

    imbalance_ratio = max(n_sr, n_afib) / min(n_sr, n_afib) if min(n_sr, n_afib) > 0 else float("inf")
    log(f"\n  SR/AFib imbalance ratio : {imbalance_ratio:.2f}")

    time_label_corr = sp_stats.spearmanr(all_times[all_labels == -1],
                                          np.arange((all_labels == -1).sum()),
                                          nan_policy="omit")
    log(f"  Time-to-event monotonicity (Spearman with index): rho={time_label_corr.correlation:.4f}")

    scaled_nan = np.isnan(segments_scaled).sum()
    scaled_inf = np.isinf(segments_scaled).sum()
    log(f"\n  Post-scaling NaN: {scaled_nan}, Inf: {scaled_inf}")

    kurtosis = sp_stats.kurtosis(scaled_flat[:100000])
    skewness = sp_stats.skew(scaled_flat[:100000])
    log(f"  Scaled distribution — skewness: {skewness:.3f}, kurtosis: {kurtosis:.3f}")

    # ── Plot 7: ML-readiness dashboard ───────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("ML-Readiness Checks", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(per_seg_mean, bins=100, color="#3498db", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Segment Mean RR (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Segment Mean Distribution")

    ax = axes[0, 1]
    ax.hist(per_seg_std, bins=100, color="#e67e22", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Segment Std RR (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Segment Std Distribution")

    ax = axes[0, 2]
    ax.hist(per_window_std, bins=100, color="#9b59b6", alpha=0.8, edgecolor="none")
    ax.axvline(1e-6, color="red", ls="--", label="Zero-var threshold")
    ax.set_xlabel("Window Std (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Window Std Distribution")
    ax.legend()

    ax = axes[1, 0]
    corr_matrix = np.corrcoef(dr_scaled.T)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("HRV Feature Correlation Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    window_positions = np.arange(NUMBER_OF_WINDOWS)
    sr_window_means = all_segments[all_labels == -1].mean(axis=0)
    af_window_means = all_segments[all_labels == 1].mean(axis=0)
    sr_per_win = [np.mean(row) for row in sr_window_means]
    af_per_win = [np.mean(row) for row in af_window_means]
    ax.plot(window_positions, sr_per_win, "o-", color="#2ecc71", label="SR", lw=2)
    ax.plot(window_positions, af_per_win, "s-", color="#e74c3c", label="AFib", lw=2)
    ax.set_xlabel("Window Position in Segment")
    ax.set_ylabel("Mean RR (ms)")
    ax.set_title("Mean RR by Window Position")
    ax.legend()

    ax = axes[1, 2]
    checks = {
        "No NaN": nan_count == 0,
        "No Inf": inf_count == 0,
        "No Negatives": neg_count == 0,
        "No Zero RR": zero_count == 0,
        f"Low Zero-Var\n(<1%)": zero_var_windows < 0.01 * len(per_window_std),
        f"Balanced\n(ratio<3)": imbalance_ratio < 3,
        f"Low Outlier\n(<5%)": np.mean(per_rec_out_pct) < 5,
        "Post-Scale\nClean": scaled_nan == 0 and scaled_inf == 0,
    }
    check_names = list(checks.keys())
    check_vals = list(checks.values())
    bar_colors = ["#2ecc71" if v else "#e74c3c" for v in check_vals]
    bars = ax.barh(range(len(check_names)), [1] * len(check_names), color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(check_names)))
    ax.set_yticklabels(check_names, fontsize=9)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([])
    for i, (name, val) in enumerate(zip(check_names, check_vals)):
        ax.text(1.05, i, "PASS ✓" if val else "FAIL ✗", va="center", fontsize=11,
                color="#2ecc71" if val else "#e74c3c", fontweight="bold")
    ax.set_title("ML-Readiness Checklist")
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "10_ml_readiness.png")

    # ── 12  Final summary ────────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("SUMMARY & RECOMMENDATIONS")
    log("═" * 80)

    high_icc_feats = [f for f in feature_names if icc_values.get(f, 0) > 0.7]
    low_icc_feats = [f for f in feature_names if icc_values.get(f, 0) < 0.4]

    log(f"""
  Dataset Overview:
    • {len(unique_patients)} patients, {len(records)} records pass filters
    • {len(all_rr_flat):,} total RR intervals ({len(all_sr_flat):,} SR + {len(all_afib_flat):,} AFib)
    • {total_segs:,} segments after windowing ({len(unique_seg_patients)} patients)
    • Segment overlap: {overlap_pct:.0f}% — inflation factor: {total_segs / len(unique_seg_patients):.0f}x

  Data Quality:
    • Physiological outliers : {physio_outliers.sum():,} ({100*physio_outliers.mean():.3f}%)
    • IQR-based outliers     : {iqr_outliers.sum():,} ({100*iqr_outliers.mean():.3f}%)
    • Combined outlier rate  : {100*combined_outliers.mean():.3f}%
    • NaN/Inf in segments    : {nan_count} / {inf_count}
    • Zero-variance windows  : {zero_var_windows}

  Label Balance:
    • SR: {n_sr:,} ({100*n_sr/total_segs:.1f}%)  |  Mixed: {n_mixed:,} ({100*n_mixed/total_segs:.1f}%)  |  AFib: {n_afib:,} ({100*n_afib/total_segs:.1f}%)
    • Imbalance ratio (SR/AFib): {imbalance_ratio:.2f}

  Signal Quality (SR vs AFib separability — patient-level):
    • Strongest effect sizes: {', '.join(f'{f} (d={effect_sizes[f]:.2f})' for f in sorted(effect_sizes, key=lambda x: effect_sizes[x], reverse=True)[:3])}

  Patient Dependency:
    • Mean ICC: {mean_icc:.3f}
    • High ICC (>0.7, patient-driven): {', '.join(high_icc_feats) if high_icc_feats else 'none'}
    • Low ICC  (<0.4, rhythm-driven) : {', '.join(low_icc_feats) if low_icc_feats else 'none'}
    • Effective independent N: {len(unique_seg_patients)} patients (vs {total_segs:,} segments)

  Scaling:
    • RobustScaler centers at median, scales by IQR — suitable for data with outliers
    • Post-scaling skewness: {skewness:.3f}, kurtosis: {kurtosis:.3f}

  Recommendations:
    1. {'OUTLIER CLEANING NEEDED — consider clipping or removing extreme RR values' if 100*combined_outliers.mean() > 2 else 'Outlier rate is acceptable — no aggressive cleaning needed'}
    2. {'CONSIDER RESAMPLING — class imbalance may affect model training' if imbalance_ratio > 2 else 'Label balance is reasonable'}
    3. {'ZERO-VARIANCE WINDOWS DETECTED — inspect or remove constant segments' if zero_var_windows > 10 else 'No zero-variance window issues'}
    4. {'HIGH PATIENT DEPENDENCY (ICC>{mean_icc:.2f}) — ensure strict patient-level splits for train/val/test' if mean_icc > 0.5 else 'Patient dependency is moderate — patient-level splits still recommended'}
    5. Review the worst-quality records above for potential exclusion
    6. Consider additional artifact rejection for successive-difference outliers
""")

    log("=" * 80)
    log(f"  All plots saved to: {REPORT_DIR}")
    log("=" * 80)

    report_path = os.path.join(REPORT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[DONE] Full report saved to {report_path}")


if __name__ == "__main__":
    main()
