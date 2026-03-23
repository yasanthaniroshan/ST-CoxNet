"""
IRIDIA Pre-AFib + AFib + Combined RR Analysis (ML-focused)
===========================================================

For each AF episode, extracts RR segments at horizons 10, 20, 30, 60, 120, 180 min:
  - pre_afib: RR immediately before AF onset
  - afib: first H minutes of the labeled AF RR segment (if shorter, all available)
  - both: concatenation [pre_afib, afib]

Computes the same feature set as the pre-AFib-only script, aggregates separately and
combined, and writes CSVs, plots, and a Markdown report under Results/.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
BASE_RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results"
OUT_DIR = BASE_RESULTS_DIR / "pre_afib_afib_combined_analysis"
PLOTS_DIR = OUT_DIR / "plots"

HORIZON_MINUTES = [10, 20, 30, 60, 120, 180]
PHYSIOLOGICAL_RR_MIN_MS = 250
PHYSIOLOGICAL_RR_MAX_MS = 2500
IQR_OUTLIER_FACTOR = 1.5
CALIBRATION_SKIP_MS = 60_000  # Remove first 1 minute of RR by cumulative duration.

SEGMENT_TYPES = ("pre_afib", "afib", "both")

sns.set_theme(style="whitegrid", font_scale=1.0)


@dataclass
class EpisodeMeta:
    patient: str
    episode_index: int
    start_file_index: int
    end_file_index: int
    start_rr_index: int
    end_rr_index: int
    af_duration_sec: float
    nsr_before_sec: float


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def load_patient_rr(record_dir: str) -> tuple[np.ndarray, Dict[int, int]]:
    rr_files: Dict[int, str] = {}
    for fname in os.listdir(record_dir):
        if fname.endswith(".h5") and "_rr_" in fname:
            file_index = int(fname.split("_rr_")[1].replace(".h5", ""))
            rr_files[file_index] = fname

    rr_concat_parts: List[np.ndarray] = []
    file_start_offset: Dict[int, int] = {}
    running = 0

    for idx in sorted(rr_files.keys()):
        file_start_offset[idx] = running
        with h5py.File(os.path.join(record_dir, rr_files[idx]), "r") as f:
            arr = f["rr"][:]
        rr_concat_parts.append(arr)
        running += len(arr)

    if not rr_concat_parts:
        return np.array([], dtype=np.int64), {}
    return np.concatenate(rr_concat_parts), file_start_offset


def global_rr_index(file_offset: Dict[int, int], file_idx: int, local_idx: int) -> int:
    if file_idx not in file_offset:
        return -1
    return int(file_offset[file_idx] + local_idx)


def build_episode_list(dataset_path: str) -> List[EpisodeMeta]:
    episodes: List[EpisodeMeta] = []
    patient_dirs = sorted(os.listdir(dataset_path))

    for patient in patient_dirs:
        record_dir = os.path.join(dataset_path, patient)
        if not os.path.isdir(record_dir):
            continue

        ecg_label_path = os.path.join(record_dir, f"{patient}_ecg_labels.csv")
        rr_label_path = os.path.join(record_dir, f"{patient}_rr_labels.csv")
        if not (os.path.exists(ecg_label_path) and os.path.exists(rr_label_path)):
            continue

        ecg_df = pd.read_csv(ecg_label_path)
        rr_df = pd.read_csv(rr_label_path)
        n = min(len(ecg_df), len(rr_df))

        for i in range(n):
            episodes.append(
                EpisodeMeta(
                    patient=patient,
                    episode_index=i,
                    start_file_index=int(rr_df.loc[i, "start_file_index"]),
                    end_file_index=int(rr_df.loc[i, "end_file_index"]),
                    start_rr_index=int(rr_df.loc[i, "start_rr_index"]),
                    end_rr_index=int(rr_df.loc[i, "end_rr_index"]),
                    af_duration_sec=float(ecg_df.loc[i, "af_duration"]) / 1000.0,
                    nsr_before_sec=float(ecg_df.loc[i, "nsr_before_duration"]) / 1000.0,
                )
            )
    return episodes


def calibration_cut_index(rr: np.ndarray, skip_ms: int = CALIBRATION_SKIP_MS) -> int:
    if len(rr) == 0:
        return 0
    csum_ms = np.cumsum(rr.astype(np.int64))
    cut = int(np.searchsorted(csum_ms, skip_ms, side="left") + 1)
    return min(cut, len(rr))


def extract_pre_afib_segment(
    rr: np.ndarray, afib_start_global_idx: int, horizon_sec: int, calibration_start_idx: int
) -> np.ndarray:
    if afib_start_global_idx <= 0 or len(rr) == 0:
        return np.array([], dtype=rr.dtype)

    valid_start = min(calibration_start_idx, len(rr))
    prior = rr[valid_start:afib_start_global_idx]
    if len(prior) == 0:
        return np.array([], dtype=rr.dtype)

    rev = prior[::-1]
    rev_cumsum_sec = np.cumsum(rev) / 1000.0
    cut = np.searchsorted(rev_cumsum_sec, horizon_sec, side="left")

    if cut >= len(rev):
        seg = rev[::-1]
    else:
        seg = rev[: cut + 1][::-1]
    return seg


def extract_afib_segment_horizon(
    rr: np.ndarray,
    af_start_global: int,
    af_end_global_exclusive: int,
    horizon_sec: int,
    calibration_start_idx: int,
) -> np.ndarray:
    """First `horizon_sec` of RR within the labeled AF interval [af_start, af_end)."""
    if af_start_global < 0 or af_end_global_exclusive <= af_start_global or len(rr) == 0:
        return np.array([], dtype=rr.dtype)

    lo = max(calibration_start_idx, af_start_global)
    hi = min(len(rr), af_end_global_exclusive)
    seg = rr[lo:hi]
    if len(seg) == 0:
        return np.array([], dtype=rr.dtype)

    cum_sec = np.cumsum(seg) / 1000.0
    cut = np.searchsorted(cum_sec, horizon_sec, side="left")
    if cut >= len(seg):
        return seg
    return seg[: cut + 1]


def rr_features(rr_seg: np.ndarray) -> Dict[str, float]:
    if len(rr_seg) == 0:
        return {
            "rr_count": 0,
            "duration_sec": 0.0,
            "mean_rr_ms": 0.0,
            "median_rr_ms": 0.0,
            "std_rr_ms": 0.0,
            "iqr_rr_ms": 0.0,
            "min_rr_ms": 0.0,
            "max_rr_ms": 0.0,
            "mean_hr_bpm": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50": 0.0,
            "outlier_count_low_phys": 0,
            "outlier_count_high_phys": 0,
            "outlier_count_total_phys": 0,
            "outlier_pct_phys": 100.0,
            "outlier_count_iqr": 0,
            "outlier_pct_iqr": 0.0,
            "phys_out_of_range_pct": 100.0,
            "duplicate_pct": 0.0,
            "rr_diff_std_ms": 0.0,
            "trend_slope_ms_per_beat": 0.0,
        }

    rr = rr_seg.astype(float)
    rr_diff = np.diff(rr)
    out_of_range = np.logical_or(rr < PHYSIOLOGICAL_RR_MIN_MS, rr > PHYSIOLOGICAL_RR_MAX_MS)
    outlier_low = rr < PHYSIOLOGICAL_RR_MIN_MS
    outlier_high = rr > PHYSIOLOGICAL_RR_MAX_MS

    q1 = np.percentile(rr, 25)
    q3 = np.percentile(rr, 75)
    iqr = q3 - q1
    low_iqr_bound = q1 - IQR_OUTLIER_FACTOR * iqr
    high_iqr_bound = q3 + IQR_OUTLIER_FACTOR * iqr
    outlier_iqr = np.logical_or(rr < low_iqr_bound, rr > high_iqr_bound)

    duplicate_pct = safe_div(np.sum(np.diff(rr) == 0), max(len(rr) - 1, 1)) * 100.0

    x = np.arange(len(rr), dtype=float)
    slope = np.polyfit(x, rr, 1)[0] if len(rr) >= 2 else 0.0

    return {
        "rr_count": int(len(rr)),
        "duration_sec": float(np.sum(rr) / 1000.0),
        "mean_rr_ms": float(np.mean(rr)),
        "median_rr_ms": float(np.median(rr)),
        "std_rr_ms": float(np.std(rr)),
        "iqr_rr_ms": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
        "min_rr_ms": float(np.min(rr)),
        "max_rr_ms": float(np.max(rr)),
        "mean_hr_bpm": float(60000.0 / np.mean(rr)),
        "sdnn_ms": float(np.std(rr)),
        "rmssd_ms": float(np.sqrt(np.mean(rr_diff**2))) if len(rr_diff) > 0 else 0.0,
        "pnn50": float(np.mean(np.abs(rr_diff) > 50.0) * 100.0) if len(rr_diff) > 0 else 0.0,
        "outlier_count_low_phys": int(np.sum(outlier_low)),
        "outlier_count_high_phys": int(np.sum(outlier_high)),
        "outlier_count_total_phys": int(np.sum(out_of_range)),
        "outlier_pct_phys": float(np.mean(out_of_range) * 100.0),
        "outlier_count_iqr": int(np.sum(outlier_iqr)),
        "outlier_pct_iqr": float(np.mean(outlier_iqr) * 100.0),
        "phys_out_of_range_pct": float(np.mean(out_of_range) * 100.0),
        "duplicate_pct": float(duplicate_pct),
        "rr_diff_std_ms": float(np.std(rr_diff)) if len(rr_diff) > 0 else 0.0,
        "trend_slope_ms_per_beat": float(slope),
    }


def quality_score(features: Dict[str, float], target_duration_sec: int) -> tuple[float, bool]:
    if target_duration_sec <= 0:
        return 0.0, False
    coverage_ratio = min(1.0, safe_div(features["duration_sec"], target_duration_sec))
    coverage_pts = 40.0 * coverage_ratio

    phys_valid_ratio = max(0.0, 1.0 - features["phys_out_of_range_pct"] / 100.0)
    phys_pts = 20.0 * phys_valid_ratio

    count_target = max(200.0, target_duration_sec / 0.9)
    count_ratio = min(1.0, safe_div(features["rr_count"], count_target))
    count_pts = 15.0 * count_ratio

    duplicate_penalty = min(1.0, features["duplicate_pct"] / 20.0)
    artifact_pts = 15.0 * (1.0 - duplicate_penalty)

    std = features["std_rr_ms"]
    if std < 10:
        variability_pts = 2.0
    elif std > 400:
        variability_pts = 4.0
    else:
        variability_pts = 10.0

    score = coverage_pts + phys_pts + count_pts + artifact_pts + variability_pts

    ml_ready = (
        coverage_ratio >= 0.95
        and features["rr_count"] >= 200
        and features["phys_out_of_range_pct"] <= 5.0
        and features["duplicate_pct"] <= 10.0
        and score >= 70.0
    )
    return float(score), bool(ml_ready)


def add_traceability_pre(feat: Dict, afib_start: int, calibration_start_idx: int) -> Dict[str, int | str]:
    n = feat["rr_count"]
    start = max(calibration_start_idx, afib_start - n)
    return {
        "segment_start_global_idx": int(start),
        "segment_end_global_idx": int(afib_start),
        "segment_global_index_range": f"{start}:{afib_start}",
    }


def add_traceability_af(feat: Dict, af_start: int) -> Dict[str, int | str]:
    n = feat["rr_count"]
    end = af_start + n
    return {
        "segment_start_global_idx": int(af_start),
        "segment_end_global_idx": int(end),
        "segment_global_index_range": f"{af_start}:{end}",
    }


def add_traceability_both(
    pre_feat: Dict, af_feat: Dict, pre_start: int, pre_end: int, af_start: int
) -> Dict[str, int | str]:
    n_pre = pre_feat["rr_count"]
    n_af = af_feat["rr_count"]
    if n_pre == 0 and n_af == 0:
        return {
            "segment_start_global_idx": int(pre_start),
            "segment_end_global_idx": int(pre_start),
            "segment_global_index_range": f"{pre_start}:{pre_start}",
        }
    combined_start = pre_start if n_pre > 0 else af_start
    combined_end = (af_start + n_af) if n_af > 0 else pre_end
    return {
        "segment_start_global_idx": int(combined_start),
        "segment_end_global_idx": int(combined_end),
        "segment_global_index_range": f"{combined_start}:{combined_end}",
    }


def generate_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) ML-ready by horizon and segment_type
    agg = (
        df.groupby(["segment_type", "horizon_min"])[["is_sufficient_duration", "ml_ready"]]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = agg.pivot(index="horizon_min", columns="segment_type", values="ml_ready")
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Fraction ML-ready")
    ax.set_title("ML-ready rate by horizon and segment (pre_afib / afib / both)")
    ax.legend(title="segment_type")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_ml_ready_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    # 2) Quality score boxplots per segment_type
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="horizon_min", y="quality_score", hue="segment_type", ax=ax)
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Quality score")
    ax.set_title("Quality score distribution by horizon and segment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_quality_score_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    # 3) Mean RR by segment (shows SR vs AF separation)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="horizon_min", y="mean_rr_ms", hue="segment_type", ax=ax, showfliers=False)
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Mean RR (ms)")
    ax.set_title("Mean RR by horizon and segment (pre vs AF vs both)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_mean_rr_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    # 4) Outlier % by segment
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="horizon_min", y="outlier_pct_phys", hue="segment_type", ax=ax, showfliers=False)
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("Physical outlier %")
    ax.set_title("Outlier rate by horizon and segment")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_outlier_pct_by_horizon_segment.png", dpi=150)
    plt.close(fig)

    # 5) Heatmap: ml_ready mean
    heat = df.groupby(["segment_type", "horizon_min"])["ml_ready"].mean().unstack(0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat.T, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "ML-ready %"})
    ax.set_title("ML-ready % (rows = segment_type, cols = horizon min)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_ml_ready_heatmap.png", dpi=150)
    plt.close(fig)

    # 6) ECDF of outlier percentage
    fig, ax = plt.subplots(figsize=(10, 5))
    ne = df[df["rr_count"] > 0].copy()
    for seg in SEGMENT_TYPES:
        vals = np.sort(ne.loc[ne["segment_type"] == seg, "outlier_pct_phys"].values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=seg)
    ax.set_xlabel("Outlier % per segment")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of Physical Outlier Percentage (non-empty segments)")
    ax.legend(title="segment_type")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "06_outlier_pct_ecdf.png", dpi=150)
    plt.close(fig)

    # 7) Segment cleanliness bands
    band_df = ne.copy()
    bins = [-1e-9, 0, 0.5, 1.0, 5.0, 100.0]
    labels = ["0%", "0-0.5%", "0.5-1%", "1-5%", ">5%"]
    band_df["outlier_band"] = pd.cut(band_df["outlier_pct_phys"], bins=bins, labels=labels)
    band = (
        band_df.groupby(["segment_type", "outlier_band"], observed=False).size().reset_index(name="n")
    )
    band["pct"] = band.groupby("segment_type")["n"].transform(lambda x: x / x.sum() * 100.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=band, x="outlier_band", y="pct", hue="segment_type", ax=ax)
    ax.set_xlabel("Outlier % band")
    ax.set_ylabel("Segments (%)")
    ax.set_title("Distribution of Segment Outlier Bands")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "07_outlier_band_distribution.png", dpi=150)
    plt.close(fig)


def build_cleanliness_verification(df: pd.DataFrame) -> pd.DataFrame:
    ne = df[df["rr_count"] > 0].copy()
    rows = []
    for (seg, h), g in ne.groupby(["segment_type", "horizon_min"]):
        n = len(g)
        fail_outlier_2 = float(np.mean(g["outlier_pct_phys"] > 2.0) * 100.0)
        fail_outlier_5 = float(np.mean(g["outlier_pct_phys"] > 5.0) * 100.0)
        fail_dup_10 = float(np.mean(g["duplicate_pct"] > 10.0) * 100.0)
        fail_rr_200 = float(np.mean(g["rr_count"] < 200) * 100.0)
        fail_min = float(np.mean(g["min_rr_ms"] < PHYSIOLOGICAL_RR_MIN_MS) * 100.0)
        fail_max = float(np.mean(g["max_rr_ms"] > PHYSIOLOGICAL_RR_MAX_MS) * 100.0)
        fail_any = float(
            np.mean(
                (g["outlier_pct_phys"] > 5.0)
                | (g["duplicate_pct"] > 10.0)
                | (g["rr_count"] < 200)
            )
            * 100.0
        )
        total_out = float(g["outlier_count_total_phys"].sum())
        total_rr = float(g["rr_count"].sum())
        rows.append(
            {
                "segment_type": seg,
                "horizon_min": int(h),
                "n_nonempty": int(n),
                "global_outlier_pct": (total_out / total_rr * 100.0) if total_rr > 0 else np.nan,
                "mean_outlier_pct_nonempty": float(g["outlier_pct_phys"].mean()),
                "median_outlier_pct_nonempty": float(g["outlier_pct_phys"].median()),
                "p95_outlier_pct_nonempty": float(np.percentile(g["outlier_pct_phys"], 95)),
                "p99_outlier_pct_nonempty": float(np.percentile(g["outlier_pct_phys"], 99)),
                "fail_outlier_gt_2_pct": fail_outlier_2,
                "fail_outlier_gt_5_pct": fail_outlier_5,
                "fail_duplicate_gt_10_pct": fail_dup_10,
                "fail_rr_count_lt_200_pct": fail_rr_200,
                "fail_min_rr_lt_250_pct": fail_min,
                "fail_max_rr_gt_2500_pct": fail_max,
                "fail_any_rule_pct": fail_any,
                "p1_min_rr_ms": float(np.percentile(g["min_rr_ms"], 1)),
                "p99_max_rr_ms": float(np.percentile(g["max_rr_ms"], 99)),
            }
        )
    return pd.DataFrame(rows).sort_values(["segment_type", "horizon_min"]).reset_index(drop=True)


def write_report(df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# IRIDIA Pre-AFib + AFib + Combined Episode Analysis")
    lines.append("")
    lines.append(f"> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "For each AF episode, analyze RR windows of **10, 20, 30, 60, 120, 180 minutes** for: "
        "**pre_afib** (before onset), **afib** (within labeled AF, up to H min), and **both** "
        "(concatenation pre then AF)."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- AF onset: global index = `file_offset[start_file_index] + start_rr_index`.")
    lines.append("- AF interval: `[af_start, af_end)` from RR labels (`end_file_index`, `end_rr_index`).")
    lines.append("- **pre_afib**: last H minutes of RR strictly before `af_start`.")
    lines.append("- **afib**: first H minutes of RR inside the labeled AF interval (or all AF RR if shorter).")
    lines.append("- **both**: `concat(pre_afib, afib)`; target duration for sufficiency/score = **2×H** minutes.")
    lines.append(
        f"- Calibration handling: first **{CALIBRATION_SKIP_MS/1000:.0f} seconds** of RR are removed "
        "using cumulative RR duration."
    )
    lines.append("- Same ML heuristics as pre-AFib script: coverage ≥95%, RR count, outliers [250,2500] ms, duplicates, quality score.")
    lines.append("")
    lines.append("## Summary by segment_type × horizon")
    lines.append("")
    lines.append(
        "| segment_type | horizon_min | n_rows | n_nonempty | sufficient_% | ml_ready_% | mean_quality | "
        "mean_rr_count | mean_outlier_pct_nonempty | median_outlier_pct_nonempty | "
        "p95_outlier_pct_nonempty | global_outlier_pct | mean_mean_rr_ms | mean_std_rr_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    summary_base = (
        df.groupby(["segment_type", "horizon_min"])
        .agg(
            n=("episode_index", "count"),
            sufficient_pct=("is_sufficient_duration", lambda x: float(np.mean(x) * 100)),
            ml_ready_pct=("ml_ready", lambda x: float(np.mean(x) * 100)),
            mean_quality=("quality_score", "mean"),
            mean_rr_count=("rr_count", "mean"),
            mean_mean_rr=("mean_rr_ms", "mean"),
            mean_std_rr=("std_rr_ms", "mean"),
        )
    )

    non_empty = df[df["rr_count"] > 0].copy()
    summary_nonempty = (
        non_empty.groupby(["segment_type", "horizon_min"])
        .agg(
            n_nonempty=("rr_count", "size"),
            mean_outlier_pct_nonempty=("outlier_pct_phys", "mean"),
            median_outlier_pct_nonempty=("outlier_pct_phys", "median"),
            p95_outlier_pct_nonempty=("outlier_pct_phys", lambda x: float(np.percentile(x, 95))),
            total_outliers=("outlier_count_total_phys", "sum"),
            total_rr=("rr_count", "sum"),
        )
    )

    summary = summary_base.join(summary_nonempty, how="left").reset_index()
    summary["global_outlier_pct"] = np.where(
        summary["total_rr"] > 0,
        (summary["total_outliers"] / summary["total_rr"]) * 100.0,
        np.nan,
    )

    for _, r in summary.sort_values(["segment_type", "horizon_min"]).iterrows():
        lines.append(
            f"| {r['segment_type']} | {int(r['horizon_min'])} | {int(r['n'])} | "
            f"{int(r['n_nonempty']) if pd.notna(r['n_nonempty']) else 0} | "
            f"{r['sufficient_pct']:.2f} | {r['ml_ready_pct']:.2f} | {r['mean_quality']:.2f} | "
            f"{r['mean_rr_count']:.1f} | {r['mean_outlier_pct_nonempty']:.4f} | "
            f"{r['median_outlier_pct_nonempty']:.4f} | {r['p95_outlier_pct_nonempty']:.4f} | "
            f"{r['global_outlier_pct']:.4f} | {r['mean_mean_rr']:.1f} | {r['mean_std_rr']:.1f} |"
        )

    lines.append("")
    lines.append("## Physical Cleanliness Verification")
    lines.append("")
    ver = build_cleanliness_verification(df)
    lines.append(
        "| segment_type | horizon_min | n_nonempty | global_outlier_% | mean_outlier_% | p95_outlier_% | "
        "p99_outlier_% | fail(>2%)_% | fail(>5%)_% | fail(dup>10%)_% | fail(rr<200)_% | "
        "fail(min<250)_% | fail(max>2500)_% | fail(any rule)_% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ver.iterrows():
        lines.append(
            f"| {r['segment_type']} | {int(r['horizon_min'])} | {int(r['n_nonempty'])} | "
            f"{r['global_outlier_pct']:.4f} | {r['mean_outlier_pct_nonempty']:.4f} | "
            f"{r['p95_outlier_pct_nonempty']:.4f} | {r['p99_outlier_pct_nonempty']:.4f} | "
            f"{r['fail_outlier_gt_2_pct']:.2f} | {r['fail_outlier_gt_5_pct']:.2f} | "
            f"{r['fail_duplicate_gt_10_pct']:.2f} | {r['fail_rr_count_lt_200_pct']:.2f} | "
            f"{r['fail_min_rr_lt_250_pct']:.2f} | {r['fail_max_rr_gt_2500_pct']:.2f} | "
            f"{r['fail_any_rule_pct']:.2f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **afib** segments are often shorter than H minutes because labeled AF duration in this dataset "
        "can be brief; `is_sufficient_duration` reflects whether ≥95% of requested H min was captured."
    )
    lines.append(
        "- **both** uses a **2H-minute** target for coverage/score; pre and AF parts are still each capped at H min."
    )
    lines.append(
        "- Outlier stats use non-empty segments only (`rr_count > 0`) and include "
        "`global_outlier_pct = sum(outliers)/sum(rr_count)*100`."
    )
    lines.append("")
    lines.append("## Visualizations")
    for name, title in [
        ("01_ml_ready_by_horizon_segment.png", "ML-ready by horizon and segment"),
        ("02_quality_score_by_horizon_segment.png", "Quality score by horizon and segment"),
        ("03_mean_rr_by_horizon_segment.png", "Mean RR by horizon and segment"),
        ("04_outlier_pct_by_horizon_segment.png", "Outlier % by horizon and segment"),
        ("05_ml_ready_heatmap.png", "ML-ready heatmap"),
        ("06_outlier_pct_ecdf.png", "Outlier % ECDF"),
        ("07_outlier_band_distribution.png", "Outlier band distribution"),
    ]:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}](plots/{name})")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `dataset_experiments/IRIDIA/pre_afib_afib_combined_analysis.py`*")

    (OUT_DIR / "IRIDIA_PreAFib_AFib_Combined_Report.md").write_text("\n".join(lines))
    summary.to_csv(OUT_DIR / "horizon_segment_summary.csv", index=False)
    with open(OUT_DIR / "horizon_segment_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    ver = build_cleanliness_verification(df)
    ver.to_csv(OUT_DIR / "cleanliness_verification_summary.csv", index=False)
    with open(OUT_DIR / "cleanliness_verification_summary.json", "w") as f:
        json.dump(ver.to_dict(orient="records"), f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("=" * 72)
    print("IRIDIA Pre-AFib + AFib + Combined analysis")
    print("=" * 72)

    episodes = build_episode_list(DATASET_PATH)
    print(f"Total AF episodes: {len(episodes)}")

    patient_cache: Dict[str, tuple[np.ndarray, Dict[int, int]]] = {}
    rows: List[Dict] = []

    for idx, ep in enumerate(episodes):
        if ep.patient not in patient_cache:
            record_dir = os.path.join(DATASET_PATH, ep.patient)
            patient_cache[ep.patient] = load_patient_rr(record_dir)

        rr_all, file_offset = patient_cache[ep.patient]
        if ep.start_file_index not in file_offset or ep.end_file_index not in file_offset:
            continue

        calibration_start_idx = calibration_cut_index(rr_all, CALIBRATION_SKIP_MS)
        af_start = global_rr_index(file_offset, ep.start_file_index, ep.start_rr_index)
        af_end = global_rr_index(file_offset, ep.end_file_index, ep.end_rr_index)

        if af_start < 0 or af_end <= af_start:
            continue

        for horizon_min in HORIZON_MINUTES:
            target_sec = horizon_min * 60
            target_both_sec = 2 * target_sec

            seg_pre = extract_pre_afib_segment(rr_all, af_start, target_sec, calibration_start_idx)
            seg_af = extract_afib_segment_horizon(rr_all, af_start, af_end, target_sec, calibration_start_idx)

            feat_pre = rr_features(seg_pre)
            feat_af = rr_features(seg_af)

            if len(seg_pre) > 0 and len(seg_af) > 0:
                seg_both = np.concatenate([seg_pre, seg_af])
            elif len(seg_pre) > 0:
                seg_both = seg_pre.copy()
            elif len(seg_af) > 0:
                seg_both = seg_af.copy()
            else:
                seg_both = np.array([], dtype=rr_all.dtype)

            feat_both = rr_features(seg_both)

            pre_trace = add_traceability_pre(feat_pre, af_start, calibration_start_idx)
            af_trace = add_traceability_af(feat_af, af_start)
            both_trace = add_traceability_both(
                feat_pre, feat_af, pre_trace["segment_start_global_idx"], af_start, af_start
            )

            sufficient_pre = feat_pre["duration_sec"] >= target_sec * 0.95
            sufficient_af = feat_af["duration_sec"] >= target_sec * 0.95
            sufficient_both = feat_both["duration_sec"] >= target_both_sec * 0.95

            score_pre, ready_pre = quality_score(feat_pre, target_sec)
            score_af, ready_af = quality_score(feat_af, target_sec)
            score_both, ready_both = quality_score(feat_both, target_both_sec)

            base = {
                "patient": ep.patient,
                "episode_index": ep.episode_index,
                "start_file_index": ep.start_file_index,
                "end_file_index": ep.end_file_index,
                "afib_start_global_idx": af_start,
                "afib_end_global_idx_exclusive": af_end,
                "horizon_min": horizon_min,
                "target_duration_sec_pre_or_af": target_sec,
                "target_duration_sec_both": target_both_sec,
                "af_labeled_duration_sec": float(np.sum(rr_all[af_start:af_end]) / 1000.0),
                "af_duration_sec_meta": ep.af_duration_sec,
                "nsr_before_sec": ep.nsr_before_sec,
            }

            for seg_type, feat, suff, sc, ml, trace in [
                ("pre_afib", feat_pre, sufficient_pre, score_pre, ready_pre, pre_trace),
                ("afib", feat_af, sufficient_af, score_af, ready_af, af_trace),
                ("both", feat_both, sufficient_both, score_both, ready_both, both_trace),
            ]:
                row = {
                    **base,
                    "segment_type": seg_type,
                    "is_sufficient_duration": bool(suff),
                    "quality_score": sc,
                    "ml_ready": bool(ml),
                    "outlier_pct_phys_nonempty": feat["outlier_pct_phys"] if feat["rr_count"] > 0 else np.nan,
                    "outlier_pct_iqr_nonempty": feat["outlier_pct_iqr"] if feat["rr_count"] > 0 else np.nan,
                    **trace,
                    **feat,
                }
                rows.append(row)

        if (idx + 1) % 50 == 0 or idx + 1 == len(episodes):
            print(f"Processed episodes: {idx + 1}/{len(episodes)}")

    df = pd.DataFrame(rows)
    df.sort_values(["patient", "episode_index", "horizon_min", "segment_type"], inplace=True)
    df.to_csv(OUT_DIR / "episode_pre_afib_afib_combined_features.csv", index=False)

    trace_cols = [
        "patient",
        "episode_index",
        "segment_type",
        "horizon_min",
        "start_file_index",
        "end_file_index",
        "afib_start_global_idx",
        "afib_end_global_idx_exclusive",
        "segment_start_global_idx",
        "segment_end_global_idx",
        "segment_global_index_range",
        "target_duration_sec_pre_or_af",
        "target_duration_sec_both",
        "duration_sec",
        "is_sufficient_duration",
        "rr_count",
        "outlier_count_low_phys",
        "outlier_count_high_phys",
        "outlier_count_total_phys",
        "outlier_pct_phys",
        "outlier_count_iqr",
        "outlier_pct_iqr",
        "mean_rr_ms",
        "median_rr_ms",
        "std_rr_ms",
        "iqr_rr_ms",
        "min_rr_ms",
        "max_rr_ms",
        "mean_hr_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50",
        "duplicate_pct",
        "quality_score",
        "ml_ready",
        "af_labeled_duration_sec",
    ]
    df[trace_cols].to_csv(OUT_DIR / "episode_segment_traceability.csv", index=False)

    generate_plots(df)
    write_report(df)

    run_summary = {
        "n_episodes": int(df[["patient", "episode_index"]].drop_duplicates().shape[0]),
        "rows": int(len(df)),
        "segment_types": list(SEGMENT_TYPES),
        "horizons_min": HORIZON_MINUTES,
        "output_dir": str(OUT_DIR),
    }

    with open(OUT_DIR / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\nOutputs:")
    print(f"- {OUT_DIR / 'episode_pre_afib_afib_combined_features.csv'}")
    print(f"- {OUT_DIR / 'episode_segment_traceability.csv'}")
    print(f"- {OUT_DIR / 'horizon_segment_summary.csv'}")
    print(f"- {OUT_DIR / 'IRIDIA_PreAFib_AFib_Combined_Report.md'}")
    print(f"- {PLOTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
