"""
IRIDIA AF Episode Pre-Onset RR Analysis (ML-focused)
===================================================

For every AF episode in IRIDIA, this script extracts RR data before AF onset
at fixed horizons (10, 20, 30, 60, 120, 180 minutes), computes quality and
feature statistics, and creates a report to assess if RR data is suitable
for machine learning.
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
OUT_DIR = BASE_RESULTS_DIR / "pre_afib_episode_analysis"
PLOTS_DIR = OUT_DIR / "plots"

HORIZON_MINUTES = [10, 20, 30, 60, 120, 180]
PHYSIOLOGICAL_RR_MIN_MS = 250
PHYSIOLOGICAL_RR_MAX_MS = 2500
IQR_OUTLIER_FACTOR = 1.5

sns.set_theme(style="whitegrid", font_scale=1.0)


@dataclass
class EpisodeMeta:
    patient: str
    episode_index: int
    start_file_index: int
    start_rr_index: int
    af_duration_sec: float
    nsr_before_sec: float


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def load_patient_rr(record_dir: str, patient: str) -> tuple[np.ndarray, Dict[int, int]]:
    """
    Load and concatenate all RR files for a patient.
    Returns:
      - rr_concat: full concatenated RR array
      - file_start_offset: map file_index -> global start offset in rr_concat
    """
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
                    start_rr_index=int(rr_df.loc[i, "start_rr_index"]),
                    af_duration_sec=float(ecg_df.loc[i, "af_duration"]) / 1000.0,
                    nsr_before_sec=float(ecg_df.loc[i, "nsr_before_duration"]) / 1000.0,
                )
            )
    return episodes


def extract_pre_afib_segment(rr: np.ndarray, afib_start_global_idx: int, horizon_sec: int) -> np.ndarray:
    """Extract RR data directly before AF onset with target duration horizon_sec."""
    if afib_start_global_idx <= 0 or len(rr) == 0:
        return np.array([], dtype=rr.dtype)

    prior = rr[:afib_start_global_idx]
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
    if len(rr) >= 2:
        slope = np.polyfit(x, rr, 1)[0]
    else:
        slope = 0.0

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


def quality_score(features: Dict[str, float], target_horizon_sec: int) -> tuple[float, bool]:
    """
    Heuristic ML-readiness score [0,100].
    Higher means cleaner, sufficient and informative RR segment.
    """
    coverage_ratio = min(1.0, safe_div(features["duration_sec"], target_horizon_sec))
    coverage_pts = 40.0 * coverage_ratio

    phys_valid_ratio = max(0.0, 1.0 - features["phys_out_of_range_pct"] / 100.0)
    phys_pts = 20.0 * phys_valid_ratio

    # Encourage adequate sample count for model features.
    count_target = max(200.0, target_horizon_sec / 0.9)
    count_ratio = min(1.0, safe_div(features["rr_count"], count_target))
    count_pts = 15.0 * count_ratio

    # Penalize repetitive or likely-artifact-heavy signal.
    duplicate_penalty = min(1.0, features["duplicate_pct"] / 20.0)
    artifact_pts = 15.0 * (1.0 - duplicate_penalty)

    # Encourage non-degenerate variability.
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


def generate_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Availability / ML-ready by horizon
    agg = (
        df.groupby("horizon_min")[["is_sufficient_duration", "ml_ready"]]
        .mean()
        .rename(columns={"is_sufficient_duration": "sufficient_ratio", "ml_ready": "ml_ready_ratio"})
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    x = np.arange(len(agg))
    ax.bar(x - width / 2, agg["sufficient_ratio"] * 100, width=width, label="Sufficient Duration", color="#4C72B0")
    ax.bar(x + width / 2, agg["ml_ready_ratio"] * 100, width=width, label="ML Ready", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["horizon_min"].astype(int).astype(str))
    ax.set_xlabel("Pre-AFib Horizon (minutes)")
    ax.set_ylabel("Episodes (%)")
    ax.set_title("Episode Coverage and ML Readiness by Horizon")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_coverage_ml_ready_by_horizon.png", dpi=150)
    plt.close(fig)

    # 2) Score distribution by horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="horizon_min", y="quality_score", ax=ax, color="#8172B2")
    ax.set_xlabel("Pre-AFib Horizon (minutes)")
    ax.set_ylabel("Quality Score (0-100)")
    ax.set_title("ML Quality Score Distribution by Horizon")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_quality_score_by_horizon.png", dpi=150)
    plt.close(fig)

    # 3) RR counts by horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x="horizon_min", y="rr_count", ax=ax, color="#64B5CD", cut=0)
    ax.set_xlabel("Pre-AFib Horizon (minutes)")
    ax.set_ylabel("RR Count")
    ax.set_title("RR Interval Count by Horizon")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_rr_count_by_horizon.png", dpi=150)
    plt.close(fig)

    # 4) Artifact/out-of-range by horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="horizon_min", y="phys_out_of_range_pct", ax=ax, color="#CCB974")
    ax.set_xlabel("Pre-AFib Horizon (minutes)")
    ax.set_ylabel("Out-of-range RR (%)")
    ax.set_title("Physiological Out-of-range RR by Horizon")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_out_of_range_by_horizon.png", dpi=150)
    plt.close(fig)

    # 5) Physical outlier counts by horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="horizon_min", y="outlier_count_total_phys", ax=ax, color="#E17C05")
    ax.set_xlabel("Pre-AFib Horizon (minutes)")
    ax.set_ylabel("Outlier Count (RR < 250 ms or RR > 2500 ms)")
    ax.set_title("Physical Outlier Count per Segment by Horizon")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_outlier_count_by_horizon.png", dpi=150)
    plt.close(fig)

    # 6) Threshold pass rates (heatmap)
    pass_df = df.copy()
    pass_df["pass_duration"] = pass_df["is_sufficient_duration"].astype(int)
    pass_df["pass_rr_count"] = (pass_df["rr_count"] >= 200).astype(int)
    pass_df["pass_outlier_pct"] = (pass_df["outlier_pct_phys"] <= 5.0).astype(int)
    pass_df["pass_duplicate"] = (pass_df["duplicate_pct"] <= 10.0).astype(int)
    pass_df["pass_quality"] = (pass_df["quality_score"] >= 70.0).astype(int)

    pass_rates = (
        pass_df.groupby("horizon_min")[["pass_duration", "pass_rr_count", "pass_outlier_pct", "pass_duplicate", "pass_quality"]]
        .mean()
        .T
    )
    pass_rates.index = [
        "Duration >= 95%",
        "RR Count >= 200",
        "Outlier % <= 5",
        "Duplicate % <= 10",
        "Quality Score >= 70",
    ]
    pass_rates = pass_rates * 100.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(pass_rates, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "Pass Rate (%)"})
    ax.set_xlabel("Pre-AFib Horizon (minutes)")
    ax.set_ylabel("ML Readiness Rules")
    ax.set_title("Threshold Pass Rates by Horizon")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "06_threshold_passrate_heatmap.png", dpi=150)
    plt.close(fig)

    # 7) Feature correlation (all episode-horizon rows)
    corr_cols = [
        "rr_count",
        "duration_sec",
        "std_rr_ms",
        "rmssd_ms",
        "pnn50",
        "outlier_pct_phys",
        "duplicate_pct",
        "quality_score",
    ]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, square=True, ax=ax)
    ax.set_title("Feature Correlation Matrix (Episode-Horizon Rows)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "07_feature_correlation.png", dpi=150)
    plt.close(fig)

    # 8) Quality score vs outlier percentage
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df.sample(min(2000, len(df)), random_state=42),
        x="outlier_pct_phys",
        y="quality_score",
        hue="horizon_min",
        palette="viridis",
        alpha=0.7,
        s=40,
        ax=ax,
    )
    ax.set_xlabel("Physical Outlier Percentage (%)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality Score vs Outlier Percentage")
    ax.legend(title="Horizon (min)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "08_quality_vs_outlier_scatter.png", dpi=150)
    plt.close(fig)


def write_report(df: pd.DataFrame) -> None:
    metric_catalog = [
        {
            "metric": "is_sufficient_duration",
            "definition": "Segment contains at least 95% of requested pre-AFib horizon duration.",
            "threshold_for_ml": "True",
            "reason": "Insufficient context increases label noise and feature instability.",
        },
        {
            "metric": "rr_count",
            "definition": "Number of RR intervals in segment.",
            "threshold_for_ml": ">= 200",
            "reason": "Low counts produce weak HRV and distribution estimates.",
        },
        {
            "metric": "outlier_count_total_phys / outlier_pct_phys",
            "definition": f"RR values outside [{PHYSIOLOGICAL_RR_MIN_MS}, {PHYSIOLOGICAL_RR_MAX_MS}] ms.",
            "threshold_for_ml": "<= 5%",
            "reason": "High outlier rate indicates artifact or poor signal quality.",
        },
        {
            "metric": "duplicate_pct",
            "definition": "Fraction of consecutive RR differences equal to zero.",
            "threshold_for_ml": "<= 10%",
            "reason": "Excess duplicates can indicate quantization or sensor artifacts.",
        },
        {
            "metric": "std_rr_ms",
            "definition": "RR standard deviation.",
            "threshold_for_ml": "10-400 ms (soft range)",
            "reason": "Near-zero or extreme variability can be non-physiologic.",
        },
        {
            "metric": "quality_score",
            "definition": "Composite score from coverage, count, outliers, duplicates and variability.",
            "threshold_for_ml": ">= 70",
            "reason": "Single summary metric for model-ready segment filtering.",
        },
    ]

    agg = (
        df.groupby("horizon_min")
        .agg(
            episodes=("episode_index", "count"),
            sufficient_pct=("is_sufficient_duration", lambda x: float(np.mean(x) * 100)),
            ml_ready_pct=("ml_ready", lambda x: float(np.mean(x) * 100)),
            mean_quality_score=("quality_score", "mean"),
            median_quality_score=("quality_score", "median"),
            mean_rr_count=("rr_count", "mean"),
            mean_duration_min=("duration_sec", lambda x: float(np.mean(x) / 60.0)),
            mean_std_rr_ms=("std_rr_ms", "mean"),
            mean_rmssd_ms=("rmssd_ms", "mean"),
            mean_outlier_count_phys=("outlier_count_total_phys", "mean"),
            median_outlier_count_phys=("outlier_count_total_phys", "median"),
            mean_outlier_pct_phys=("outlier_pct_phys_nonempty", "mean"),
            p95_outlier_pct_phys=("outlier_pct_phys_nonempty", lambda x: float(np.nanpercentile(x, 95))),
            mean_outlier_count_iqr=("outlier_count_iqr", "mean"),
            mean_outlier_pct_iqr=("outlier_pct_iqr_nonempty", "mean"),
            mean_duplicate_pct=("duplicate_pct", "mean"),
        )
        .reset_index()
    )

    best_h_row = agg.sort_values(["ml_ready_pct", "mean_quality_score"], ascending=False).iloc[0]
    best_h = int(best_h_row["horizon_min"])

    lines: List[str] = []
    lines.append("# IRIDIA Pre-AFib Episode Analysis (ML Focus)")
    lines.append("")
    lines.append(f"> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Evaluate RR data quality before each AF episode at horizons of "
        "`10, 20, 30, 60, 120, 180` minutes to determine suitability for machine learning."
    )
    lines.append("")
    lines.append("## Metrics Checklist for ML Readiness")
    lines.append("")
    lines.append("| Metric | Definition | ML Threshold | Why it matters |")
    lines.append("|---|---|---|---|")
    for m in metric_catalog:
        lines.append(
            f"| `{m['metric']}` | {m['definition']} | `{m['threshold_for_ml']}` | {m['reason']} |"
        )
    lines.append("")
    lines.append("## Data and Method")
    lines.append("")
    lines.append("- Each AF episode uses its RR start index as AF onset.")
    lines.append("- For each horizon, RR intervals immediately preceding AF onset are extracted.")
    lines.append("- Features include RR central tendency, variability, signal quality, and artifacts.")
    lines.append("- A heuristic quality score (0-100) and `ml_ready` flag are computed per episode/horizon.")
    lines.append("")
    lines.append("## Horizon-level Summary")
    lines.append("")
    lines.append(
        "| Horizon (min) | Episodes | Sufficient (%) | ML Ready (%) | Mean Score | Mean RR Count | "
        "Mean Phys Outlier Count | Mean Phys Outlier (%) | P95 Phys Outlier (%) | Mean IQR Outlier (%) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        lines.append(
            f"| {int(r['horizon_min'])} | {int(r['episodes'])} | {r['sufficient_pct']:.2f} | "
            f"{r['ml_ready_pct']:.2f} | {r['mean_quality_score']:.2f} | {r['mean_rr_count']:.1f} | "
            f"{r['mean_outlier_count_phys']:.1f} | {r['mean_outlier_pct_phys']:.3f} | "
            f"{r['p95_outlier_pct_phys']:.3f} | {r['mean_outlier_pct_iqr']:.3f} |"
        )

    lines.append("")
    lines.append("## Statistical Evidence (Outliers and Quality)")
    lines.append("")
    lines.append(
        f"- Physiological outlier is defined as RR < **{PHYSIOLOGICAL_RR_MIN_MS} ms** or RR > "
        f"**{PHYSIOLOGICAL_RR_MAX_MS} ms**."
    )
    lines.append("- For each segment, both **outlier counts** and **outlier percentages** are reported.")
    lines.append(
        "- IQR-based outliers are additionally reported to detect distribution-level anomalies within each segment."
    )
    lines.append("- Report includes P95 outlier percentage to highlight worst-case tails, not only averages.")
    lines.append("")

    for _, r in agg.iterrows():
        h = int(r["horizon_min"])
        lines.append(
            f"- **{h} min**: mean physical outliers = **{r['mean_outlier_count_phys']:.1f}** "
            f"({r['mean_outlier_pct_phys']:.2f}%), P95={r['p95_outlier_pct_phys']:.2f}%, "
            f"mean IQR-outliers={r['mean_outlier_pct_iqr']:.2f}%."
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        f"- Best-performing horizon by ML readiness is **{best_h} minutes** "
        f"({best_h_row['ml_ready_pct']:.2f}% episodes marked ML-ready)."
    )
    lines.append(
        "- Shorter horizons usually have higher coverage; longer horizons can add context but may lose episodes "
        "due to insufficient pre-AFib history."
    )
    lines.append(
        "- For model design, use horizons with both high `ml_ready_pct` and stable RR count distributions."
    )
    lines.append(
        "- Outlier evidence supports that many segments are usable, but strict filtering is still required "
        "to remove high-artifact segments before model training."
    )
    lines.append("")
    lines.append("## Recommended ML Filtering Rules")
    lines.append("")
    lines.append("- `is_sufficient_duration == True`")
    lines.append("- `rr_count >= 200`")
    lines.append("- `phys_out_of_range_pct <= 5`")
    lines.append("- `duplicate_pct <= 10`")
    lines.append("- `quality_score >= 70`")
    lines.append("")
    lines.append("## Visualizations")
    lines.append("")
    lines.append("![Coverage and ML readiness](plots/01_coverage_ml_ready_by_horizon.png)")
    lines.append("")
    lines.append("![Quality score by horizon](plots/02_quality_score_by_horizon.png)")
    lines.append("")
    lines.append("![RR count by horizon](plots/03_rr_count_by_horizon.png)")
    lines.append("")
    lines.append("![Out-of-range by horizon](plots/04_out_of_range_by_horizon.png)")
    lines.append("")
    lines.append("![Outlier count by horizon](plots/05_outlier_count_by_horizon.png)")
    lines.append("")
    lines.append("![Threshold pass-rate heatmap](plots/06_threshold_passrate_heatmap.png)")
    lines.append("")
    lines.append("![Feature correlation](plots/07_feature_correlation.png)")
    lines.append("")
    lines.append("![Quality vs outlier scatter](plots/08_quality_vs_outlier_scatter.png)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `dataset_experiments/IRIDIA/pre_afib_episode_analysis.py`*")

    report_path = OUT_DIR / "IRIDIA_PreAFib_Episode_Report.md"
    report_path.write_text("\n".join(lines))

    # Save aggregated summary CSV/JSON too.
    agg.to_csv(OUT_DIR / "horizon_summary.csv", index=False)
    with open(OUT_DIR / "horizon_summary.json", "w") as f:
        json.dump(agg.to_dict(orient="records"), f, indent=2)
    with open(OUT_DIR / "metrics_catalog.json", "w") as f:
        json.dump(metric_catalog, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("=" * 72)
    print("IRIDIA Episode-level Pre-AFib RR Analysis")
    print("=" * 72)

    episodes = build_episode_list(DATASET_PATH)
    print(f"Total AF episodes discovered: {len(episodes)}")

    # Cache patient RR to avoid repeated disk reads.
    patient_cache: Dict[str, tuple[np.ndarray, Dict[int, int]]] = {}

    rows = []
    for idx, ep in enumerate(episodes):
        if ep.patient not in patient_cache:
            record_dir = os.path.join(DATASET_PATH, ep.patient)
            patient_cache[ep.patient] = load_patient_rr(record_dir, ep.patient)

        rr_all, file_offset = patient_cache[ep.patient]
        if ep.start_file_index not in file_offset:
            continue

        afib_start_global_idx = file_offset[ep.start_file_index] + ep.start_rr_index

        for horizon_min in HORIZON_MINUTES:
            target_sec = horizon_min * 60
            seg = extract_pre_afib_segment(rr_all, afib_start_global_idx, target_sec)
            feat = rr_features(seg)

            sufficient = feat["duration_sec"] >= target_sec * 0.95
            score, ml_ready = quality_score(feat, target_sec)

            rows.append(
                {
                    "patient": ep.patient,
                    "episode_index": ep.episode_index,
                    "start_file_index": ep.start_file_index,
                    "horizon_min": horizon_min,
                    "target_duration_sec": target_sec,
                    "afib_start_global_idx": afib_start_global_idx,
                    "is_sufficient_duration": bool(sufficient),
                    "quality_score": score,
                    "ml_ready": bool(ml_ready),
                    "af_duration_sec": ep.af_duration_sec,
                    "nsr_before_sec": ep.nsr_before_sec,
                    "outlier_pct_phys_nonempty": feat["outlier_pct_phys"] if feat["rr_count"] > 0 else np.nan,
                    "outlier_pct_iqr_nonempty": feat["outlier_pct_iqr"] if feat["rr_count"] > 0 else np.nan,
                    **feat,
                }
            )

        if (idx + 1) % 25 == 0 or idx + 1 == len(episodes):
            print(f"Processed episodes: {idx + 1}/{len(episodes)}")

    df = pd.DataFrame(rows)
    df["segment_start_global_idx"] = (df["afib_start_global_idx"] - df["rr_count"]).clip(lower=0).astype(int)
    df["segment_end_global_idx"] = df["afib_start_global_idx"].astype(int)  # exclusive end index
    df["segment_global_index_range"] = (
        df["segment_start_global_idx"].astype(str) + ":" + df["segment_end_global_idx"].astype(str)
    )
    df.sort_values(["patient", "episode_index", "horizon_min"], inplace=True)
    df.to_csv(OUT_DIR / "episode_pre_afib_features.csv", index=False)

    # A compact CSV optimized for re-loading exact RR segments later.
    segment_traceability_cols = [
        "patient",
        "episode_index",
        "horizon_min",
        "start_file_index",
        "afib_start_global_idx",
        "segment_start_global_idx",
        "segment_end_global_idx",
        "segment_global_index_range",
        "target_duration_sec",
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
        "rr_diff_std_ms",
        "trend_slope_ms_per_beat",
        "quality_score",
        "ml_ready",
    ]
    df[segment_traceability_cols].to_csv(OUT_DIR / "episode_segment_traceability.csv", index=False)

    generate_plots(df)
    write_report(df)

    quick_summary = {
        "n_episodes": int(df[["patient", "episode_index"]].drop_duplicates().shape[0]),
        "horizons_min": HORIZON_MINUTES,
        "rows": int(len(df)),
        "overall_ml_ready_pct": float(df["ml_ready"].mean() * 100.0),
        "output_dir": str(OUT_DIR),
    }
    with open(OUT_DIR / "run_summary.json", "w") as f:
        json.dump(quick_summary, f, indent=2)

    print("\nOutputs:")
    print(f"- {OUT_DIR / 'episode_pre_afib_features.csv'}")
    print(f"- {OUT_DIR / 'episode_segment_traceability.csv'}")
    print(f"- {OUT_DIR / 'horizon_summary.csv'}")
    print(f"- {OUT_DIR / 'IRIDIA_PreAFib_Episode_Report.md'}")
    print(f"- {PLOTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()

