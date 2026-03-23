"""
IRIDIA AF Dataset - Comprehensive Statistical Analysis
=======================================================
Loads the IRIDIA atrial fibrillation dataset, computes per-patient and
dataset-wide statistics on RR intervals and AF episodes, generates
visualizations, and writes a Markdown report to dataset_experiments/Results/.
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "Results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("deep")


def load_all_patient_data(dataset_path: str) -> pd.DataFrame:
    """Iterate over every patient directory and collect summary statistics."""
    patient_dirs = sorted(os.listdir(dataset_path))
    rows = []

    for patient in patient_dirs:
        record_dir = os.path.join(dataset_path, patient)
        if not os.path.isdir(record_dir):
            continue

        ecg_label_file = os.path.join(record_dir, f"{patient}_ecg_labels.csv")
        rr_label_file = os.path.join(record_dir, f"{patient}_rr_labels.csv")

        if not os.path.exists(ecg_label_file) or not os.path.exists(rr_label_file):
            continue

        ecg_df = pd.read_csv(ecg_label_file)
        rr_label_df = pd.read_csv(rr_label_file)

        rr_files = sorted(
            [f for f in os.listdir(record_dir) if f.endswith(".h5") and "_rr_" in f]
        )

        all_rr = []
        for rr_f in rr_files:
            with h5py.File(os.path.join(record_dir, rr_f), "r") as hf:
                all_rr.append(hf["rr"][:])
        if not all_rr:
            continue
        rr_data = np.concatenate(all_rr)

        ecg_files = sorted(
            [f for f in os.listdir(record_dir) if f.endswith(".h5") and "_ecg_" in f]
        )
        total_ecg_samples = 0
        n_ecg_leads = 0
        for ef in ecg_files:
            with h5py.File(os.path.join(record_dir, ef), "r") as hf:
                ecg_shape = hf["ecg"].shape
                total_ecg_samples += ecg_shape[0]
                n_ecg_leads = ecg_shape[1] if len(ecg_shape) > 1 else 1

        n_episodes = len(ecg_df)
        af_durations = ecg_df["af_duration"].values  # in ms
        nsr_before_durations = ecg_df["nsr_before_duration"].values

        rows.append(
            {
                "patient": patient,
                "n_af_episodes": n_episodes,
                "total_rr_intervals": len(rr_data),
                "rr_mean_ms": rr_data.mean(),
                "rr_median_ms": np.median(rr_data),
                "rr_std_ms": rr_data.std(),
                "rr_min_ms": rr_data.min(),
                "rr_max_ms": rr_data.max(),
                "rr_iqr_ms": np.percentile(rr_data, 75) - np.percentile(rr_data, 25),
                "rr_q25_ms": np.percentile(rr_data, 25),
                "rr_q75_ms": np.percentile(rr_data, 75),
                "mean_hr_bpm": 60000.0 / rr_data.mean(),
                "total_recording_duration_sec": rr_data.sum() / 1000.0,
                "total_af_duration_sec": af_durations.sum() / 1000.0,
                "mean_af_episode_duration_sec": af_durations.mean() / 1000.0,
                "max_af_episode_duration_sec": af_durations.max() / 1000.0,
                "min_af_episode_duration_sec": af_durations.min() / 1000.0,
                "mean_nsr_before_sec": nsr_before_durations.mean() / 1000.0,
                "total_ecg_samples": total_ecg_samples,
                "n_ecg_leads": n_ecg_leads,
                "n_rr_files": len(rr_files),
                "n_ecg_files": len(ecg_files),
            }
        )
        sys.stdout.write(f"\r  Loaded {len(rows)}/{len(patient_dirs)} patients")
        sys.stdout.flush()

    print()
    return pd.DataFrame(rows)


def collect_all_rr_intervals(dataset_path: str, sample_fraction: float = 0.1) -> np.ndarray:
    """Collect a random sample of RR intervals across all patients for distribution plots."""
    patient_dirs = sorted(os.listdir(dataset_path))
    sampled = []
    for patient in patient_dirs:
        record_dir = os.path.join(dataset_path, patient)
        if not os.path.isdir(record_dir):
            continue
        rr_files = sorted(
            [f for f in os.listdir(record_dir) if f.endswith(".h5") and "_rr_" in f]
        )
        for rr_f in rr_files:
            with h5py.File(os.path.join(record_dir, rr_f), "r") as hf:
                rr = hf["rr"][:]
                n_sample = max(1, int(len(rr) * sample_fraction))
                idx = np.random.choice(len(rr), size=n_sample, replace=False)
                sampled.append(rr[idx])
    return np.concatenate(sampled)


def collect_af_sr_rr_intervals(dataset_path: str) -> tuple:
    """Collect RR intervals separated into AF segments and SR segments."""
    patient_dirs = sorted(os.listdir(dataset_path))
    af_rr_all, sr_rr_all = [], []

    for patient in patient_dirs:
        record_dir = os.path.join(dataset_path, patient)
        if not os.path.isdir(record_dir):
            continue

        rr_label_file = os.path.join(record_dir, f"{patient}_rr_labels.csv")
        if not os.path.exists(rr_label_file):
            continue
        rr_label_df = pd.read_csv(rr_label_file)

        rr_files_map = {}
        for f in os.listdir(record_dir):
            if f.endswith(".h5") and "_rr_" in f:
                idx = int(f.split("_rr_")[1].replace(".h5", ""))
                rr_files_map[idx] = f

        for _, row in rr_label_df.iterrows():
            fi = row["start_file_index"]
            if fi not in rr_files_map:
                continue
            with h5py.File(os.path.join(record_dir, rr_files_map[fi]), "r") as hf:
                rr = hf["rr"][:]
            start_idx = row["start_rr_index"]
            end_idx = row["end_rr_index"]
            af_seg = rr[start_idx:end_idx]
            sr_seg = rr[:start_idx]
            n_af = max(1, int(len(af_seg) * 0.15))
            n_sr = max(1, int(len(sr_seg) * 0.15))
            if len(af_seg) > 0:
                af_rr_all.append(af_seg[np.random.choice(len(af_seg), n_af, replace=False)])
            if len(sr_seg) > 0:
                sr_rr_all.append(sr_seg[np.random.choice(len(sr_seg), n_sr, replace=False)])

    af_rr = np.concatenate(af_rr_all) if af_rr_all else np.array([])
    sr_rr = np.concatenate(sr_rr_all) if sr_rr_all else np.array([])
    return af_rr, sr_rr


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_rr_distribution(rr_sample: np.ndarray, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rr_sample, bins=150, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("RR Interval (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of RR Intervals (Sampled)")
    axes[0].axvline(np.median(rr_sample), color=PALETTE[3], ls="--", lw=1.5, label=f"Median={np.median(rr_sample):.0f} ms")
    axes[0].legend()

    axes[1].hist(rr_sample, bins=150, color=PALETTE[1], edgecolor="white", alpha=0.85, cumulative=True, density=True)
    axes[1].set_xlabel("RR Interval (ms)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("CDF of RR Intervals")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_af_vs_sr_rr(af_rr: np.ndarray, sr_rr: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(100, 2500, 200)
    ax.hist(sr_rr, bins=bins, alpha=0.6, label=f"SR (n={len(sr_rr):,})", color=PALETTE[2], density=True)
    ax.hist(af_rr, bins=bins, alpha=0.6, label=f"AF (n={len(af_rr):,})", color=PALETTE[3], density=True)
    ax.set_xlabel("RR Interval (ms)")
    ax.set_ylabel("Density")
    ax.set_title("RR Interval Distribution: AF vs Sinus Rhythm")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_patient_stats(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    axes[0, 0].bar(range(len(df)), df["n_af_episodes"].values, color=PALETTE[0], alpha=0.8)
    axes[0, 0].set_xlabel("Patient Index")
    axes[0, 0].set_ylabel("Number of AF Episodes")
    axes[0, 0].set_title("AF Episodes per Patient")

    axes[0, 1].bar(range(len(df)), df["total_recording_duration_sec"].values / 3600, color=PALETTE[1], alpha=0.8)
    axes[0, 1].set_xlabel("Patient Index")
    axes[0, 1].set_ylabel("Duration (hours)")
    axes[0, 1].set_title("Total Recording Duration per Patient")

    axes[1, 0].bar(range(len(df)), df["mean_hr_bpm"].values, color=PALETTE[2], alpha=0.8)
    axes[1, 0].set_xlabel("Patient Index")
    axes[1, 0].set_ylabel("Heart Rate (bpm)")
    axes[1, 0].set_title("Mean Heart Rate per Patient")

    axes[1, 1].bar(range(len(df)), df["total_af_duration_sec"].values / 60, color=PALETTE[3], alpha=0.8)
    axes[1, 1].set_xlabel("Patient Index")
    axes[1, 1].set_ylabel("Duration (minutes)")
    axes[1, 1].set_title("Total AF Duration per Patient")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rr_boxplots(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    box_data = [df["rr_mean_ms"], df["rr_median_ms"], df["rr_std_ms"]]
    labels = ["Mean RR (ms)", "Median RR (ms)", "Std RR (ms)"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[4]]
    for ax, data, label, c in zip(axes, box_data, labels, colors):
        bp = ax.boxplot(data, patch_artist=True, vert=True)
        bp["boxes"][0].set_facecolor(c)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_ylabel(label)
        ax.set_title(f"Distribution of {label} Across Patients")
        ax.set_xticks([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_af_episode_analysis(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(df["n_af_episodes"], bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Number of AF Episodes")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_title("Distribution of AF Episode Count")

    af_dur_min = df["mean_af_episode_duration_sec"] / 60
    axes[1].hist(af_dur_min, bins=30, color=PALETTE[3], edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Mean AF Episode Duration (min)")
    axes[1].set_ylabel("Number of Patients")
    axes[1].set_title("Distribution of Mean AF Episode Duration")

    nsr_dur_min = df["mean_nsr_before_sec"] / 60
    axes[2].hist(nsr_dur_min, bins=30, color=PALETTE[2], edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("Mean NSR Before AF (min)")
    axes[2].set_ylabel("Number of Patients")
    axes[2].set_title("Distribution of Mean NSR Duration Before AF")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path):
    cols = [
        "n_af_episodes", "rr_mean_ms", "rr_std_ms", "rr_iqr_ms",
        "mean_hr_bpm", "total_recording_duration_sec",
        "total_af_duration_sec", "mean_af_episode_duration_sec",
        "mean_nsr_before_sec", "total_rr_intervals",
    ]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hr_vs_af_burden(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    af_burden_pct = (df["total_af_duration_sec"] / df["total_recording_duration_sec"]) * 100
    axes[0].scatter(df["mean_hr_bpm"], af_burden_pct, alpha=0.6, color=PALETTE[0], edgecolors="white", s=50)
    axes[0].set_xlabel("Mean Heart Rate (bpm)")
    axes[0].set_ylabel("AF Burden (%)")
    axes[0].set_title("Heart Rate vs AF Burden")

    axes[1].scatter(df["rr_std_ms"], af_burden_pct, alpha=0.6, color=PALETTE[3], edgecolors="white", s=50)
    axes[1].set_xlabel("RR Interval Std Dev (ms)")
    axes[1].set_ylabel("AF Burden (%)")
    axes[1].set_title("RR Variability vs AF Burden")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recording_overview(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df["total_rr_intervals"], bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0, 0].set_xlabel("Total RR Intervals")
    axes[0, 0].set_ylabel("Number of Patients")
    axes[0, 0].set_title("Distribution of Total RR Intervals per Patient")

    axes[0, 1].hist(df["total_recording_duration_sec"] / 3600, bins=30, color=PALETTE[1], edgecolor="white", alpha=0.85)
    axes[0, 1].set_xlabel("Recording Duration (hours)")
    axes[0, 1].set_ylabel("Number of Patients")
    axes[0, 1].set_title("Distribution of Recording Duration")

    axes[1, 0].hist(df["mean_hr_bpm"], bins=30, color=PALETTE[2], edgecolor="white", alpha=0.85)
    axes[1, 0].set_xlabel("Mean Heart Rate (bpm)")
    axes[1, 0].set_ylabel("Number of Patients")
    axes[1, 0].set_title("Distribution of Mean Heart Rate")

    axes[1, 1].hist(df["rr_iqr_ms"], bins=30, color=PALETTE[4], edgecolor="white", alpha=0.85)
    axes[1, 1].set_xlabel("IQR of RR Intervals (ms)")
    axes[1, 1].set_ylabel("Number of Patients")
    axes[1, 1].set_title("Distribution of RR Interval IQR")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, all_rr_sample: np.ndarray, af_rr: np.ndarray, sr_rr: np.ndarray):
    af_burden_pct = (df["total_af_duration_sec"] / df["total_recording_duration_sec"]) * 100

    report_lines = []

    def add(line=""):
        report_lines.append(line)

    add("# IRIDIA AF Dataset — Comprehensive Statistical Report")
    add()
    add(f"> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add()
    add("---")
    add()

    # ---- 1. Overview ----
    add("## 1. Dataset Overview")
    add()
    add(f"| Property | Value |")
    add(f"|---|---|")
    add(f"| Dataset Name | IRIDIA Atrial Fibrillation Records v1.0.1 |")
    add(f"| Dataset Path | `{DATASET_PATH}` |")
    add(f"| Total Patients | {len(df)} |")
    add(f"| Total AF Episodes | {df['n_af_episodes'].sum()} |")
    add(f"| Total RR Intervals | {df['total_rr_intervals'].sum():,} |")
    add(f"| Total Recording Duration | {df['total_recording_duration_sec'].sum()/3600:.1f} hours |")
    add(f"| Total AF Duration | {df['total_af_duration_sec'].sum()/3600:.1f} hours |")
    add(f"| ECG Leads | {df['n_ecg_leads'].mode().values[0]} |")
    add(f"| Total ECG Samples | {df['total_ecg_samples'].sum():,} |")
    add()

    # ---- 2. Per-Patient Summary ----
    add("## 2. Per-Patient Summary Statistics")
    add()
    add("### 2.1 Recording Characteristics")
    add()
    add(f"| Statistic | Mean | Median | Std | Min | Max |")
    add(f"|---|---|---|---|---|---|")
    for col, label, scale in [
        ("total_recording_duration_sec", "Recording Duration (hours)", 1/3600),
        ("total_rr_intervals", "Total RR Intervals", 1),
        ("n_af_episodes", "AF Episodes", 1),
        ("n_rr_files", "RR Files per Patient", 1),
    ]:
        vals = df[col] * scale
        add(f"| {label} | {vals.mean():.2f} | {vals.median():.2f} | {vals.std():.2f} | {vals.min():.2f} | {vals.max():.2f} |")
    add()

    add("### 2.2 RR Interval Statistics (across patients)")
    add()
    add(f"| Statistic | Mean | Median | Std | Min | Max |")
    add(f"|---|---|---|---|---|---|")
    for col, label in [
        ("rr_mean_ms", "Mean RR (ms)"),
        ("rr_median_ms", "Median RR (ms)"),
        ("rr_std_ms", "Std Dev RR (ms)"),
        ("rr_iqr_ms", "IQR of RR (ms)"),
        ("rr_min_ms", "Min RR (ms)"),
        ("rr_max_ms", "Max RR (ms)"),
        ("mean_hr_bpm", "Mean HR (bpm)"),
    ]:
        vals = df[col]
        add(f"| {label} | {vals.mean():.1f} | {vals.median():.1f} | {vals.std():.1f} | {vals.min():.1f} | {vals.max():.1f} |")
    add()

    # ---- 3. AF Episode Analysis ----
    add("## 3. Atrial Fibrillation Episode Analysis")
    add()
    add(f"| Statistic | Mean | Median | Std | Min | Max |")
    add(f"|---|---|---|---|---|---|")
    for col, label in [
        ("mean_af_episode_duration_sec", "Mean AF Episode Duration (sec)"),
        ("max_af_episode_duration_sec", "Max AF Episode Duration (sec)"),
        ("min_af_episode_duration_sec", "Min AF Episode Duration (sec)"),
        ("total_af_duration_sec", "Total AF Duration (sec)"),
        ("mean_nsr_before_sec", "Mean NSR Before AF (sec)"),
    ]:
        vals = df[col]
        add(f"| {label} | {vals.mean():.1f} | {vals.median():.1f} | {vals.std():.1f} | {vals.min():.1f} | {vals.max():.1f} |")
    add()

    add("### 3.1 AF Burden")
    add()
    add(f"| Statistic | Value |")
    add(f"|---|---|")
    add(f"| Mean AF Burden (%) | {af_burden_pct.mean():.2f} |")
    add(f"| Median AF Burden (%) | {af_burden_pct.median():.2f} |")
    add(f"| Std AF Burden (%) | {af_burden_pct.std():.2f} |")
    add(f"| Min AF Burden (%) | {af_burden_pct.min():.2f} |")
    add(f"| Max AF Burden (%) | {af_burden_pct.max():.2f} |")
    add()

    # ---- 4. RR Interval Distribution ----
    add("## 4. Global RR Interval Distribution (Sampled)")
    add()
    add(f"| Statistic | Value |")
    add(f"|---|---|")
    add(f"| Sample Size | {len(all_rr_sample):,} |")
    add(f"| Mean (ms) | {all_rr_sample.mean():.1f} |")
    add(f"| Median (ms) | {np.median(all_rr_sample):.1f} |")
    add(f"| Std Dev (ms) | {all_rr_sample.std():.1f} |")
    add(f"| 5th Percentile (ms) | {np.percentile(all_rr_sample, 5):.1f} |")
    add(f"| 25th Percentile (ms) | {np.percentile(all_rr_sample, 25):.1f} |")
    add(f"| 75th Percentile (ms) | {np.percentile(all_rr_sample, 75):.1f} |")
    add(f"| 95th Percentile (ms) | {np.percentile(all_rr_sample, 95):.1f} |")
    add(f"| Skewness | {pd.Series(all_rr_sample).skew():.3f} |")
    add(f"| Kurtosis | {pd.Series(all_rr_sample).kurtosis():.3f} |")
    add()

    # ---- 5. AF vs SR comparison ----
    add("## 5. RR Intervals: AF vs Sinus Rhythm")
    add()
    add(f"| Statistic | Sinus Rhythm | Atrial Fibrillation |")
    add(f"|---|---|---|")
    add(f"| Sample Size | {len(sr_rr):,} | {len(af_rr):,} |")
    add(f"| Mean (ms) | {sr_rr.mean():.1f} | {af_rr.mean():.1f} |")
    add(f"| Median (ms) | {np.median(sr_rr):.1f} | {np.median(af_rr):.1f} |")
    add(f"| Std Dev (ms) | {sr_rr.std():.1f} | {af_rr.std():.1f} |")
    add(f"| IQR (ms) | {np.percentile(sr_rr,75)-np.percentile(sr_rr,25):.1f} | {np.percentile(af_rr,75)-np.percentile(af_rr,25):.1f} |")
    add(f"| CoV | {sr_rr.std()/sr_rr.mean():.3f} | {af_rr.std()/af_rr.mean():.3f} |")
    add()
    add("> **Observation:** AF segments exhibit higher RR variability (larger Std Dev and IQR) compared to SR, consistent with the irregular ventricular response characteristic of atrial fibrillation.")
    add()

    # ---- 6. Visualizations ----
    add("## 6. Visualizations")
    add()
    plot_names = [
        ("01_rr_distribution.png", "Global RR Interval Distribution"),
        ("02_af_vs_sr_rr.png", "AF vs SR RR Interval Distribution"),
        ("03_per_patient_stats.png", "Per-Patient Key Statistics"),
        ("04_rr_boxplots.png", "RR Interval Boxplots Across Patients"),
        ("05_af_episode_analysis.png", "AF Episode Analysis"),
        ("06_correlation_heatmap.png", "Feature Correlation Heatmap"),
        ("07_hr_vs_af_burden.png", "Heart Rate and Variability vs AF Burden"),
        ("08_recording_overview.png", "Recording Overview Distributions"),
    ]
    for fname, title in plot_names:
        add(f"### {title}")
        add()
        add(f"![{title}](plots/{fname})")
        add()

    # ---- 7. Key Findings ----
    add("## 7. Key Findings")
    add()
    add(f"1. The dataset contains **{len(df)} patients** with a total of **{df['n_af_episodes'].sum()} AF episodes**.")
    add(f"2. Mean recording duration per patient is **{df['total_recording_duration_sec'].mean()/3600:.1f} hours** (range: {df['total_recording_duration_sec'].min()/3600:.1f}–{df['total_recording_duration_sec'].max()/3600:.1f} hours).")
    add(f"3. Average heart rate across patients is **{df['mean_hr_bpm'].mean():.1f} bpm** (range: {df['mean_hr_bpm'].min():.1f}–{df['mean_hr_bpm'].max():.1f} bpm).")
    add(f"4. Mean AF episode duration is **{df['mean_af_episode_duration_sec'].mean():.1f} seconds** ({df['mean_af_episode_duration_sec'].mean()/60:.1f} minutes).")
    add(f"5. Mean AF burden across patients is **{af_burden_pct.mean():.2f}%** (range: {af_burden_pct.min():.2f}%–{af_burden_pct.max():.2f}%).")
    add(f"6. AF segments show **{af_rr.std()/af_rr.mean()*100:.1f}%** coefficient of variation vs **{sr_rr.std()/sr_rr.mean()*100:.1f}%** in SR, confirming greater irregularity during AF.")
    add(f"7. Mean NSR duration before AF onset is **{df['mean_nsr_before_sec'].mean()/60:.1f} minutes**, providing context for AF prediction tasks.")
    add()

    add("---")
    add()
    add("*Report generated by `dataset_experiments/IRIDIA/stats.py`*")

    report_text = "\n".join(report_lines)
    report_path = RESULTS_DIR / "IRIDIA_Dataset_Report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    print("=" * 60)
    print("IRIDIA AF Dataset — Statistical Analysis")
    print("=" * 60)

    print("\n[1/5] Loading per-patient statistics...")
    df = load_all_patient_data(DATASET_PATH)
    print(f"  Loaded data for {len(df)} patients.")

    csv_path = RESULTS_DIR / "per_patient_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved per-patient stats to {csv_path}")

    json_summary = {
        "n_patients": int(len(df)),
        "total_af_episodes": int(df["n_af_episodes"].sum()),
        "total_rr_intervals": int(df["total_rr_intervals"].sum()),
        "total_recording_hours": round(df["total_recording_duration_sec"].sum() / 3600, 2),
        "total_af_hours": round(df["total_af_duration_sec"].sum() / 3600, 2),
        "mean_hr_bpm": round(df["mean_hr_bpm"].mean(), 2),
        "mean_af_burden_pct": round(
            ((df["total_af_duration_sec"] / df["total_recording_duration_sec"]) * 100).mean(), 2
        ),
    }
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    print("\n[2/5] Sampling global RR intervals...")
    all_rr_sample = collect_all_rr_intervals(DATASET_PATH, sample_fraction=0.1)
    print(f"  Sampled {len(all_rr_sample):,} RR intervals.")

    print("\n[3/5] Collecting AF vs SR RR intervals...")
    af_rr, sr_rr = collect_af_sr_rr_intervals(DATASET_PATH)
    print(f"  AF: {len(af_rr):,} intervals, SR: {len(sr_rr):,} intervals.")

    print("\n[4/5] Generating plots...")
    plot_rr_distribution(all_rr_sample, PLOTS_DIR / "01_rr_distribution.png")
    print("  [1/8] RR distribution")
    plot_af_vs_sr_rr(af_rr, sr_rr, PLOTS_DIR / "02_af_vs_sr_rr.png")
    print("  [2/8] AF vs SR")
    plot_per_patient_stats(df, PLOTS_DIR / "03_per_patient_stats.png")
    print("  [3/8] Per-patient stats")
    plot_rr_boxplots(df, PLOTS_DIR / "04_rr_boxplots.png")
    print("  [4/8] RR boxplots")
    plot_af_episode_analysis(df, PLOTS_DIR / "05_af_episode_analysis.png")
    print("  [5/8] AF episode analysis")
    plot_correlation_heatmap(df, PLOTS_DIR / "06_correlation_heatmap.png")
    print("  [6/8] Correlation heatmap")
    plot_hr_vs_af_burden(df, PLOTS_DIR / "07_hr_vs_af_burden.png")
    print("  [7/8] HR vs AF burden")
    plot_recording_overview(df, PLOTS_DIR / "08_recording_overview.png")
    print("  [8/8] Recording overview")

    print("\n[5/5] Generating report...")
    generate_report(df, all_rr_sample, af_rr, sr_rr)

    print("\n" + "=" * 60)
    print("Analysis complete! Outputs:")
    print(f"  Report:     {RESULTS_DIR / 'IRIDIA_Dataset_Report.md'}")
    print(f"  Stats CSV:  {RESULTS_DIR / 'per_patient_stats.csv'}")
    print(f"  Summary:    {RESULTS_DIR / 'dataset_summary.json'}")
    print(f"  Plots:      {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
