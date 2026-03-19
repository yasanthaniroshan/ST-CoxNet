"""
Debug v2: Patient-Aware Signal Analysis
========================================
Extends debug_time_signal.py with:
  1. Patient-level grouped RF CV  (was the 82% result just leakage?)
  2. Within-patient Spearman      (does signal exist per-patient?)
  3. Patient-normalized features   (remove baseline, isolate dynamics)
  4. Mixed-effects model           (proper statistical test)

Requires the HRV feature cache from debug_time_signal.py.
"""

import os
import sys
import json
import hashlib
import random
import pickle

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots/debug_signal"
os.makedirs(PLOTS_DIR, exist_ok=True)

BIN_NAMES = ["Near (0-33%)", "Mid (33-67%)", "Far (67-100%)"]
BIN_EDGES = [0.0, 1 / 3, 2 / 3, 1.01]

PROCESSED_PATH = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
DATASET_PARAMS = dict(
    afib_length=60 * 60,
    sr_length=int(1.5 * 60 * 60),
    number_of_windows_in_segment=10,
    stride=20,
    window_size=100,
    validation_split=0.15,
)

FEATURE_NAMES = [
    "artifact_rate", "mean", "std", "median", "iqr", "p5", "p95",
    "skewness", "kurtosis", "cv", "rmssd", "sdsd", "pnn50_scaled",
    "alpha1", "sample_entropy",
]


# ======================================================================
# 1. Recover patient IDs from the h5 file
# ======================================================================


def get_dataset_hash():
    dataset_prop = {
        "dataset_name": "IRIDIA AFIB Dataset",
        "AFIB_length_seconds": DATASET_PARAMS["afib_length"],
        "SR_length_seconds": DATASET_PARAMS["sr_length"],
        "window_size": DATASET_PARAMS["window_size"],
        "segment_size": DATASET_PARAMS["number_of_windows_in_segment"] * DATASET_PARAMS["window_size"],
        "stride": DATASET_PARAMS["stride"],
        "validation_split": DATASET_PARAMS["validation_split"],
        "scaler": "RobustScaler",
    }
    return hashlib.sha256(json.dumps(dataset_prop, sort_keys=True).encode()).hexdigest()[:32]


def recover_patient_ids(split="train"):
    """Detect patient boundaries from the time array in the h5 file.

    Within a patient's segments (ordered by position in recording):
      - SR segments have monotonically decreasing times
      - Mixed/AFib segments have time=0
    When a new patient starts, the time jumps back up.

    After filtering to SR-only (label==-1), a new patient boundary is where
    time[i] > time[i-1] (time increases).
    """
    h = get_dataset_hash()
    h5_path = os.path.join(PROCESSED_PATH, f"{h}_{split}.h5")

    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
        times = f["times"][:]

    sr_mask = labels == -1
    sr_times = times[sr_mask]
    n = len(sr_times)

    patient_ids = np.zeros(n, dtype=np.int32)
    pid = 0
    for i in range(1, n):
        if sr_times[i] > sr_times[i - 1] + 1e-6:
            pid += 1
        patient_ids[i] = pid

    n_patients = pid + 1
    return patient_ids, n_patients, sr_times


def load_features_and_labels():
    """Load cached HRV features and bin labels from the Phase 1 run."""
    cache_path = os.path.join(PLOTS_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found. Run debug_time_signal.py first.")
        sys.exit(1)

    cached = np.load(cache_path, allow_pickle=True)
    X = cached["X"]
    feature_names = list(cached["feature_names"])

    h = get_dataset_hash()
    h5_path = os.path.join(PROCESSED_PATH, f"{h}_train.h5")
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
        times = f["times"][:]

    sr_mask = labels == -1
    sr_times = times[sr_mask]

    bin_labels = np.digitize(sr_times, np.array(BIN_EDGES, dtype=np.float64)) - 1
    num_bins = len(BIN_EDGES) - 1
    bin_labels = np.clip(bin_labels, 0, num_bins - 1)

    return X, feature_names, bin_labels, sr_times


# ======================================================================
# Experiment 1: Patient-Level Grouped CV
# ======================================================================


def experiment_patient_cv(X, feature_names, bin_labels, patient_ids, n_patients):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Patient-Grouped RF CV (no data leakage)")
    print("=" * 70)
    print(f"  {len(X)} segments, {n_patients} patients", flush=True)

    segs_per_patient = np.bincount(patient_ids)
    print(f"  Segments/patient: median={np.median(segs_per_patient):.0f}, "
          f"min={segs_per_patient.min()}, max={segs_per_patient.max()}", flush=True)

    n_splits = min(5, n_patients)
    gkf = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ])

    scores = cross_validate(
        pipe, X, bin_labels, cv=gkf, groups=patient_ids,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )

    print(f"\n  Patient-Grouped {n_splits}-fold CV (RF on {len(feature_names)} HRV features):")
    print(f"    Train acc:  {scores['train_accuracy'].mean():.4f} +/- {scores['train_accuracy'].std():.4f}")
    print(f"    Val acc:    {scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores['test_f1_macro'].mean():.4f} +/- {scores['test_f1_macro'].std():.4f}")

    # Compare with segment-level CV for reference
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_seg = cross_validate(
        pipe, X, bin_labels, cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(f"\n  Segment-level 5-fold CV (same RF, for comparison):")
    print(f"    Val acc:    {scores_seg['test_accuracy'].mean():.4f} +/- {scores_seg['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores_seg['test_f1_macro'].mean():.4f} +/- {scores_seg['test_f1_macro'].std():.4f}")

    print(f"\n  >>> Segment-level acc: {scores_seg['test_accuracy'].mean():.1%}")
    print(f"  >>> Patient-level acc: {scores['test_accuracy'].mean():.1%}")
    print(f"  >>> Random baseline:   33.3%")

    return scores, scores_seg


# ======================================================================
# Experiment 2: Within-Patient Spearman Correlation
# ======================================================================


def experiment_within_patient_spearman(X, feature_names, sr_times, patient_ids, n_patients):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Within-Patient Spearman Correlation")
    print("=" * 70)

    min_segs = 10
    per_patient_rhos = {fn: [] for fn in feature_names}

    for pid in range(n_patients):
        mask = patient_ids == pid
        if mask.sum() < min_segs:
            continue
        t = sr_times[mask]
        for fi, fn in enumerate(feature_names):
            vals = X[mask, fi]
            if np.std(vals) < 1e-10:
                continue
            rho, _ = spearmanr(vals, t)
            if not np.isnan(rho):
                per_patient_rhos[fn].append(rho)

    print(f"\n  Patients with >= {min_segs} SR segments: "
          f"{len(per_patient_rhos[feature_names[0]])}/{n_patients}")
    print(f"\n  {'Feature':<20} {'Mean rho':>10} {'Median rho':>12} {'Std':>8} {'% pos':>8}")
    print("  " + "-" * 60)

    summary_data = []
    for fn in feature_names:
        rhos = np.array(per_patient_rhos[fn])
        if len(rhos) == 0:
            continue
        mean_rho = np.mean(rhos)
        med_rho = np.median(rhos)
        std_rho = np.std(rhos)
        pct_pos = np.mean(rhos > 0) * 100
        print(f"  {fn:<20} {mean_rho:>10.4f} {med_rho:>12.4f} {std_rho:>8.4f} {pct_pos:>7.1f}%")
        summary_data.append((fn, rhos))

    # Plot distribution of per-patient rhos for top features
    top_features = ["alpha1", "p95", "mean", "rmssd", "artifact_rate"]
    top_features = [f for f in top_features if f in per_patient_rhos and len(per_patient_rhos[f]) > 0]

    if top_features:
        fig, axes = plt.subplots(1, len(top_features), figsize=(4 * len(top_features), 4))
        if len(top_features) == 1:
            axes = [axes]
        for ax, fn in zip(axes, top_features):
            rhos = np.array(per_patient_rhos[fn])
            ax.hist(rhos, bins=20, alpha=0.7, edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", alpha=0.5)
            ax.axvline(np.mean(rhos), color="blue", linestyle="-", linewidth=2, label=f"mean={np.mean(rhos):.3f}")
            ax.set_title(fn)
            ax.set_xlabel("Spearman rho")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("Per-Patient Spearman Correlations (feature vs time-to-event)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(PLOTS_DIR, "within_patient_spearman.png"), dpi=200)
        plt.close(fig)
        print("  Saved within_patient_spearman.png")


# ======================================================================
# Experiment 3: Patient-Normalized Features + RF
# ======================================================================


def experiment_patient_normalized(X, feature_names, bin_labels, patient_ids, n_patients):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Patient-Normalized Features + RF")
    print("=" * 70)

    X_norm = np.copy(X)
    for pid in range(n_patients):
        mask = patient_ids == pid
        if mask.sum() < 2:
            continue
        p_mean = X[mask].mean(axis=0)
        p_std = X[mask].std(axis=0)
        p_std[p_std < 1e-10] = 1.0
        X_norm[mask] = (X[mask] - p_mean) / p_std

    nan_mask = np.isnan(X_norm)
    for col in range(X_norm.shape[1]):
        col_nan = nan_mask[:, col]
        if col_nan.any():
            X_norm[col_nan, col] = 0.0

    n_splits = min(5, n_patients)
    gkf = GroupKFold(n_splits=n_splits)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ])

    scores_norm = cross_validate(
        pipe, X_norm, bin_labels, cv=gkf, groups=patient_ids,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )

    print(f"\n  Patient-Normalized + Patient-Grouped CV:")
    print(f"    Train acc:  {scores_norm['train_accuracy'].mean():.4f} +/- {scores_norm['train_accuracy'].std():.4f}")
    print(f"    Val acc:    {scores_norm['test_accuracy'].mean():.4f} +/- {scores_norm['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores_norm['test_f1_macro'].mean():.4f} +/- {scores_norm['test_f1_macro'].std():.4f}")

    # Also try segment-level CV on normalized features (to check if normalization helps at all)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_norm_seg = cross_validate(
        pipe, X_norm, bin_labels, cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(f"\n  Patient-Normalized + Segment-level CV:")
    print(f"    Val acc:    {scores_norm_seg['test_accuracy'].mean():.4f} +/- {scores_norm_seg['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores_norm_seg['test_f1_macro'].mean():.4f} +/- {scores_norm_seg['test_f1_macro'].std():.4f}")

    # ANOVA on normalized features
    from scipy.stats import f_oneway
    print(f"\n  ANOVA on patient-normalized features:")
    print(f"  {'Feature':<20} {'F':>10} {'p':>12} {'Sig?':>6}")
    print("  " + "-" * 50)
    for fi, fn in enumerate(feature_names):
        groups = [X_norm[bin_labels == c, fi] for c in range(len(BIN_NAMES))]
        f_stat, f_p = f_oneway(*groups)
        sig = "**" if f_p < 0.001 else ("*" if f_p < 0.05 else "")
        print(f"  {fn:<20} {f_stat:>10.2f} {f_p:>12.2e} {sig:>6}")

    return scores_norm, X_norm


# ======================================================================
# Experiment 4: Linear Mixed-Effects Model
# ======================================================================


def experiment_mixed_effects(X, feature_names, sr_times, patient_ids):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Linear Mixed-Effects Model")
    print("=" * 70)

    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError:
        print("  statsmodels not available, skipping mixed-effects analysis")
        return

    print(f"  Testing: feature ~ time_to_event + (1 | patient)")
    print(f"\n  {'Feature':<20} {'Coeff':>10} {'z':>10} {'p':>12} {'Sig?':>6}")
    print("  " + "-" * 60)

    for fi, fn in enumerate(feature_names):
        df = pd.DataFrame({
            "feature": X[:, fi],
            "time": sr_times,
            "patient": patient_ids.astype(str),
        })
        try:
            model = smf.mixedlm("feature ~ time", df, groups=df["patient"])
            result = model.fit(reml=True, method="lbfgs", maxiter=200)
            coeff = result.params["time"]
            z = result.tvalues["time"]
            p = result.pvalues["time"]
            sig = "**" if p < 0.001 else ("*" if p < 0.05 else "")
            print(f"  {fn:<20} {coeff:>10.4f} {z:>10.2f} {p:>12.2e} {sig:>6}")
        except Exception as e:
            print(f"  {fn:<20} {'FAILED':>10} (convergence issue)")


# ======================================================================
# Summary
# ======================================================================


def print_summary(scores_patient, scores_segment, scores_norm):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Random baseline (3-class):                 33.3%")
    print(f"  NN v9 (CPC, patient-held-out):             38.4%")
    print(f"  NN v10 (CPC+HRV, patient-held-out):        33.4%")
    print()
    print(f"  RF segment-level CV (data leakage):         {scores_segment['test_accuracy'].mean():.1%}")
    print(f"  RF patient-grouped CV (no leakage):         {scores_patient['test_accuracy'].mean():.1%}")
    print(f"  RF patient-normalized + patient-grouped:    {scores_norm['test_accuracy'].mean():.1%}")
    print()

    patient_acc = scores_patient['test_accuracy'].mean()
    if patient_acc < 0.37:
        print("  CONCLUSION: Signal does NOT generalize across patients.")
        print("  The 82% RF result was inflated by data leakage from overlapping segments.")
        print("  Individual SR segments lack cross-patient temporal proximity information.")
    elif patient_acc < 0.45:
        print("  CONCLUSION: Weak cross-patient signal exists but is marginal.")
        print("  A sequence-level model or longer segments may help.")
    else:
        print("  CONCLUSION: Cross-patient signal EXISTS.")
        print("  The NN architecture is the bottleneck; consider ResNetEncoder or HRV branch.")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("Loading features and labels...", flush=True)
    X, feature_names, bin_labels, sr_times = load_features_and_labels()
    print(f"  Features: {X.shape}, labels: {bin_labels.shape}", flush=True)

    print("Recovering patient IDs...", flush=True)
    patient_ids, n_patients, _ = recover_patient_ids(split="train")
    print(f"  Detected {n_patients} patients from time boundaries", flush=True)

    assert len(patient_ids) == len(X), \
        f"Patient ID count {len(patient_ids)} != feature count {len(X)}"

    scores_patient, scores_segment = experiment_patient_cv(
        X, feature_names, bin_labels, patient_ids, n_patients,
    )

    experiment_within_patient_spearman(
        X, feature_names, sr_times, patient_ids, n_patients,
    )

    scores_norm, X_norm = experiment_patient_normalized(
        X, feature_names, bin_labels, patient_ids, n_patients,
    )

    experiment_mixed_effects(X, feature_names, sr_times, patient_ids)

    print_summary(scores_patient, scores_segment, scores_norm)

    print(f"\nAll plots saved to: {PLOTS_DIR}", flush=True)
