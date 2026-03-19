"""
Debug: Temporal Proximity Signal in SR Segments
================================================================
Phase 1: Compute features, violin plots, ANOVA, PCA/t-SNE
Phase 2: Classical ML baseline (RandomForest) with 5-fold CV
================================================================
Data is RobustScaler-transformed (not raw ms).
Artifact filtering uses statistical outlier detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

from Utils.Dataset.CPCTimeBinDataset import CPCTimeBinDataset

BIN_NAMES = ["Near (0-33%)", "Mid (33-67%)", "Far (67-100%)"]
BIN_EDGES = [0.0, 1 / 3, 2 / 3, 1.01]
PLOTS_DIR = "plots/debug_signal"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_KWARGS = dict(
    processed_dataset_path="/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets",
    afib_length=60 * 60,
    sr_length=int(1.5 * 60 * 60),
    number_of_windows_in_segment=10,
    stride=20,
    window_size=100,
    bin_edges=BIN_EDGES,
    validation_split=0.15,
)


def filter_segment(seg_flat, iqr_factor=5.0):
    """Remove statistical outliers (bad R-peak artifacts) from a flattened segment."""
    q1, q3 = np.percentile(seg_flat, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-8:
        return seg_flat
    lo = q1 - iqr_factor * iqr
    hi = q3 + iqr_factor * iqr
    mask = (seg_flat >= lo) & (seg_flat <= hi)
    return seg_flat[mask]


def compute_features(seg_flat_raw):
    """Compute robust + HRV-like features from a single flattened segment."""
    artifact_rate = 0.0
    filtered = filter_segment(seg_flat_raw)
    if len(seg_flat_raw) > 0:
        artifact_rate = 1.0 - len(filtered) / len(seg_flat_raw)

    rr = filtered if len(filtered) > 10 else seg_flat_raw

    feats = {}
    feats["artifact_rate"] = artifact_rate
    feats["mean"] = np.mean(rr)
    feats["std"] = np.std(rr)
    feats["median"] = np.median(rr)
    feats["iqr"] = np.percentile(rr, 75) - np.percentile(rr, 25)
    feats["p5"] = np.percentile(rr, 5)
    feats["p95"] = np.percentile(rr, 95)
    feats["skewness"] = float(np.mean(((rr - np.mean(rr)) / (np.std(rr) + 1e-8)) ** 3))
    feats["kurtosis"] = float(np.mean(((rr - np.mean(rr)) / (np.std(rr) + 1e-8)) ** 4))
    feats["cv"] = np.std(rr) / (np.abs(np.mean(rr)) + 1e-8)

    diffs = np.diff(rr)
    feats["rmssd"] = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0
    feats["sdsd"] = np.std(diffs) if len(diffs) > 0 else 0.0
    feats["pnn50_scaled"] = np.mean(np.abs(diffs) > 0.5) if len(diffs) > 0 else 0.0

    # DFA alpha1 (works on any scaling)
    feats["alpha1"] = _dfa_alpha1(rr)

    # Sample entropy
    feats["sample_entropy"] = _sample_entropy(rr, m=2, r_frac=0.2)

    return feats


def _dfa_alpha1(rr, scales=None):
    """Short-term DFA scaling exponent."""
    N = len(rr)
    if N < 16:
        return np.nan
    y = np.cumsum(rr - np.mean(rr))
    if scales is None:
        scales = np.arange(4, min(12, N // 4 + 1))
    if len(scales) < 2:
        return np.nan
    flucts = []
    for n in scales:
        segs = N // n
        if segs == 0:
            flucts.append(np.nan)
            continue
        rms_vals = []
        for i in range(segs):
            seg = y[i * n : (i + 1) * n]
            x = np.arange(n)
            poly = np.polyfit(x, seg, 1)
            trend = np.polyval(poly, x)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        flucts.append(np.mean(rms_vals))
    flucts = np.array(flucts)
    valid = ~np.isnan(flucts) & (flucts > 0)
    if valid.sum() < 2:
        return np.nan
    return np.polyfit(np.log(scales[valid]), np.log(flucts[valid]), 1)[0]


def _sample_entropy(rr, m=2, r_frac=0.2, max_len=100):
    """Sample entropy via fully vectorized Chebyshev distance."""
    if len(rr) > max_len:
        rr = rr[:max_len]
    N = len(rr)
    if N <= m + 1:
        return np.nan
    r = r_frac * np.std(rr)
    if r < 1e-10:
        return np.nan

    def _count_full(data, m_val, r_val):
        n = len(data) - m_val
        if n < 2:
            return 0
        idx = np.arange(m_val)[None, :] + np.arange(n)[:, None]
        templates = data[idx]  # (n, m_val)
        # Chebyshev distance matrix (upper triangle only)
        dist = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        # Exclude self-matches: upper triangle
        tri = np.triu_indices(n, k=1)
        return int(np.sum(dist[tri] <= r_val))

    A = _count_full(rr, m + 1, r)
    B = _count_full(rr, m, r)
    if B == 0:
        return np.nan
    if A == 0:
        return -np.log(1.0 / (B + 1e-6))
    return float(-np.log(A / B))


def load_data():
    print("Loading train dataset...", flush=True)
    ds = CPCTimeBinDataset(**DATASET_KWARGS, train=True)
    data = ds.data.numpy()
    bin_labels = ds.bin_labels.numpy()
    times = ds.times.numpy()
    print(f"  Loaded {len(data)} SR segments, shape={data.shape}", flush=True)
    counts = np.bincount(bin_labels, minlength=len(BIN_NAMES))
    for name, c in zip(BIN_NAMES, counts):
        print(f"  {name}: {c}", flush=True)
    return data, bin_labels, times


def compute_all_features(data):
    import sys
    print("Computing features per segment...", flush=True)
    n = len(data)
    feature_list = []
    for i in range(n):
        seg_flat = data[i].flatten()
        feats = compute_features(seg_flat)
        feature_list.append(feats)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}", flush=True)
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in feature_list])
    nan_mask = np.isnan(X)
    for col in range(X.shape[1]):
        col_nan = nan_mask[:, col]
        if col_nan.any():
            X[col_nan, col] = np.nanmedian(X[:, col])
    print(f"  Feature matrix: {X.shape}, features: {feature_names}", flush=True)
    return X, feature_names


# ======================================================================
# Phase 1: Visualization & Statistical Tests
# ======================================================================


def phase1_plots(X, feature_names, bin_labels, times):
    num_classes = len(BIN_NAMES)

    # Violin plots
    n_feats = len(feature_names)
    cols = 4
    rows = (n_feats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for i, fname in enumerate(feature_names):
        groups = [X[bin_labels == c, i] for c in range(num_classes)]
        parts = axes[i].violinplot(groups, showmeans=True, showmedians=True)
        axes[i].set_xticks(range(1, num_classes + 1))
        axes[i].set_xticklabels(BIN_NAMES, fontsize=7, rotation=15)
        axes[i].set_title(fname, fontsize=9)
        axes[i].grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Time Bin", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOTS_DIR, "violin_plots.png"), dpi=200)
    plt.close(fig)
    print("  Saved violin_plots.png")

    # ANOVA + Kruskal-Wallis
    print("\n  === Statistical Tests (ANOVA + Kruskal-Wallis) ===")
    print(f"  {'Feature':<20} {'ANOVA F':>10} {'ANOVA p':>12} {'KW H':>10} {'KW p':>12} {'Sig?':>6}")
    print("  " + "-" * 72)
    for i, fname in enumerate(feature_names):
        groups = [X[bin_labels == c, i] for c in range(num_classes)]
        f_stat, f_p = f_oneway(*groups)
        h_stat, h_p = kruskal(*groups)
        sig = "*" if min(f_p, h_p) < 0.05 else ""
        sig = "**" if min(f_p, h_p) < 0.001 else sig
        print(f"  {fname:<20} {f_stat:>10.2f} {f_p:>12.2e} {h_stat:>10.2f} {h_p:>12.2e} {sig:>6}")

    # Spearman correlation with raw time
    print("\n  === Spearman Correlation (feature vs time-to-event) ===")
    print(f"  {'Feature':<20} {'rho':>10} {'p-value':>12}")
    print("  " + "-" * 44)
    for i, fname in enumerate(feature_names):
        rho, p = spearmanr(X[:, i], times)
        print(f"  {fname:<20} {rho:>10.4f} {p:>12.2e}")

    # PCA on features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#2196F3", "#4CAF50", "#F44336"]
    for c in range(num_classes):
        mask = bin_labels == c
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c], alpha=0.3, s=6, label=BIN_NAMES[c])
    ax1.set_title(f"PCA of HRV Features (var={pca.explained_variance_ratio_.sum():.2%})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # PCA on raw flattened segments
    data_flat = X_scaled
    pca_raw = PCA(n_components=2, random_state=42)
    raw_flat = scaler.fit_transform(X)
    X_pca_raw = pca_raw.fit_transform(raw_flat)
    for c in range(num_classes):
        mask = bin_labels == c
        ax2.scatter(X_pca_raw[mask, 0], X_pca_raw[mask, 1], c=colors[c], alpha=0.3, s=6, label=BIN_NAMES[c])
    ax2.set_title("PCA of Feature Vectors")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pca_plots.png"), dpi=200)
    plt.close(fig)
    print("  Saved pca_plots.png")

    # t-SNE (subsample for speed)
    max_tsne = 5000
    if len(X_scaled) > max_tsne:
        idx = np.random.RandomState(42).choice(len(X_scaled), max_tsne, replace=False)
        X_sub = X_scaled[idx]
        labels_sub = bin_labels[idx]
    else:
        X_sub = X_scaled
        labels_sub = bin_labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(num_classes):
        mask = labels_sub == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[c], alpha=0.4, s=8, label=BIN_NAMES[c])
    ax.set_title("t-SNE of HRV Features")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "tsne_plot.png"), dpi=200)
    plt.close(fig)
    print("  Saved tsne_plot.png")


# ======================================================================
# Phase 2: Classical ML Baseline
# ======================================================================


def phase2_baseline(X, feature_names, bin_labels, data):
    num_classes = len(BIN_NAMES)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Phase 2: Classical ML Baseline (5-fold CV) ===")

    # RandomForest on HRV features
    pipe_feat = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ])
    scores_feat = cross_validate(
        pipe_feat, X, bin_labels, cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(f"\n  RandomForest on {len(feature_names)} HRV features:")
    print(f"    Train acc:  {scores_feat['train_accuracy'].mean():.4f} +/- {scores_feat['train_accuracy'].std():.4f}")
    print(f"    Val acc:    {scores_feat['test_accuracy'].mean():.4f} +/- {scores_feat['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores_feat['test_f1_macro'].mean():.4f} +/- {scores_feat['test_f1_macro'].std():.4f}")

    # RandomForest on raw flattened segments
    data_flat = data.reshape(len(data), -1)
    pipe_raw = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ])
    scores_raw = cross_validate(
        pipe_raw, data_flat, bin_labels, cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(f"\n  RandomForest on raw flattened segments ({data_flat.shape[1]} features):")
    print(f"    Train acc:  {scores_raw['train_accuracy'].mean():.4f} +/- {scores_raw['train_accuracy'].std():.4f}")
    print(f"    Val acc:    {scores_raw['test_accuracy'].mean():.4f} +/- {scores_raw['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores_raw['test_f1_macro'].mean():.4f} +/- {scores_raw['test_f1_macro'].std():.4f}")

    # Combined: HRV features + raw segments
    X_combined = np.hstack([X, data_flat])
    pipe_comb = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ])
    scores_comb = cross_validate(
        pipe_comb, X_combined, bin_labels, cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(f"\n  RandomForest on combined (HRV + raw) ({X_combined.shape[1]} features):")
    print(f"    Train acc:  {scores_comb['train_accuracy'].mean():.4f} +/- {scores_comb['train_accuracy'].std():.4f}")
    print(f"    Val acc:    {scores_comb['test_accuracy'].mean():.4f} +/- {scores_comb['test_accuracy'].std():.4f}")
    print(f"    Val F1m:    {scores_comb['test_f1_macro'].mean():.4f} +/- {scores_comb['test_f1_macro'].std():.4f}")

    # Feature importance from RF on HRV features
    pipe_feat.fit(X, bin_labels)
    importances = pipe_feat.named_steps["rf"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_names)), importances[sorted_idx])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature Importance")
    ax.set_title("RandomForest Feature Importance (HRV features)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=200)
    plt.close(fig)
    print("  Saved feature_importance.png")

    print(f"\n  === Summary ===")
    print(f"  Random baseline (3-class):      33.3%")
    print(f"  NN time-bin classifier (v9):     38.4%")
    print(f"  RF on HRV features:              {scores_feat['test_accuracy'].mean():.1%}")
    print(f"  RF on raw segments:              {scores_raw['test_accuracy'].mean():.1%}")
    print(f"  RF on combined:                  {scores_comb['test_accuracy'].mean():.1%}")


if __name__ == "__main__":
    cache_path = os.path.join(PLOTS_DIR, "features_cache.npz")
    data, bin_labels, times = load_data()

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        feature_names = list(cached["feature_names"])
        print(f"  Feature matrix: {X.shape}, features: {feature_names}", flush=True)
    else:
        X, feature_names = compute_all_features(data)
        np.savez(cache_path, X=X, feature_names=np.array(feature_names))
        print(f"  Cached features to {cache_path}", flush=True)

    print("\n=== Phase 1: Data Analysis ===", flush=True)
    phase1_plots(X, feature_names, bin_labels, times)

    phase2_baseline(X, feature_names, bin_labels, data)

    print("\nDone. All plots saved to:", PLOTS_DIR, flush=True)
