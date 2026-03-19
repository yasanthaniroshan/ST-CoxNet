"""
CPC Temporal Signal Verification
=================================
Compares learned CPC representations against raw HRV statistics to determine
whether the encoder + GRU context captures richer temporal signal for
predicting time-to-AFib onset.

Metrics:
  1. Within-patient Spearman ρ (each embedding dim vs time-to-event)
  2. Linear probe R² (CPC embeddings → time-to-event)
  3. Patient-level Concordance index
  4. PCA visualisation coloured by time-to-event

Outputs:
  plots/cpc_temporal_signal/  (plots + report.txt)
"""

import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Checkpoint-compatible model (matches saved cpc_model.pth architecture) ───
class _CheckpointEncoder(nn.Module):
    """Encoder architecture that produced the existing .pth checkpoints."""
    def __init__(self, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(8, 16, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 13, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, rr_window: torch.Tensor) -> torch.Tensor:
        x = rr_window.unsqueeze(1)
        h = self.feature_extractor(x)
        return self.projection(h)


class _CheckpointARBlock(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=context_dim,
                          batch_first=True)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        c_seq, _ = self.gru(z_seq)
        return c_seq


class _CheckpointCPC(nn.Module):
    """CPC wrapper matching the checkpoint's state_dict keys."""
    def __init__(self, latent_dim: int, context_dim: int,
                 number_of_prediction_steps: int,
                 temperature: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.encoder = _CheckpointEncoder(latent_dim, dropout=dropout)
        self.ar_block = _CheckpointARBlock(latent_dim, context_dim)
        self.number_of_prediction_steps = number_of_prediction_steps
        self.temperature = temperature
        self.Wk = nn.ModuleList(
            [nn.Linear(context_dim, latent_dim)
             for _ in range(number_of_prediction_steps)]
        )

    def forward(self, rr_windows: torch.Tensor):
        B, T, W = rr_windows.shape
        z_seq = self.encoder(rr_windows.view(-1, W)).view(B, T, -1)
        z_seq = F.normalize(z_seq, dim=-1)
        c_seq = self.ar_block(z_seq)
        return None, None, z_seq, c_seq


# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
AFIB_LENGTH = 60 * 60
SR_LENGTH = int(1.5 * 60 * 60)
WINDOW_SIZE = 100
NUMBER_OF_WINDOWS = 10
STRIDE = 20
SEGMENT_SIZE = NUMBER_OF_WINDOWS * WINDOW_SIZE

CPC_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpc_model.pth")
CPC_CONFIG = dict(
    latent_dim=64, context_dim=128,
    number_of_prediction_steps=6, dropout=0.2, temperature=0.2,
)
MIN_SEGS_FOR_CORR = 10
SUBSAMPLE_STEP_DIVISOR = 2

REPORT_DIR = os.path.join(os.getcwd(), "plots", "cpc_temporal_signal")
os.makedirs(REPORT_DIR, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

report_lines: list[str] = []


def log(msg: str = ""):
    print(msg)
    report_lines.append(msg)


def save_fig(fig, name: str):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  [saved] {path}")


# ── Data Loading (mirrors raw_dataset_analysis.py / CreateDataset.py) ────────
def load_record_details(patient_list, dataset_path, afib_length, sr_length):
    records, skipped = [], defaultdict(list)
    for patient in patient_list:
        record_dir = os.path.join(dataset_path, patient)
        ecg_csv = os.path.join(record_dir, f"{patient}_ecg_labels.csv")
        if not os.path.exists(ecg_csv):
            skipped["missing_csv"].append(patient)
            continue
        df = pd.read_csv(ecg_csv)
        for idx, row in df.iterrows():
            if row["start_file_index"] != row["end_file_index"]:
                continue
            if row["af_duration"] < afib_length:
                continue
            if row["nsr_before_duration"] < sr_length + 6000:
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
    record_dir = os.path.join(dataset_path, patient)
    rr_df = pd.read_csv(os.path.join(record_dir, f"{patient}_rr_labels.csv"))
    row = rr_df.loc[patient_data["record_index"]]
    rr_start, rr_end = int(row["start_rr_index"]), int(row["end_rr_index"])
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
    return combined, len(sr_seg)


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


def compute_hrv_features(rr_ms):
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


# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 80)
    log("  CPC TEMPORAL SIGNAL VERIFICATION")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n  Device       : {device}")
    log(f"  Checkpoint   : {CPC_CHECKPOINT}")

    # ── 1  Load dataset ──────────────────────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 1: DATA LOADING & SEGMENTATION")
    log("─" * 80)

    patient_list = sorted(os.listdir(DATASET_PATH))
    records, _ = load_record_details(patient_list, DATASET_PATH, AFIB_LENGTH, SR_LENGTH)

    per_record_rr = []
    per_record_patient = []
    for rec in tqdm(records, desc="Loading RR data"):
        try:
            combined, afib_start = load_rr_data(rec, DATASET_PATH, AFIB_LENGTH, SR_LENGTH)
            per_record_rr.append((combined, afib_start))
            per_record_patient.append(rec["patient"])
        except Exception as e:
            log(f"  [WARN] Failed {rec['patient']} idx={rec['record_index']}: {e}")

    all_segments, all_labels, all_times, all_patient_ids = [], [], [], []
    for rec_idx, (combined, afib_start) in enumerate(tqdm(per_record_rr, desc="Segmenting")):
        segs, labs, tms = segment_rr_data(combined, afib_start, SEGMENT_SIZE, STRIDE, WINDOW_SIZE)
        if len(segs) == 0:
            continue
        all_segments.append(segs)
        all_labels.append(labs)
        all_times.append(tms)
        all_patient_ids.append(np.full(len(segs), per_record_patient[rec_idx], dtype=object))

    all_segments = np.concatenate(all_segments, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_times = np.concatenate(all_times, axis=0)
    all_patient_ids = np.concatenate(all_patient_ids, axis=0)

    sr_mask = all_labels == -1
    sr_segments = all_segments[sr_mask]
    sr_times_s = all_times[sr_mask] / 1000.0
    sr_pids = all_patient_ids[sr_mask]

    log(f"\n  Total segments: {len(all_segments):,}  |  SR segments: {sr_mask.sum():,}")
    log(f"  Unique patients (SR): {len(np.unique(sr_pids))}")

    # ── 2  Load CPC & extract embeddings ─────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 2: CPC MODEL — LOAD & EXTRACT EMBEDDINGS")
    log("─" * 80)

    cpc = _CheckpointCPC(**CPC_CONFIG).to(device)
    state_dict = torch.load(CPC_CHECKPOINT, map_location=device, weights_only=True)
    cpc.load_state_dict(state_dict)
    cpc.eval()
    log(f"  Loaded CPC checkpoint ({sum(p.numel() for p in cpc.parameters()):,} parameters)")

    scaler = RobustScaler()
    sr_flat = sr_segments.reshape(-1, WINDOW_SIZE)
    sr_scaled = scaler.fit_transform(sr_flat).reshape(sr_segments.shape)

    z_all, c_all = [], []
    batch_size = 256
    n_sr = len(sr_scaled)
    with torch.no_grad():
        for start in tqdm(range(0, n_sr, batch_size), desc="Extracting CPC embeddings"):
            batch = torch.tensor(sr_scaled[start:start + batch_size], dtype=torch.float32).to(device)
            _, _, z_seq, c_seq = cpc(batch)
            z_all.append(z_seq[:, -1, :].cpu().numpy())
            c_all.append(c_seq[:, -1, :].cpu().numpy())

    z_last = np.concatenate(z_all, axis=0)  # [N_sr, 64]
    c_last = np.concatenate(c_all, axis=0)  # [N_sr, 128]
    zc_combined = np.concatenate([z_last, c_last], axis=1)  # [N_sr, 192]

    log(f"  z_last shape: {z_last.shape}  (encoder embeddings, last timestep)")
    log(f"  c_last shape: {c_last.shape}  (GRU context, last timestep)")
    log(f"  combined shape: {zc_combined.shape}")

    # ── 3  HRV baseline features for comparison ──────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 3: HRV BASELINE FEATURES")
    log("─" * 80)

    hrv_matrix = []
    for seg in tqdm(sr_segments, desc="Computing HRV features"):
        f = compute_hrv_features(seg.flatten())
        hrv_matrix.append(list(f.values()))
    hrv_matrix = np.array(hrv_matrix)
    hrv_names = list(compute_hrv_features(sr_segments[0].flatten()).keys())

    valid_mask = ~(np.isnan(hrv_matrix).any(axis=1) | np.isinf(hrv_matrix).any(axis=1))
    log(f"  HRV feature matrix: {hrv_matrix.shape}  ({valid_mask.sum()} valid rows)")
    log(f"  Features: {', '.join(hrv_names)}")

    # ── 4  Within-patient Spearman correlations ──────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 4: WITHIN-PATIENT SPEARMAN ρ — CPC vs HRV")
    log("─" * 80)

    subsample_step = max(1, SEGMENT_SIZE // STRIDE // SUBSAMPLE_STEP_DIVISOR)

    def within_patient_spearman(feature_matrix, feature_labels, times, pids):
        rho_per_feat = {name: [] for name in feature_labels}
        pval_per_feat = {name: [] for name in feature_labels}
        n_patients = 0
        for pid in np.unique(pids):
            pmask = pids == pid
            p_times = times[pmask]
            p_feats = feature_matrix[pmask]
            if len(p_feats) < MIN_SEGS_FOR_CORR:
                continue
            sub_idx = np.arange(0, len(p_feats), subsample_step)
            p_feats_sub = p_feats[sub_idx]
            p_times_sub = p_times[sub_idx]
            if len(p_feats_sub) < MIN_SEGS_FOR_CORR:
                continue
            n_patients += 1
            for fi, name in enumerate(feature_labels):
                vals = p_feats_sub[:, fi]
                valid = ~(np.isnan(vals) | np.isnan(p_times_sub))
                if valid.sum() < MIN_SEGS_FOR_CORR:
                    continue
                rho, pval = sp_stats.spearmanr(vals[valid], p_times_sub[valid])
                rho_per_feat[name].append(rho)
                pval_per_feat[name].append(pval)
        return rho_per_feat, pval_per_feat, n_patients

    z_labels = [f"z_{i}" for i in range(z_last.shape[1])]
    c_labels = [f"c_{i}" for i in range(c_last.shape[1])]
    zc_labels = z_labels + c_labels

    log("  Computing within-patient Spearman for CPC embeddings...")
    cpc_rhos, cpc_pvals, n_pts_cpc = within_patient_spearman(
        zc_combined, zc_labels, sr_times_s, sr_pids
    )

    log("  Computing within-patient Spearman for HRV features...")
    hrv_rhos, hrv_pvals, n_pts_hrv = within_patient_spearman(
        hrv_matrix, hrv_names, sr_times_s, sr_pids
    )

    def summarize_rhos(rho_dict, pval_dict, label):
        abs_median_rhos = {}
        for name in rho_dict:
            rhos = np.array(rho_dict[name])
            if len(rhos) == 0:
                abs_median_rhos[name] = 0
                continue
            abs_median_rhos[name] = np.nanmedian(np.abs(rhos))
        return abs_median_rhos

    hrv_abs_medians = summarize_rhos(hrv_rhos, hrv_pvals, "HRV")
    cpc_abs_medians = summarize_rhos(cpc_rhos, cpc_pvals, "CPC")

    best_hrv = max(hrv_abs_medians.values()) if hrv_abs_medians else 0
    best_hrv_name = max(hrv_abs_medians, key=hrv_abs_medians.get) if hrv_abs_medians else "N/A"

    z_medians = {k: v for k, v in cpc_abs_medians.items() if k.startswith("z_")}
    c_medians = {k: v for k, v in cpc_abs_medians.items() if k.startswith("c_")}

    best_z = max(z_medians.values()) if z_medians else 0
    best_z_name = max(z_medians, key=z_medians.get) if z_medians else "N/A"
    best_c = max(c_medians.values()) if c_medians else 0
    best_c_name = max(c_medians, key=c_medians.get) if c_medians else "N/A"

    mean_z_rho = np.mean(list(z_medians.values())) if z_medians else 0
    mean_c_rho = np.mean(list(c_medians.values())) if c_medians else 0
    mean_hrv_rho = np.mean(list(hrv_abs_medians.values())) if hrv_abs_medians else 0

    top10_z = sorted(z_medians.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_c = sorted(c_medians.items(), key=lambda x: x[1], reverse=True)[:10]

    log(f"\n  Patients analysed: {n_pts_cpc} (CPC), {n_pts_hrv} (HRV)")
    log(f"  Sub-sampling step: every {subsample_step} segments")

    log(f"\n  ── HRV Features (baseline) ──")
    log(f"  {'Feature':<12} {'|ρ| median':>12} {'% sig':>8}")
    log(f"  {'─'*12} {'─'*12} {'─'*8}")
    for feat in hrv_names:
        rhos = np.array(hrv_rhos[feat])
        pvals = np.array(hrv_pvals[feat])
        if len(rhos) == 0:
            continue
        med_abs = np.nanmedian(np.abs(rhos))
        pct = 100 * np.mean(pvals < 0.05)
        log(f"  {feat:<12} {med_abs:>12.4f} {pct:>7.1f}%")
    log(f"  BEST HRV: {best_hrv_name} |ρ|={best_hrv:.4f}")
    log(f"  MEAN HRV |ρ|: {mean_hrv_rho:.4f}")

    log(f"\n  ── CPC Encoder (z) — Top 10 dims ──")
    log(f"  {'Dim':<8} {'|ρ| median':>12} {'% sig':>8}")
    log(f"  {'─'*8} {'─'*12} {'─'*8}")
    for name, med in top10_z:
        rhos = np.array(cpc_rhos[name])
        pvals = np.array(cpc_pvals[name])
        pct = 100 * np.mean(pvals < 0.05) if len(pvals) else 0
        log(f"  {name:<8} {med:>12.4f} {pct:>7.1f}%")
    log(f"  BEST z: {best_z_name} |ρ|={best_z:.4f}  |  MEAN z |ρ|: {mean_z_rho:.4f}")

    log(f"\n  ── CPC Context (c) — Top 10 dims ──")
    log(f"  {'Dim':<8} {'|ρ| median':>12} {'% sig':>8}")
    log(f"  {'─'*8} {'─'*12} {'─'*8}")
    for name, med in top10_c:
        rhos = np.array(cpc_rhos[name])
        pvals = np.array(cpc_pvals[name])
        pct = 100 * np.mean(pvals < 0.05) if len(pvals) else 0
        log(f"  {name:<8} {med:>12.4f} {pct:>7.1f}%")
    log(f"  BEST c: {best_c_name} |ρ|={best_c:.4f}  |  MEAN c |ρ|: {mean_c_rho:.4f}")

    # ── 5  Linear probe R² comparison ────────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 5: LINEAR PROBE R² — CPC vs HRV → TIME-TO-EVENT")
    log("─" * 80)

    valid = valid_mask & (sr_times_s > 0)
    X_hrv = RobustScaler().fit_transform(hrv_matrix[valid])
    X_z = RobustScaler().fit_transform(z_last[valid])
    X_c = RobustScaler().fit_transform(c_last[valid])
    X_zc = RobustScaler().fit_transform(zc_combined[valid])
    y = sr_times_s[valid]
    groups = sr_pids[valid]

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    def evaluate_linear_probe(X, y, groups, label, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        r2_scores = []
        spearman_scores = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            ss_res = np.sum((y[test_idx] - y_pred) ** 2)
            ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2_scores.append(r2)
            rho, _ = sp_stats.spearmanr(y[test_idx], y_pred)
            spearman_scores.append(rho)
        return np.array(r2_scores), np.array(spearman_scores)

    log(f"\n  Patient-level GroupKFold (k={n_splits}) — Ridge regression")
    log(f"  Target: time-to-event (seconds), {valid.sum():,} valid SR segments")

    results = {}
    for name, X in [("HRV (9 feats)", X_hrv), ("z_last (64d)", X_z),
                     ("c_last (128d)", X_c), ("z+c (192d)", X_zc)]:
        r2s, rhos = evaluate_linear_probe(X, y, groups, name, n_splits)
        results[name] = (r2s, rhos)

    log(f"\n  {'Representation':<20} {'R² mean±std':>16} {'Spearman ρ mean±std':>22}")
    log(f"  {'─'*20} {'─'*16} {'─'*22}")
    for name in results:
        r2s, rhos = results[name]
        log(f"  {name:<20} {np.mean(r2s):>7.4f}±{np.std(r2s):.4f} "
            f"{np.mean(rhos):>11.4f}±{np.std(rhos):.4f}")

    # ── 6  Patient-level C-index ─────────────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 6: PATIENT-LEVEL CONCORDANCE INDEX")
    log("─" * 80)

    def patient_concordance(X, y, pids):
        gkf = GroupKFold(n_splits=min(5, len(np.unique(pids))))
        c_indices = []
        for train_idx, test_idx in gkf.split(X, y, pids):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            concordant, discordant = 0, 0
            y_test = y[test_idx]
            n = len(y_test)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(y_test[i] - y_test[j]) < 1e-6:
                        continue
                    if (y_test[i] > y_test[j] and y_pred[i] > y_pred[j]) or \
                       (y_test[i] < y_test[j] and y_pred[i] < y_pred[j]):
                        concordant += 1
                    else:
                        discordant += 1
            total = concordant + discordant
            c_indices.append(concordant / total if total > 0 else 0.5)
        return np.array(c_indices)

    subsample_n = min(5000, valid.sum())
    sub_idx = np.random.choice(valid.sum(), subsample_n, replace=False)
    X_hrv_sub = X_hrv[sub_idx]
    X_zc_sub = X_zc[sub_idx]
    y_sub = y[sub_idx]
    g_sub = groups[sub_idx]

    log(f"  C-index computed on {subsample_n} subsampled SR segments (patient-level splits)")

    c_hrv = patient_concordance(X_hrv_sub, y_sub, g_sub)
    c_zc = patient_concordance(X_zc_sub, y_sub, g_sub)

    log(f"\n  {'Representation':<20} {'C-index mean±std':>20}")
    log(f"  {'─'*20} {'─'*20}")
    log(f"  {'HRV (9 feats)':<20} {np.mean(c_hrv):>9.4f}±{np.std(c_hrv):.4f}")
    log(f"  {'z+c (192d)':<20} {np.mean(c_zc):>9.4f}±{np.std(c_zc):.4f}")

    # ── 7  Plots ─────────────────────────────────────────────────────────────
    log("\n" + "─" * 80)
    log("SECTION 7: VISUALISATIONS")
    log("─" * 80)

    # -- Plot 1: Spearman comparison bar chart ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Within-Patient Spearman |ρ| — CPC Representations vs HRV",
                 fontsize=15, fontweight="bold")

    ax = axes[0]
    hrv_vals = [hrv_abs_medians[f] for f in hrv_names]
    bars = ax.barh(range(len(hrv_names)), hrv_vals, color="#3498db", alpha=0.85)
    ax.set_yticks(range(len(hrv_names)))
    ax.set_yticklabels(hrv_names, fontsize=10)
    ax.set_xlabel("Median |Spearman ρ|")
    ax.set_title(f"HRV Features\n(mean |ρ|={mean_hrv_rho:.4f})")
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(hrv_vals), best_z, best_c) * 1.2 + 0.01)

    ax = axes[1]
    top_z_names = [n for n, _ in top10_z]
    top_z_vals = [v for _, v in top10_z]
    ax.barh(range(len(top_z_names)), top_z_vals, color="#e74c3c", alpha=0.85)
    ax.set_yticks(range(len(top_z_names)))
    ax.set_yticklabels(top_z_names, fontsize=10)
    ax.set_xlabel("Median |Spearman ρ|")
    ax.set_title(f"CPC Encoder (z) — Top 10\n(mean |ρ|={mean_z_rho:.4f})")
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(hrv_vals), best_z, best_c) * 1.2 + 0.01)

    ax = axes[2]
    top_c_names = [n for n, _ in top10_c]
    top_c_vals = [v for _, v in top10_c]
    ax.barh(range(len(top_c_names)), top_c_vals, color="#2ecc71", alpha=0.85)
    ax.set_yticks(range(len(top_c_names)))
    ax.set_yticklabels(top_c_names, fontsize=10)
    ax.set_xlabel("Median |Spearman ρ|")
    ax.set_title(f"CPC Context (c) — Top 10\n(mean |ρ|={mean_c_rho:.4f})")
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(hrv_vals), best_z, best_c) * 1.2 + 0.01)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "01_spearman_comparison.png")

    # -- Plot 2: R² and C-index comparison bars --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Linear Probe Performance — CPC vs HRV", fontsize=15, fontweight="bold")

    ax = axes[0]
    names_r2 = list(results.keys())
    means_r2 = [np.mean(results[n][0]) for n in names_r2]
    stds_r2 = [np.std(results[n][0]) for n in names_r2]
    colors_r2 = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    bars = ax.bar(range(len(names_r2)), means_r2, yerr=stds_r2, capsize=5,
                  color=colors_r2[:len(names_r2)], alpha=0.85)
    ax.set_xticks(range(len(names_r2)))
    ax.set_xticklabels(names_r2, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("R² (GroupKFold)")
    ax.set_title("Ridge R² — Predicting Time-to-Event")
    for bar, m in zip(bars, means_r2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.4f}", ha="center", fontsize=9)

    ax = axes[1]
    c_names = ["HRV (9 feats)", "z+c (192d)"]
    c_means = [np.mean(c_hrv), np.mean(c_zc)]
    c_stds = [np.std(c_hrv), np.std(c_zc)]
    bars = ax.bar(range(2), c_means, yerr=c_stds, capsize=5,
                  color=["#3498db", "#9b59b6"], alpha=0.85)
    ax.set_xticks(range(2))
    ax.set_xticklabels(c_names, fontsize=11)
    ax.set_ylabel("Concordance Index")
    ax.set_title("Patient-Level C-Index")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.6, label="Random baseline")
    ax.legend()
    for bar, m in zip(bars, c_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.4f}", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "02_linear_probe_comparison.png")

    # -- Plot 3: PCA of CPC embeddings coloured by time-to-event ---------------
    n_pca = min(5000, len(z_last))
    pca_idx = np.random.choice(len(z_last), n_pca, replace=False)

    pca_z = PCA(n_components=2, random_state=42)
    pca_c = PCA(n_components=2, random_state=42)
    z_pca = pca_z.fit_transform(z_last[pca_idx])
    c_pca = pca_c.fit_transform(c_last[pca_idx])
    tte_pca = sr_times_s[pca_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("PCA of CPC Embeddings — Coloured by Time-to-AFib Onset (seconds)",
                 fontsize=14, fontweight="bold")

    for ax, coords, title, pca_obj in [
        (axes[0], z_pca, "Encoder Embeddings (z)", pca_z),
        (axes[1], c_pca, "Context Vectors (c)", pca_c)
    ]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=tte_pca, cmap="viridis",
                        alpha=0.5, s=8, vmin=np.percentile(tte_pca, 2),
                        vmax=np.percentile(tte_pca, 98))
        ax.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label="Time to AFib (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "03_pca_by_tte.png")

    # -- Plot 4: Distribution of per-patient Spearman ρ -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Distribution of Within-Patient Spearman ρ — Best Dims",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    rhos_best_hrv = np.array(hrv_rhos.get(best_hrv_name, []))
    if len(rhos_best_hrv) > 0:
        ax.hist(rhos_best_hrv, bins=25, color="#3498db", alpha=0.8, edgecolor="none", density=True)
        ax.axvline(np.nanmedian(rhos_best_hrv), color="red", ls="--", lw=2,
                   label=f"median={np.nanmedian(rhos_best_hrv):.3f}")
        ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Density")
    ax.set_title(f"Best HRV: {best_hrv_name}\n(|ρ| median={best_hrv:.4f})")
    ax.legend(fontsize=9)

    ax = axes[1]
    rhos_best_z = np.array(cpc_rhos.get(best_z_name, []))
    if len(rhos_best_z) > 0:
        ax.hist(rhos_best_z, bins=25, color="#e74c3c", alpha=0.8, edgecolor="none", density=True)
        ax.axvline(np.nanmedian(rhos_best_z), color="red", ls="--", lw=2,
                   label=f"median={np.nanmedian(rhos_best_z):.3f}")
        ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Spearman ρ")
    ax.set_title(f"Best Encoder dim: {best_z_name}\n(|ρ| median={best_z:.4f})")
    ax.legend(fontsize=9)

    ax = axes[2]
    rhos_best_c = np.array(cpc_rhos.get(best_c_name, []))
    if len(rhos_best_c) > 0:
        ax.hist(rhos_best_c, bins=25, color="#2ecc71", alpha=0.8, edgecolor="none", density=True)
        ax.axvline(np.nanmedian(rhos_best_c), color="red", ls="--", lw=2,
                   label=f"median={np.nanmedian(rhos_best_c):.3f}")
        ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Spearman ρ")
    ax.set_title(f"Best Context dim: {best_c_name}\n(|ρ| median={best_c:.4f})")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "04_rho_distributions.png")

    # -- Plot 5: Head-to-head scatter ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("All-Dimension Comparison — Median |ρ| per Feature/Dim",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    z_all_meds = sorted(z_medians.values(), reverse=True)
    ax.plot(range(len(z_all_meds)), z_all_meds, "o-", color="#e74c3c",
            markersize=3, label=f"Encoder z (64 dims, mean={mean_z_rho:.4f})")
    ax.axhline(best_hrv, color="#3498db", ls="--", lw=2,
               label=f"Best HRV ({best_hrv_name}={best_hrv:.4f})")
    ax.axhline(mean_hrv_rho, color="#3498db", ls=":", lw=1.5,
               label=f"Mean HRV ({mean_hrv_rho:.4f})")
    ax.set_xlabel("Encoder dimension (sorted)")
    ax.set_ylabel("Median |Spearman ρ|")
    ax.set_title("Encoder Dims vs HRV Baseline")
    ax.legend(fontsize=9)

    ax = axes[1]
    c_all_meds = sorted(c_medians.values(), reverse=True)
    ax.plot(range(len(c_all_meds)), c_all_meds, "o-", color="#2ecc71",
            markersize=3, label=f"Context c (128 dims, mean={mean_c_rho:.4f})")
    ax.axhline(best_hrv, color="#3498db", ls="--", lw=2,
               label=f"Best HRV ({best_hrv_name}={best_hrv:.4f})")
    ax.axhline(mean_hrv_rho, color="#3498db", ls=":", lw=1.5,
               label=f"Mean HRV ({mean_hrv_rho:.4f})")
    ax.set_xlabel("Context dimension (sorted)")
    ax.set_ylabel("Median |Spearman ρ|")
    ax.set_title("Context Dims vs HRV Baseline")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "05_dimension_comparison.png")

    # ── 8  Summary ───────────────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("SUMMARY — CPC vs HRV TEMPORAL SIGNAL")
    log("═" * 80)

    r2_hrv = np.mean(results["HRV (9 feats)"][0])
    r2_zc = np.mean(results["z+c (192d)"][0])
    c_hrv_mean = np.mean(c_hrv)
    c_zc_mean = np.mean(c_zc)

    z_dims_beating_best_hrv = sum(1 for v in z_medians.values() if v > best_hrv)
    c_dims_beating_best_hrv = sum(1 for v in c_medians.values() if v > best_hrv)

    log(f"""
  WITHIN-PATIENT SPEARMAN ρ (temporal monotonicity):
    HRV best   : {best_hrv_name} |ρ|={best_hrv:.4f}     mean |ρ|={mean_hrv_rho:.4f}
    CPC z best : {best_z_name} |ρ|={best_z:.4f}        mean |ρ|={mean_z_rho:.4f}
    CPC c best : {best_c_name} |ρ|={best_c:.4f}        mean |ρ|={mean_c_rho:.4f}
    z dims > best HRV : {z_dims_beating_best_hrv} / {len(z_medians)}
    c dims > best HRV : {c_dims_beating_best_hrv} / {len(c_medians)}

  LINEAR PROBE R² (Ridge, patient-level GroupKFold):
    HRV    : {r2_hrv:.4f}
    z+c    : {r2_zc:.4f}
    Δ R²   : {r2_zc - r2_hrv:+.4f}  {'(CPC better)' if r2_zc > r2_hrv else '(HRV better)'}

  CONCORDANCE INDEX (patient-level splits):
    HRV    : {c_hrv_mean:.4f}
    z+c    : {c_zc_mean:.4f}
    Δ C    : {c_zc_mean - c_hrv_mean:+.4f}  {'(CPC better)' if c_zc_mean > c_hrv_mean else '(HRV better)'}

  VERDICT:""")

    if r2_zc > r2_hrv + 0.01 and c_zc_mean > c_hrv_mean + 0.01:
        log("    CPC representations CLEARLY capture more temporal signal than HRV stats.")
        log("    The learned encoder + GRU context extracts richer temporal patterns.")
    elif r2_zc > r2_hrv or c_zc_mean > c_hrv_mean:
        log("    CPC representations show MODEST improvement over raw HRV stats.")
        log("    The signal advantage is present but limited — consider more CPC training")
        log("    or architectural improvements.")
    else:
        log("    CPC representations do NOT outperform raw HRV stats on this dataset.")
        log("    Possible causes: insufficient CPC pre-training, suboptimal architecture,")
        log("    or the temporal signal is simple enough for hand-crafted HRV features.")

    log(f"""
  RECOMMENDATIONS:
    1. {'CPC embeddings are ready for downstream survival modelling' if r2_zc > r2_hrv else 'Consider more CPC pre-training epochs or architecture changes'}
    2. {'Context vectors (c) capture sequential dynamics well — use c_last as primary features' if mean_c_rho > mean_z_rho else 'Encoder embeddings (z) are stronger — consider skipping GRU or improving AR block'}
    3. {'Combine CPC + HRV features for best performance' if abs(r2_zc - r2_hrv) < 0.05 else 'Use the stronger representation exclusively to avoid noise'}
    4. Always use patient-level train/val/test splits
""")

    log("=" * 80)
    log(f"  All plots saved to: {REPORT_DIR}")
    log("=" * 80)

    report_path = os.path.join(REPORT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[DONE] Report saved to {report_path}")


if __name__ == "__main__":
    main()
