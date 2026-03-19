"""
Probe B (v17-style): open-set nearest-centroid based on frozen embeddings,
with patient-level seen/unseen holdout, using a v21 checkpoint.

This script ONLY runs the probe. It does NOT re-train the regressor.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dotenv import load_dotenv
import wandb

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset


load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TTEPredictor(nn.Module):
    """Encoder -> GRU -> head(h). Provides extract_h(h) for probing."""

    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def extract_h(self, rr_windows: torch.Tensor) -> torch.Tensor:
        bsz, steps, win = rr_windows.shape
        z = self.encoder(rr_windows.view(bsz * steps, win)).view(bsz, steps, -1)
        _, h = self.gru(z)
        return h.squeeze(0)

    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        h = self.extract_h(rr_windows)
        return self.head(h).squeeze(-1)


def load_excellent_ids(json_path: str):
    with open(json_path, "r") as f:
        q = json.load(f)
    return sorted(int(p["pid"]) for p in q["patients"] if p["tier"] == "excellent")


def pairwise_l2(a, b):
    # a: [N, D], b: [M, D] -> [N, M]
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    return np.sqrt(np.clip(aa + bb - 2.0 * ab, 0.0, None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="a", choices=list("abcdef"))
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to v21*_best.pth. If not set, uses v21{variant}_best.pth in CWD.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--probe_seen_patient_ratio", type=float, default=0.8)
    parser.add_argument("--probe_train_segment_ratio", type=float, default=0.8)

    args = parser.parse_args()

    cfg = {
        "seed": args.seed,
        "afib_length": 3600,
        "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        "batch_size_embed": 256,
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "probe_seen_patient_ratio": args.probe_seen_patient_ratio,
        "probe_train_segment_ratio": args.probe_train_segment_ratio,
    }

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = f"v21{args.variant}_best.pth"
    ckpt_path = os.path.abspath(ckpt_path)

    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        name=f"v21-probeB-from-ckpt-{args.variant}",
        config=cfg,
    )

    root = os.path.dirname(__file__)
    excellent_ids = load_excellent_ids(os.path.join(root, "v15c_patient_quality.json"))
    logger.info(f"[v21 ProbeB] excellent patients: {len(excellent_ids)}")

    ds_args = dict(
        processed_dataset_path="//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets",
        afib_length=cfg["afib_length"],
        sr_length=cfg["sr_length"],
        number_of_windows_in_segment=cfg["number_of_windows_in_segment"],
        stride=cfg["stride"],
        window_size=cfg["window_size"],
        validation_split=cfg["validation_split"],
    )

    train_ds = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    all_x = torch.cat([train_ds.data, val_ds.data], dim=0)
    all_pid = torch.cat([train_ds.patient_ids, val_ds.patient_ids], dim=0)

    mask = np.isin(all_pid.numpy(), excellent_ids)
    all_x = all_x[mask]
    all_pid = all_pid[mask]

    all_pid_np = all_pid.numpy().astype(int)
    logger.info(f"[v21 ProbeB] pooled excellent-only segments: {len(all_x)}")

    # Load checkpoint and extract embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictor(cfg["latent_dim"], cfg["hidden_dim"], cfg["dropout"]).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    emb = []
    pid_arr = []
    loader = DataLoader(
        TensorDataset(all_x, all_pid),
        batch_size=cfg["batch_size_embed"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    with torch.no_grad():
        for rr, pid in tqdm(loader, desc=f"extract_h v21{args.variant}"):
            rr = rr.to(device)
            emb.append(model.extract_h(rr).cpu().numpy())
            pid_arr.append(pid.numpy())
    emb = np.concatenate(emb)
    pid_arr = np.concatenate(pid_arr).astype(int)

    # Patient-level seen/unseen split
    pids = np.array(sorted(np.unique(pid_arr)))
    np.random.shuffle(pids)
    n_seen = int(len(pids) * cfg["probe_seen_patient_ratio"])
    seen_ids = set(pids[:n_seen].tolist())
    unseen_ids = set(pids[n_seen:].tolist())

    seen_mask = np.isin(pid_arr, list(seen_ids))
    unseen_mask = np.isin(pid_arr, list(unseen_ids))
    x_seen = emb[seen_mask]
    y_seen_pid = pid_arr[seen_mask]
    x_unseen = emb[unseen_mask]

    logger.info(
        f"[v21 ProbeB] seen_patients={len(seen_ids)} unseen_patients={len(unseen_ids)} "
        f"seen_segments={len(x_seen)} unseen_segments={len(x_unseen)}"
    )

    # Split seen segments into centroid-train vs seen-val
    idx_seen = np.random.permutation(len(x_seen))
    s_split = int(len(idx_seen) * cfg["probe_train_segment_ratio"])
    idx_seen_tr, idx_seen_va = idx_seen[:s_split], idx_seen[s_split:]

    x_seen_tr = x_seen[idx_seen_tr]
    y_seen_tr = y_seen_pid[idx_seen_tr]
    x_seen_va = x_seen[idx_seen_va]
    y_seen_va = y_seen_pid[idx_seen_va]

    # Normalize using seen-train stats
    mu = x_seen_tr.mean(axis=0, keepdims=True)
    sd = x_seen_tr.std(axis=0, keepdims=True) + 1e-8
    x_seen_tr = (x_seen_tr - mu) / sd
    x_seen_va = (x_seen_va - mu) / sd
    x_unseen = (x_unseen - mu) / sd

    # Build centroids by seen patient
    centroid_ids = sorted(seen_ids)
    centroids = []
    for pid in centroid_ids:
        centroids.append(x_seen_tr[y_seen_tr == pid].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    pid2cls = {pid: i for i, pid in enumerate(centroid_ids)}

    # Seen closed-set accuracy
    d_seen = pairwise_l2(x_seen_va, centroids)
    pred_cls_seen = np.argmin(d_seen, axis=1)
    true_cls_seen = np.array([pid2cls[int(p)] for p in y_seen_va], dtype=np.int64)
    seen_closed_acc = float((pred_cls_seen == true_cls_seen).mean())
    min_dist_seen = d_seen[np.arange(len(d_seen)), pred_cls_seen]

    # Unseen open-set detection AUROC using min distance
    d_unseen = pairwise_l2(x_unseen, centroids)
    min_dist_unseen = np.min(d_unseen, axis=1)

    y_det = np.concatenate(
        [np.zeros_like(min_dist_seen, dtype=np.int64), np.ones_like(min_dist_unseen, dtype=np.int64)]
    )
    scores = np.concatenate([min_dist_seen, min_dist_unseen])
    auroc = float(roc_auc_score(y_det, scores))

    thr95 = float(np.percentile(min_dist_seen, 95.0))
    pred_unseen = (scores >= thr95).astype(np.int64)
    binary_acc = float((pred_unseen == y_det).mean())
    tpr = float((pred_unseen[len(min_dist_seen) :] == 1).mean())
    fpr = float((pred_unseen[: len(min_dist_seen)] == 1).mean())

    results = {
        "experiment": f"v21_probeB_open_set_centroid_from_ckpt_{args.variant}",
        "ckpt_path": ckpt_path,
        "n_excellent_patients": len(excellent_ids),
        "n_seen_patients_for_probe": len(seen_ids),
        "n_unseen_patients_for_probe": len(unseen_ids),
        "seen_closed_set_accuracy": seen_closed_acc,
        "unseen_detection_auroc": auroc,
        "threshold_95pct_seen": thr95,
        "threshold_95_binary_acc": binary_acc,
        "threshold_95_tpr_unseen": tpr,
        "threshold_95_fpr_seen": fpr,
    }

    out_json = os.path.join(
        os.path.dirname(__file__),
        "results",
        f"v21_probeB_open_set_{args.variant}.json",
    )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    run.log(
        {
            "probe/seen_closed_set_acc": seen_closed_acc,
            "probe/unseen_det_auroc": auroc,
            "probe/unseen_det_thr95_binary_acc": binary_acc,
            "probe/unseen_det_thr95_tpr_unseen": tpr,
            "probe/unseen_det_thr95_fpr_seen": fpr,
        }
    )

    logger.info(
        f"[v21 ProbeB] done {args.variant} | seen_acc={seen_closed_acc:.4f} "
        f"unseen_auroc={auroc:.4f}"
    )

    run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

