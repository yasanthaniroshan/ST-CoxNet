"""
Probe A (v17-style): linear patient-ID classification on frozen embeddings,
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
    """
    Shared regressor backbone from v21:
    Encoder -> GRU -> head(h) where h is GRU hidden state.
    """

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
        # rr_windows: [B, steps, window_size]
        bsz, steps, win = rr_windows.shape
        z = self.encoder(rr_windows.view(bsz * steps, win)).view(bsz, steps, -1)
        _, h = self.gru(z)
        return h.squeeze(0)  # [B, hidden_dim]

    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        h = self.extract_h(rr_windows)
        return self.head(h).squeeze(-1)


def load_excellent_ids(json_path: str):
    with open(json_path, "r") as f:
        q = json.load(f)
    return sorted(int(p["pid"]) for p in q["patients"] if p["tier"] == "excellent")


def best_threshold(scores, labels):
    thresholds = np.unique(scores)
    best_acc = -1.0
    best_t = float(thresholds[len(thresholds) // 2])
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return best_t, best_acc


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
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=256)

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
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "probe_batch_size": args.probe_batch_size,
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
        name=f"v21-probeA-from-ckpt-{args.variant}",
        config=cfg,
    )

    # ── Load excellent-only segments ──
    root = os.path.dirname(__file__)
    excellent_ids = load_excellent_ids(os.path.join(root, "v15c_patient_quality.json"))
    logger.info(f"[v21 ProbeA] excellent patients: {len(excellent_ids)}")

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
    logger.info(f"[v21 ProbeA] pooled excellent-only segments: {len(all_x)}")

    # ── Load checkpoint and extract embeddings ──
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

    # ── Patient-level seen/unseen split for the probe ──
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
        f"[v21 ProbeA] seen_patients={len(seen_ids)} unseen_patients={len(unseen_ids)} "
        f"seen_segments={len(x_seen)} unseen_segments={len(x_unseen)}"
    )

    # segment-level split within seen patients
    idx_seen = np.random.permutation(len(x_seen))
    s_split = int(len(idx_seen) * cfg["probe_train_segment_ratio"])
    idx_seen_tr, idx_seen_va = idx_seen[:s_split], idx_seen[s_split:]
    seen_train_x = x_seen[idx_seen_tr]
    seen_train_pid = y_seen_pid[idx_seen_tr]
    seen_val_x = x_seen[idx_seen_va]
    seen_val_pid = y_seen_pid[idx_seen_va]

    # map seen patient IDs -> class IDs
    pid2cls = {pid: i for i, pid in enumerate(sorted(seen_ids))}
    seen_train_y = np.array([pid2cls[int(p)] for p in seen_train_pid], dtype=np.int64)
    seen_val_y = np.array([pid2cls[int(p)] for p in seen_val_pid], dtype=np.int64)

    # normalize probe features using seen-train stats
    mu = seen_train_x.mean(axis=0, keepdims=True)
    sd = seen_train_x.std(axis=0, keepdims=True) + 1e-8
    seen_train_x = (seen_train_x - mu) / sd
    seen_val_x = (seen_val_x - mu) / sd
    unseen_x = (x_unseen - mu) / sd

    probe = nn.Linear(cfg["hidden_dim"], len(seen_ids)).to(device)
    p_opt = torch.optim.AdamW(probe.parameters(), lr=cfg["probe_lr"], weight_decay=1e-4)
    p_loss_fn = nn.CrossEntropyLoss()

    train_loader_probe = DataLoader(
        TensorDataset(
            torch.tensor(seen_train_x, dtype=torch.float32),
            torch.tensor(seen_train_y, dtype=torch.long),
        ),
        batch_size=cfg["probe_batch_size"],
        shuffle=True,
    )

    best_seen_cls_acc = 0.0
    best_conf_seen = None
    best_conf_unseen = None

    for ep in range(cfg["probe_epochs"]):
        probe.train()
        for xb, yb in train_loader_probe:
            xb = xb.to(device)
            yb = yb.to(device)
            p_opt.zero_grad()
            logits = probe(xb)
            loss = p_loss_fn(logits, yb)
            loss.backward()
            p_opt.step()

        probe.eval()
        with torch.no_grad():
            logits_seen = probe(torch.tensor(seen_val_x, dtype=torch.float32, device=device))
            probs_seen = torch.softmax(logits_seen, dim=-1)
            conf_seen = probs_seen.max(dim=-1).values.detach().cpu().numpy()
            pred_seen = probs_seen.argmax(dim=-1).detach().cpu().numpy()

            seen_cls_acc = float((pred_seen == seen_val_y).mean())

            logits_unseen = probe(torch.tensor(unseen_x, dtype=torch.float32, device=device))
            probs_unseen = torch.softmax(logits_unseen, dim=-1)
            conf_unseen = probs_unseen.max(dim=-1).values.detach().cpu().numpy()

        # unseen detection: higher => more likely unseen
        y_det = np.concatenate(
            [np.zeros_like(conf_seen, dtype=np.int64), np.ones_like(conf_unseen, dtype=np.int64)]
        )
        scores = np.concatenate([-conf_seen, -conf_unseen])
        auroc = float(roc_auc_score(y_det, scores))

        run.log(
            {
                "probe/epoch": ep + 1,
                "probe/seen_cls_acc": seen_cls_acc,
                "probe/unseen_det_auroc": auroc,
                "probe/random_baseline_seen_cls": 1.0 / max(len(seen_ids), 1),
            }
        )

        if seen_cls_acc > best_seen_cls_acc:
            best_seen_cls_acc = seen_cls_acc
            best_conf_seen = conf_seen.copy()
            best_conf_unseen = conf_unseen.copy()

        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(
                f"[v21 ProbeA] epoch {ep+1}/{cfg['probe_epochs']} "
                f"seen_acc={seen_cls_acc:.4f} unseen_auroc={auroc:.4f}"
            )

    # final at best checkpoint
    y_det = np.concatenate(
        [
            np.zeros_like(best_conf_seen, dtype=np.int64),
            np.ones_like(best_conf_unseen, dtype=np.int64),
        ]
    )
    scores = np.concatenate([-best_conf_seen, -best_conf_unseen])
    final_auroc = float(roc_auc_score(y_det, scores))
    thr, bin_acc = best_threshold(scores, y_det)

    results = {
        "experiment": f"v21_probeA_seen_vs_unseen_from_ckpt_{args.variant}",
        "ckpt_path": ckpt_path,
        "n_excellent_patients": len(excellent_ids),
        "n_seen_patients_for_probe": len(seen_ids),
        "n_unseen_patients_for_probe": len(unseen_ids),
        "seen_cls_random_baseline": 1.0 / max(len(seen_ids), 1),
        "best_seen_classification_acc": float(best_seen_cls_acc),
        "unseen_detection_auroc": final_auroc,
        "unseen_detection_best_threshold": float(thr),
        "unseen_detection_best_binary_acc": float(bin_acc),
    }

    out_json = os.path.join(
        os.path.dirname(__file__), "results", f"v21_probeA_seen_unseen_{args.variant}.json"
    )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    run.log(
        {
            "final/seen_cls_acc": float(best_seen_cls_acc),
            "final/unseen_det_auroc": float(final_auroc),
            "final/unseen_det_best_binary_acc": float(bin_acc),
        }
    )
    logger.info(f"[v21 ProbeA] done variant={args.variant} auroc={final_auroc:.4f} seen_acc={best_seen_cls_acc:.4f}")

    run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

