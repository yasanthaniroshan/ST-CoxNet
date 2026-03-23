"""
Evaluation-only script for v27:
Compute per-patient validation Spearman rho distribution for the v27 best checkpoint.

Goal:
Determine whether a small number of val patients drive the overall rho.
"""

import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FiLM(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, context: torch.Tensor):
        # context: [N, hidden_dim]
        params = self.mlp(context)
        delta_gamma, beta = params.chunk(2, dim=-1)
        gamma = 1.0 + delta_gamma
        return gamma, beta


class TTEPredictorFiLMConditionalAttention(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2, temperature=0.1):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.temperature = float(temperature)

        self.film = FiLM(hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_h(self, rr_windows: torch.Tensor) -> torch.Tensor:
        # rr_windows: [N, steps, window_size]
        n, steps, w = rr_windows.shape
        z = self.encoder(rr_windows.view(n * steps, w)).view(n, steps, -1)
        _, h = self.gru(z)
        return h.squeeze(0)  # [N, hidden_dim]

    def forward(self, support_rr: torch.Tensor, query_rr: torch.Tensor) -> torch.Tensor:
        # support_rr: [P, S, steps, W], query_rr: [P, Q, steps, W]
        p, s_k, steps, w = support_rr.shape
        _, q_k, _, _ = query_rr.shape

        support_flat = support_rr.view(p * s_k, steps, w)
        query_flat = query_rr.view(p * q_k, steps, w)

        h_support = self.encode_h(support_flat).view(p, s_k, -1)  # [P,S,D]
        h_query = self.encode_h(query_flat).view(p, q_k, -1)      # [P,Q,D]

        # Cosine attention over support (query-specific context)
        h_support_n = h_support / (h_support.norm(dim=-1, keepdim=True) + 1e-8)
        h_query_n = h_query / (h_query.norm(dim=-1, keepdim=True) + 1e-8)

        sim = torch.einsum("pqd,psd->pqs", h_query_n, h_support_n)  # [P,Q,S]
        weights = torch.softmax(sim / self.temperature, dim=-1)     # [P,Q,S]
        context = torch.einsum("pqs,psd->pqd", weights, h_support)  # [P,Q,D]

        # FiLM params per query token
        context_flat = context.reshape(p * q_k, -1)  # [P*Q,D]
        gamma_flat, beta_flat = self.film(context_flat)
        gamma = gamma_flat.view(p, q_k, -1)
        beta = beta_flat.view(p, q_k, -1)

        h_mod = gamma * h_query + beta
        preds = self.head(h_mod).squeeze(-1)  # [P,Q]
        return preds


def relative_rr_remove_patient_baseline(all_data, all_pids, eps=1e-8):
    out = all_data.clone()
    pids_np = np.unique(all_pids.cpu().numpy())
    stats = {}
    for pid in pids_np:
        mask = all_pids == int(pid)
        x = out[mask]
        mu = float(x.mean().item())
        sd = float(x.std(unbiased=False).item())
        out[mask] = (x - mu) / (sd + eps)
        stats[int(pid)] = (mu, sd)
    return out, stats


def compute_per_patient_spearman(scores, tte_min, pids, min_segments=10):
    out = {}
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() < min_segments:
            continue
        s = scores[m]
        tt = tte_min[m]
        if (tt.max() - tt.min()) < 1e-6:
            continue
        rho, _ = spearmanr(s, tt)
        if not np.isnan(rho):
            out[int(pid)] = float(rho)
    return out


def main():
    # Match v27 config
    config = {
        "seed": 42,
        "afib_length": 3600,
        "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        "support_k": 5,
        "query_k": 3,
        # Model
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "temperature": 0.1,
        "relative_rr": True,
        "relative_rr_eps": 1e-8,
    }

    # Load excellent patient pool
    with open("v15c_patient_quality.json", "r") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == "excellent"}

    processed = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    ds_args = dict(
        processed_dataset_path=processed,
        afib_length=config["afib_length"],
        sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"],
        window_size=config["window_size"],
        validation_split=config["validation_split"],
    )

    train_ds_full = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds_full = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    all_data = torch.cat([train_ds_full.data, val_ds_full.data], dim=0)
    all_times = torch.cat([train_ds_full.times, val_ds_full.times], dim=0)
    all_pids = torch.cat([train_ds_full.patient_ids, val_ds_full.patient_ids], dim=0)

    # Filter excellent patients
    mask_ex = np.isin(all_pids.numpy(), list(excellent_pids))
    all_data = all_data[mask_ex]
    all_times = all_times[mask_ex]
    all_pids = all_pids[mask_ex]

    all_tte_min = all_times.float().clamp(min=0) / 60.0

    if config["relative_rr"]:
        all_data, _ = relative_rr_remove_patient_baseline(
            all_data, all_pids, eps=config["relative_rr_eps"]
        )

    # Recreate patient-level split (v27 uses np.random.seed + np.random.shuffle on uniq pids)
    np.random.seed(config["seed"])
    uniq = np.unique(all_pids.numpy())
    np.random.shuffle(uniq)
    split = int(len(uniq) * (1 - config["validation_split"]))
    train_pids = set(uniq[:split].tolist())
    val_pids = set(uniq[split:].tolist())
    assert not train_pids.intersection(val_pids)

    # Load v27 best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorFiLMConditionalAttention(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    ckpt_candidates = ["v27_best.pth", os.path.join(os.path.dirname(__file__), "v27_best.pth")]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            "Could not find `v27_best.pth`. Run the v27 training from the project root so it saves there."
        )

    state = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Deterministic per-patient evaluation consistent with v27 validation loop:
    # idx_pool = val_indices_by_pid[pid] which is np.where(all_pids_np == pid)[0]
    # s_idx = idx_pool[:support_k], q_idx = idx_pool[support_k:]
    all_pids_np = all_pids.numpy().astype(int)
    per_pid_rho = {}

    for pid in sorted(val_pids):
        idx_pool = np.where(all_pids_np == pid)[0]
        if len(idx_pool) <= config["support_k"]:
            continue
        s_idx = idx_pool[: config["support_k"]]
        q_idx = idx_pool[config["support_k"] :]

        support_rr = all_data[s_idx].unsqueeze(0).to(device)  # [1,S,steps,W]
        query_rr = all_data[q_idx].unsqueeze(0).to(device)      # [1,Q,steps,W]
        query_tte = all_tte_min[q_idx].cpu().numpy()           # [Q]

        with torch.no_grad():
            preds = model(support_rr, query_rr).squeeze(0).cpu().numpy()  # [Q]

        rho, _ = spearmanr(preds, query_tte)
        if not np.isnan(rho):
            per_pid_rho[int(pid)] = float(rho)

    # Summary stats
    rhos = np.array(list(per_pid_rho.values()), dtype=float)
    frac_pos = float((rhos > 0).mean()) if len(rhos) else 0.0
    stats = {
        "n_val_patients": int(len(val_pids)),
        "n_evaluated": int(len(rhos)),
        "mean_rho": float(rhos.mean()) if len(rhos) else 0.0,
        "median_rho": float(np.median(rhos)) if len(rhos) else 0.0,
        "q25_rho": float(np.quantile(rhos, 0.25)) if len(rhos) else 0.0,
        "q75_rho": float(np.quantile(rhos, 0.75)) if len(rhos) else 0.0,
        "frac_positive_rho": frac_pos,
    }

    top3 = sorted(per_pid_rho.items(), key=lambda kv: kv[1], reverse=True)[:3]
    bot3 = sorted(per_pid_rho.items(), key=lambda kv: kv[1], reverse=False)[:3]

    print("Per-patient Spearman rho (val, v27_best):")
    for pid, rho in sorted(per_pid_rho.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  PID {pid}: rho={rho:.4f}")

    print("\nSummary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nTop 3 patients: {top3}")
    print(f"Bottom 3 patients: {bot3}")

    out = {
        "checkpoint": ckpt_path,
        "config": config,
        "per_patient_rho": per_pid_rho,
        "summary": stats,
        "top3": top3,
        "bottom3": bot3,
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "v27_per_patient_spearman.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Saved per-patient Spearman to: {out_path}")


if __name__ == "__main__":
    main()

