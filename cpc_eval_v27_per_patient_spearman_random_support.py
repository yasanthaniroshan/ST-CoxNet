"""
Evaluate v27 robustness to RANDOM support subsets.

For each val patient (excellent-only, patient-level split), repeat:
  - sample `support_k` segments uniformly at random as support
  - use all remaining segments as query
  - run model(support, query) and compute Spearman(preds, tte)

This produces a per-patient distribution of Spearman rhos, to test whether
a few patients (or specific support choices) dominate the overall rho.
"""

import json
import os
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from scipy.stats import spearmanr

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
    def __init__(
        self,
        latent_dim=64,
        hidden_dim=128,
        dropout=0.2,
        temperature=0.1,
    ):
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

        # Cosine attention over support (per query)
        h_support_n = h_support / (h_support.norm(dim=-1, keepdim=True) + 1e-8)
        h_query_n = h_query / (h_query.norm(dim=-1, keepdim=True) + 1e-8)

        sim = torch.einsum("pqd,psd->pqs", h_query_n, h_support_n)  # [P,Q,S]
        weights = torch.softmax(sim / self.temperature, dim=-1)      # [P,Q,S]

        context = torch.einsum("pqs,psd->pqd", weights, h_support)  # [P,Q,D]

        context_flat = context.reshape(p * q_k, -1)  # [P*Q, D]
        gamma_flat, beta_flat = self.film(context_flat)
        gamma = gamma_flat.view(p, q_k, -1)
        beta = beta_flat.view(p, q_k, -1)

        h_mod = gamma * h_query + beta
        preds = self.head(h_mod).squeeze(-1)  # [P,Q]
        return preds


def relative_rr_remove_patient_baseline(all_data: torch.Tensor, all_pids: torch.Tensor, eps=1e-8):
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


def build_splits(all_pids: torch.Tensor, validation_split: float, seed: int):
    np.random.seed(seed)
    uniq = np.unique(all_pids.cpu().numpy())
    np.random.shuffle(uniq)
    split = int(len(uniq) * (1 - validation_split))
    train_pids = set(uniq[:split].tolist())
    val_pids = set(uniq[split:].tolist())
    assert not train_pids.intersection(val_pids)
    return train_pids, val_pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_support_samples", type=int, default=30)
    parser.add_argument("--support_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="v27_best.pth")
    args = parser.parse_args()

    config = {
        "seed": args.seed,
        "afib_length": 3600,
        "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        "support_k": args.support_k,
        "query_k": 999999,  # unused; we use "all remaining as query" for robustness
        "latent_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "temperature": 0.1,
        "relative_rr": True,
        "relative_rr_eps": 1e-8,
        "checkpoint": args.checkpoint,
    }

    # Excellent patient pool
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

    # Filter excellent-only
    mask_ex = np.isin(all_pids.numpy(), list(excellent_pids))
    all_data = all_data[mask_ex]
    all_times = all_times[mask_ex]
    all_pids = all_pids[mask_ex]
    # We need tte min in minutes from all_times (already is times in seconds from dataset)
    all_tte_min = all_times.float().clamp(min=0) / 60.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorFiLMConditionalAttention(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    if not os.path.exists(config["checkpoint"]):
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint']}")
    state = torch.load(config["checkpoint"], weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Apply relative RR baseline per patient
    if config["relative_rr"]:
        logger.info("Applying relative RR preprocessing...")
        all_data, _ = relative_rr_remove_patient_baseline(all_data, all_pids, eps=config["relative_rr_eps"])

    train_pids, val_pids = build_splits(all_pids, config["validation_split"], config["seed"])
    all_pids_np = all_pids.cpu().numpy().astype(int)

    rng = np.random.default_rng(config["seed"])
    per_patient_samples = {}

    for pid in sorted(val_pids):
        idx_pool = np.where(all_pids_np == pid)[0]
        if len(idx_pool) <= args.support_k:
            continue

        rhos = []
        for _ in range(args.n_support_samples):
            # Sample support as a subset of GLOBAL segment indices for this patient.
            s_idx = rng.choice(idx_pool, size=args.support_k, replace=False)
            q_idx = np.setdiff1d(idx_pool, s_idx, assume_unique=False)
            if q_idx.size < 2:
                continue

            support_rr = all_data[s_idx].unsqueeze(0).to(device)  # [1,S,steps,W]
            query_rr = all_data[q_idx].unsqueeze(0).to(device)      # [1,Q,steps,W]
            query_tte = all_tte_min[q_idx].cpu().numpy()

            with torch.no_grad():
                preds = model(support_rr, query_rr).squeeze(0).cpu().numpy()  # [Q]

            rho, _ = spearmanr(preds, query_tte)
            if not np.isnan(rho):
                rhos.append(float(rho))

        per_patient_samples[int(pid)] = rhos

    # Aggregate
    summary = {
        "support_k": args.support_k,
        "n_support_samples": args.n_support_samples,
        "checkpoint": args.checkpoint,
        "val_patients_total": len(val_pids),
        "val_patients_evaluated": len(per_patient_samples),
    }

    # Per-patient summary stats
    per_patient_summary = {}
    all_rhos = []
    for pid, rhos in per_patient_samples.items():
        if len(rhos) == 0:
            continue
        arr = np.array(rhos, dtype=float)
        per_patient_summary[pid] = {
            "n": int(len(arr)),
            "mean_rho": float(arr.mean()),
            "median_rho": float(np.median(arr)),
            "q25_rho": float(np.quantile(arr, 0.25)),
            "q75_rho": float(np.quantile(arr, 0.75)),
            "frac_pos": float((arr > 0).mean()),
            "min_rho": float(arr.min()),
            "max_rho": float(arr.max()),
        }
        all_rhos.extend(rhos)

    all_rhos = np.array(all_rhos, dtype=float) if len(all_rhos) else np.array([], dtype=float)
    summary.update(
        {
            "overall_mean_rho": float(all_rhos.mean()) if len(all_rhos) else 0.0,
            "overall_median_rho": float(np.median(all_rhos)) if len(all_rhos) else 0.0,
            "overall_frac_pos": float((all_rhos > 0).mean()) if len(all_rhos) else 0.0,
        }
    )

    # Identify top/bottom by mean_rho
    ranked = sorted(
        per_patient_summary.items(),
        key=lambda kv: kv[1]["mean_rho"],
        reverse=True,
    )
    top5 = ranked[:5]
    bottom5 = ranked[-5:][::-1]

    print("Per-patient rho across random support subsets:")
    for pid, stats in ranked:
        print(
            f"  PID {pid}: mean={stats['mean_rho']:.3f} median={stats['median_rho']:.3f} "
            f"frac_pos={stats['frac_pos']:.2f} n={stats['n']}"
        )

    out = {
        "config": config,
        "summary": summary,
        "per_patient_summary": per_patient_summary,
        "top_by_mean": top5,
        "bottom_by_mean": bottom5,
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "v27_per_patient_spearman_random_support.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

