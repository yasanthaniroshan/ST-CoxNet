"""
CPC + Within-Patient Temporal Ranking  (v14)

Key improvements over v13:
  1. Within-patient ranking loss — prevents learning patient-identity shortcuts
  2. Patient-balanced batching   — P patients × K segments per batch
  3. Reduced segment overlap     — stride 100 (90%) vs 20 (98%)
  4. Cox early stopping          — prevents overfitting in Phase 2
  5. Within-SR Spearman metric   — the true measure of temporal signal quality
"""

from Model.CPC import CPC
from Model.CoxHead.Base import CoxHead
from Loss.DeepSurvLoss import DeepSurvLoss
from Loss.TemporalRankingLoss import TemporalRankingLoss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset, PatientBatchSampler
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def compute_baseline_hazard(risk, time, event):
    order = torch.argsort(time)
    time, event, risk = time[order], event[order], risk[order]
    unique_times = torch.unique(time[event == 1])
    hazards = []
    for t in unique_times:
        d = ((time == t) & (event == 1)).sum()
        risk_set = torch.exp(risk[time >= t]).sum()
        hazards.append(d / risk_set)
    cumhaz = torch.cumsum(torch.stack(hazards), dim=0)
    return unique_times, cumhaz


def predict_median_survival(risk, times, baseline_cumhaz):
    surv = torch.exp(-baseline_cumhaz * torch.exp(risk))
    idx = torch.where(surv <= 0.5)[0]
    return times[idx[0]] if len(idx) > 0 else times[-1]


def concordance_index(pred, target):
    pred, target = pred.squeeze(), target.squeeze()
    pred_d = pred.unsqueeze(0) - pred.unsqueeze(1)
    tgt_d = target.unsqueeze(0) - target.unsqueeze(1)
    mask = tgt_d.abs() > 1e-4
    return ((pred_d * tgt_d) > 0).float()[mask].mean()


def compute_within_patient_spearman(emb, ctx, tte, pids, n_per_patient=50):
    """Spearman ρ between pairwise cosine sim and TTE proximity, per patient."""
    combined = np.concatenate([emb, ctx], axis=1)
    norms = np.clip(np.linalg.norm(combined, axis=1, keepdims=True), 1e-8, None)
    combined = combined / norms

    rhos = []
    for pid in np.unique(pids):
        m = pids == pid
        z_p, t_p = combined[m], tte[m]
        if len(z_p) < 10 or t_p.max() - t_p.min() < 1.0:
            continue
        if len(z_p) > n_per_patient:
            idx = np.random.choice(len(z_p), n_per_patient, replace=False)
            z_p, t_p = z_p[idx], t_p[idx]

        sim = z_p @ z_p.T
        dist = np.abs(t_p[:, None] - t_p[None, :])
        tri = np.triu_indices(len(z_p), k=1)
        rho, _ = spearmanr(sim[tri], -dist[tri])
        if not np.isnan(rho):
            rhos.append(rho)

    return float(np.mean(rhos)) if rhos else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    config = {
        # Dataset
        "afib_length": 60 * 60,
        "sr_length": int(1.5 * 60 * 60),
        "number_of_windows_in_segment": 10,
        "stride": 100,
        "window_size": 100,
        "validation_split": 0.15,
        # CPC encoder
        "cpc_epochs": 50,
        "dropout": 0.2,
        "temperature": 0.2,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 6,
        "cpc_lr": 1e-3,
        # Within-patient temporal ranking
        "temporal_loss_weight": 1.0,
        "temporal_temperature": 0.07,
        "temporal_sigma": 0.15,
        "temporal_projection_dim": 64,
        "temporal_warmup_epochs": 3,
        # Patient batching  (batch_size = P × K)
        "P": 16,
        "K": 32,
        "batches_per_epoch": 50,
        "val_batches_per_epoch": 20,
        # Cox head
        "cox_epochs": 50,
        "cox_lr": 1e-3,
        "cox_batch_size": 512,
        "cox_patience": 10,
    }

    run = wandb.init(entity="eml-labs", project="CPC-New-Temporal-Ranking", config=config)

    torch.manual_seed(42)
    np.random.seed(42)

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

    train_ds = CPCTemporalDataset(**ds_args, train=True)
    val_ds   = CPCTemporalDataset(**ds_args, train=False)
    logger.info(f"Train: {len(train_ds)} segments | Val: {len(val_ds)} segments")

    P, K = config["P"], config["K"]

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PatientBatchSampler(
            train_ds.patient_ids, P, K, config["batches_per_epoch"]
        ),
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    val_full_loader = DataLoader(
        val_ds, batch_size=config["cox_batch_size"], shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Models ──
    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    temporal_proj = nn.Sequential(
        nn.Linear(config["context_dim"] + config["latent_dim"], 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, config["temporal_projection_dim"]),
    ).to(device)

    temporal_loss_fn = TemporalRankingLoss(
        temperature=config["temporal_temperature"],
        sigma=config["temporal_sigma"],
    )

    cpc_optimizer = optim.AdamW(
        list(cpc.parameters()) + list(temporal_proj.parameters()),
        lr=config["cpc_lr"], weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        cpc_optimizer, T_max=config["cpc_epochs"], eta_min=config["cpc_lr"] * 0.1,
    )

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1 — CPC + Within-Patient Temporal Ranking
    # ══════════════════════════════════════════════════════════════════
    pbar = tqdm(total=config["cpc_epochs"], desc="Phase 1")

    for epoch in range(config["cpc_epochs"]):
        cpc.train()
        temporal_proj.train()

        warmup = max(1, config["temporal_warmup_epochs"])
        alpha = config["temporal_loss_weight"] * min(1.0, (epoch + 1) / warmup)

        ep_cpc, ep_temp, ep_acc, ep_gap, ep_n = 0.0, 0.0, 0.0, 0.0, 0

        for rr, label, tte, pid in train_loader:
            rr  = rr.to(device, non_blocking=True)
            tte = tte.to(device, non_blocking=True).float().clamp(min=0)

            cpc_optimizer.zero_grad()

            loss_cpc, accuracy, z_seq, c_seq = cpc(rr)

            h = torch.cat([z_seq[:, -1, :], c_seq[:, -1, :]], dim=-1)
            h_proj = temporal_proj(h)
            loss_temporal, t_met = temporal_loss_fn(h_proj, tte, P=P, K=K)

            loss = loss_cpc + alpha * loss_temporal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cpc.parameters()) + list(temporal_proj.parameters()), 1.0,
            )
            cpc_optimizer.step()

            ep_cpc  += loss_cpc.item()
            ep_temp += loss_temporal.item()
            ep_acc  += accuracy * rr.size(0)
            ep_gap  += t_met.get("sim_gap", 0.0)
            ep_n    += 1

        scheduler.step()

        # ── Validation (full pass) ──
        cpc.eval()
        temporal_proj.eval()

        v_cpc, v_acc, v_n = 0.0, 0.0, 0
        all_emb, all_ctx, all_lab, all_tte, all_pid = [], [], [], [], []

        with torch.no_grad():
            for rr, label, tte, pid in val_full_loader:
                rr = rr.to(device, non_blocking=True)
                loss_cpc_v, acc_v, z_seq, c_seq = cpc(rr)

                v_cpc += loss_cpc_v.item()
                v_acc += acc_v * rr.size(0)
                v_n   += rr.size(0)

                all_emb.append(z_seq[:, -1, :].cpu().numpy())
                all_ctx.append(c_seq[:, -1, :].cpu().numpy())
                all_lab.extend(label.numpy().tolist())
                all_tte.extend(tte.numpy().tolist())
                all_pid.extend(pid.numpy().tolist())

        emb  = np.concatenate(all_emb)
        ctx  = np.concatenate(all_ctx)
        labs = np.array(all_lab)
        ttes = np.clip(np.nan_to_num(np.array(all_tte)), 0, None)
        pids = np.array(all_pid)

        # Within-patient Spearman (SR only — the key metric)
        sr_mask = labs == -1
        sr_spearman = compute_within_patient_spearman(
            emb[sr_mask], ctx[sr_mask], ttes[sr_mask], pids[sr_mask],
        )

        # ── PCA visualisation (2×2: label + TTE) ──
        pca_emb = PCA(n_components=2, random_state=42).fit_transform(emb)
        pca_ctx = PCA(n_components=2, random_state=42).fit_transform(ctx)

        cmap_label = {-1: "blue", 0: "green", 1: "red"}
        lcolors = [cmap_label[l] for l in all_lab]

        fig = plt.figure(figsize=(18, 16))
        fig.suptitle(f"Epoch {epoch+1}  |  α={alpha:.2f}  |  SR Spearman ρ={sr_spearman:.4f}", fontsize=14)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(pca_emb[:, 0], pca_emb[:, 1], c=lcolors, alpha=0.5, s=6)
        ax.set_title("Embeddings — label")

        ax = fig.add_subplot(gs[0, 1])
        sc = ax.scatter(pca_emb[:, 0], pca_emb[:, 1], c=ttes, cmap="RdYlGn", alpha=0.5, s=6)
        ax.set_title("Embeddings — TTE")
        plt.colorbar(sc, ax=ax, label="TTE (s)", shrink=0.8)

        ax = fig.add_subplot(gs[1, 0])
        ax.scatter(pca_ctx[:, 0], pca_ctx[:, 1], c=lcolors, alpha=0.5, s=6)
        ax.set_title("Context — label")

        ax = fig.add_subplot(gs[1, 1])
        sc = ax.scatter(pca_ctx[:, 0], pca_ctx[:, 1], c=ttes, cmap="RdYlGn", alpha=0.5, s=6)
        ax.set_title("Context — TTE")
        plt.colorbar(sc, ax=ax, label="TTE (s)", shrink=0.8)

        legend_h = [mpatches.Patch(color=c, label=l) for c, l in zip(["blue", "green", "red"], ["SR", "Mixed", "AF"])]
        fig.legend(handles=legend_h, loc="upper right")

        # ── Log ──
        avg = lambda tot, n: tot / max(n, 1)

        metrics = {
            "cpc_epoch":              epoch + 1,
            "cpc_lr":                 cpc_optimizer.param_groups[0]["lr"],
            "alpha":                  alpha,
            "train/cpc_loss":         avg(ep_cpc, ep_n),
            "train/temporal_loss":    avg(ep_temp, ep_n),
            "train/cpc_accuracy":     avg(ep_acc, P * K * ep_n),
            "train/sim_gap":          avg(ep_gap, ep_n),
            "val/cpc_loss":           avg(v_cpc, len(val_full_loader)),
            "val/cpc_accuracy":       (v_acc / v_n) if v_n else 0,
            "val/sr_spearman":        sr_spearman,
            "pca":                    wandb.Image(fig),
        }
        run.log(metrics)
        plt.close(fig)

        pbar.update(1)
        pbar.set_description(f"Epoch {epoch+1}")
        pbar.write(
            f"Epoch {epoch+1:>3} | "
            f"CPC {avg(ep_cpc,ep_n):.4f}/{avg(v_cpc,len(val_full_loader)):.4f} | "
            f"Temp {avg(ep_temp,ep_n):.4f} | "
            f"Acc {avg(ep_acc,P*K*ep_n):.1f}% | "
            f"Gap {avg(ep_gap,ep_n):.4f} | "
            f"SR-ρ {sr_spearman:.4f}"
        )

    pbar.close()

    torch.save(cpc.state_dict(), "cpc_model.pth")
    torch.save(temporal_proj.state_dict(), "temporal_proj.pth")
    art = wandb.Artifact("cpc_model_v14", type="model")
    art.add_file("cpc_model.pth")
    art.add_file("temporal_proj.pth")
    run.log_artifact(art)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2 — Cox Survival Head  (frozen encoder, early stopping)
    # ══════════════════════════════════════════════════════════════════
    sr_train = CPCTemporalDataset(**ds_args, train=True,  sr_only=True)
    sr_val   = CPCTemporalDataset(**ds_args, train=False, sr_only=True)
    logger.info(f"Cox SR — Train: {len(sr_train)} | Val: {len(sr_val)}")

    sr_train_loader = DataLoader(
        sr_train, batch_size=config["cox_batch_size"], shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    sr_val_loader = DataLoader(
        sr_val, batch_size=config["cox_batch_size"], shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    cox = CoxHead(
        context_dim=config["context_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    ).to(device)

    cox_optimizer = optim.AdamW(cox.parameters(), lr=config["cox_lr"], weight_decay=1e-2)
    cox_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        cox_optimizer, T_max=config["cox_epochs"], eta_min=config["cox_lr"] * 0.1,
    )
    deepsurv = DeepSurvLoss()

    cpc.eval()
    for p in cpc.parameters():
        p.requires_grad = False

    best_ci = 0.0
    patience_ctr = 0
    pbar = tqdm(total=config["cox_epochs"], desc="Phase 2 — Cox")

    for cox_ep in range(config["cox_epochs"]):
        cox.train()
        t_loss = 0.0
        t_risk, t_time, t_event = [], [], []

        for rr, label, tte, pid in sr_train_loader:
            rr  = rr.to(device, non_blocking=True)
            tte = tte.to(device, non_blocking=True).float().clamp(min=0)
            events = torch.ones_like(tte)

            cox_optimizer.zero_grad()
            with torch.no_grad():
                _, _, z_seq, c_seq = cpc(rr)

            logits = cox(c_seq[:, -1, :], z_seq[:, -1, :])
            loss = deepsurv(logits, tte, events)
            loss.backward()
            cox_optimizer.step()

            t_loss += loss.item()
            t_risk.append(logits.detach().cpu())
            t_time.append(tte.cpu())
            t_event.append(events.cpu())

        cox_scheduler.step()

        # Baseline hazard from training
        t_risk  = torch.cat(t_risk)
        t_time  = torch.cat(t_time)
        t_event = torch.cat(t_event)
        base_times, base_cumhaz = compute_baseline_hazard(t_risk, t_time, t_event)

        # Validation
        cox.eval()
        v_loss = 0.0
        v_risk, v_time, v_event = [], [], []
        actual_t, pred_t = [], []

        with torch.no_grad():
            for rr, label, tte, pid in sr_val_loader:
                rr  = rr.to(device, non_blocking=True)
                tte = tte.to(device, non_blocking=True).float().clamp(min=0)
                events = torch.ones_like(tte)

                _, _, z_seq, c_seq = cpc(rr)
                logits = cox(c_seq[:, -1, :], z_seq[:, -1, :])
                loss = deepsurv(logits, tte, events)
                v_loss += loss.item()

                v_risk.append(logits.cpu())
                v_time.append(tte.cpu())
                v_event.append(events.cpu())
                actual_t.append(tte.cpu().numpy())

                batch_pred = [
                    predict_median_survival(r, base_times, base_cumhaz).item()
                    for r in logits.cpu()
                ]
                pred_t.append(batch_pred)

        v_risk = torch.cat(v_risk)
        v_time = torch.cat(v_time)
        predictions = torch.tensor([x for sub in pred_t for x in sub])
        ci = concordance_index(predictions.unsqueeze(1), v_time.unsqueeze(1)).item()

        # ── Cox plots ──
        np_actual = np.concatenate(actual_t)
        np_pred   = np.concatenate(pred_t)

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(f"Cox Epoch {cox_ep+1}  |  C-index = {ci:.4f}", fontsize=14)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(np_actual, np_pred, alpha=0.3, s=4)
        lim = max(np_actual.max(), np_pred.max()) * 1.05
        ax1.plot([0, lim], [0, lim], "r--", lw=1)
        ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
        ax1.set_xlabel("Actual TTE (s)"); ax1.set_ylabel("Predicted TTE (s)")
        ax1.set_title("Actual vs Predicted"); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(v_risk.numpy(), bins=50, alpha=0.7)
        ax2.set_xlabel("Log-hazard"); ax2.set_title("Risk distribution"); ax2.grid(alpha=0.3)
        plt.tight_layout()

        run.log({
            "cox_epoch":          cox_ep + 1,
            "cox/train_loss":     t_loss / len(sr_train_loader),
            "cox/val_loss":       v_loss / len(sr_val_loader),
            "cox/concordance":    ci,
            "cox/plot":           wandb.Image(fig),
        })
        plt.close(fig)

        pbar.update(1)
        pbar.write(
            f"Cox {cox_ep+1:>3} | "
            f"Loss {t_loss/len(sr_train_loader):.4f}/{v_loss/len(sr_val_loader):.4f} | "
            f"C-index {ci:.4f} {'*' if ci > best_ci else ''}"
        )

        # Early stopping
        if ci > best_ci:
            best_ci = ci
            patience_ctr = 0
            torch.save(cox.state_dict(), "cox_best.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= config["cox_patience"]:
                pbar.write(f"Early stopping at epoch {cox_ep+1} (best C-index: {best_ci:.4f})")
                break

    pbar.close()

    cox.load_state_dict(torch.load("cox_best.pth", weights_only=True))
    torch.save(cox.state_dict(), "cox_model.pth")
    art = wandb.Artifact("cox_model_v14", type="model")
    art.add_file("cox_model.pth")
    run.log_artifact(art)
    logger.info(f"Best C-index: {best_ci:.4f}")

except Exception as e:
    if "logger" in locals():
        logger.error(f"Error: {e}", exc_info=True)
    else:
        print(f"Error: {e}")
    if "pbar" in locals():
        pbar.close()
finally:
    if "run" in locals():
        run.finish()
