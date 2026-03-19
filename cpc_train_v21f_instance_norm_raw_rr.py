"""
v21f — InstanceNorm on GRU embedding + raw RR (no baseline removal).

Excellent-only patients, patient-level 0.85/0.15 split.
"""

import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from Model.Encoder.BaseEncoder import Encoder
from Utils.Dataset.CPCTemporalDataset import CPCTemporalDataset

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TTEPredictorIN(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        # InstanceNorm1d expects input [N, C, L] with L>1 during training.
        # Apply it over the GRU sequence output (L=T) rather than the final
        # hidden vector.
        self.instance_norm = nn.InstanceNorm1d(hidden_dim, affine=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def _get_h(self, rr_windows):
        B, T, W = rr_windows.shape
        z = self.encoder(rr_windows.view(B * T, W)).view(B, T, -1)
        out_seq, _ = self.gru(z)  # [B, T, hidden_dim]
        out_seq = out_seq.transpose(1, 2)  # [B, hidden_dim, T]
        out_seq = self.instance_norm(out_seq)  # normalize across T
        h = out_seq.mean(dim=2)  # [B, hidden_dim]
        return h

    def forward(self, rr_windows):
        return self.head(self._get_h(rr_windows)).squeeze(-1)

    @torch.no_grad()
    def extract_h(self, rr_windows):
        return self._get_h(rr_windows)


def compute_per_patient_rho(preds, actual, pids, min_segments=10):
    results = {}
    for pid in np.unique(pids):
        m = pids == pid
        if m.sum() < min_segments: continue
        p, a = preds[m], actual[m]
        if a.max() - a.min() < 1e-6: continue
        rho, _ = spearmanr(p, a)
        if not np.isnan(rho): results[int(pid)] = rho
    return results


def pairwise_l2(a, b):
    aa = np.sum(a*a, axis=1, keepdims=True); bb = np.sum(b*b, axis=1, keepdims=True).T
    return np.sqrt(np.clip(aa + bb - 2.0*(a @ b.T), 0.0, None))


def centroid_diagnostic(model, train_loader, val_loader, train_pid_set, val_pid_set, device):
    model.eval(); emb_all, pid_all = [], []
    with torch.no_grad():
        for ld in [train_loader, val_loader]:
            for rr, _, pid in ld:
                emb_all.append(model.extract_h(rr.to(device)).cpu().numpy()); pid_all.append(pid.numpy())
    emb_all = np.concatenate(emb_all); pid_all = np.concatenate(pid_all).astype(int)
    tr_mask = np.isin(pid_all, list(train_pid_set)); va_mask = np.isin(pid_all, list(val_pid_set))
    tr_emb, tr_pid = emb_all[tr_mask], pid_all[tr_mask]; va_emb = emb_all[va_mask]
    cids = sorted(train_pid_set)
    centroids = np.stack([tr_emb[tr_pid == p].mean(0) for p in cids])
    pid2c = {p: i for i, p in enumerate(cids)}
    d_tr = pairwise_l2(tr_emb, centroids); pred_c = np.argmin(d_tr, axis=1)
    true_c = np.array([pid2c[p] for p in tr_pid]); seen_acc = float((pred_c == true_c).mean())
    md_tr = d_tr[np.arange(len(d_tr)), pred_c]
    d_va = pairwise_l2(va_emb, centroids); md_va = np.min(d_va, axis=1)
    y = np.concatenate([np.zeros(len(md_tr), dtype=int), np.ones(len(md_va), dtype=int)])
    s = np.concatenate([md_tr, md_va])
    try: auroc = float(roc_auc_score(y, s))
    except ValueError: auroc = 0.5
    return {"seen_acc": seen_acc, "unseen_auroc": auroc}


try:
    console = logging.StreamHandler(); console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)

    config = {
        "afib_length": 3600, "sr_length": int(1.5 * 3600),
        "number_of_windows_in_segment": 10, "stride": 100, "window_size": 100,
        "validation_split": 0.15,
        "epochs": 200, "batch_size": 64, "lr": 1e-3,
        "latent_dim": 64, "hidden_dim": 128, "dropout": 0.2, "patience": 20,
        "relative_rr": False,
    }

    run = wandb.init(entity="eml-labs", project="CPC-New-Temporal-Ranking",
                     name="v21f-instancenorm-raw-rr", config=config)
    torch.manual_seed(42); np.random.seed(42)

    with open("v15c_patient_quality.json") as f:
        quality = json.load(f)
    excellent_pids = {int(p["pid"]) for p in quality["patients"] if p["tier"] == "excellent"}
    logger.info(f"Excellent patients: {len(excellent_pids)}")

    processed = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    ds_args = dict(processed_dataset_path=processed, afib_length=config["afib_length"],
                   sr_length=config["sr_length"],
                   number_of_windows_in_segment=config["number_of_windows_in_segment"],
                   stride=config["stride"], window_size=config["window_size"],
                   validation_split=config["validation_split"])

    train_ds = CPCTemporalDataset(**ds_args, train=True, sr_only=True)
    val_ds = CPCTemporalDataset(**ds_args, train=False, sr_only=True)

    all_data = torch.cat([train_ds.data, val_ds.data])
    all_times = torch.cat([train_ds.times, val_ds.times])
    all_pids = torch.cat([train_ds.patient_ids, val_ds.patient_ids])

    mask_ex = np.isin(all_pids.numpy(), list(excellent_pids))
    all_data = all_data[mask_ex]; all_times = all_times[mask_ex]; all_pids = all_pids[mask_ex]
    all_tte_min = all_times.float().clamp(min=0) / 60.0

    # NO relative RR for v21f

    uniq = np.unique(all_pids.numpy()); np.random.shuffle(uniq)
    sp = int(len(uniq) * (1 - config["validation_split"]))
    train_pid_set = set(uniq[:sp].tolist()); val_pid_set = set(uniq[sp:].tolist())
    assert not train_pid_set & val_pid_set

    pnp = all_pids.numpy()
    tr_m = np.isin(pnp, list(train_pid_set)); va_m = np.isin(pnp, list(val_pid_set))
    train_data, train_tte, train_pid_seg = all_data[tr_m], all_tte_min[tr_m], all_pids[tr_m]
    val_data, val_tte, val_pid_seg = all_data[va_m], all_tte_min[va_m], all_pids[va_m]

    logger.info(f"Patient split: {len(train_pid_set)} train | {len(val_pid_set)} val")
    logger.info(f"Segments: {len(train_data)} train | {len(val_data)} val")

    train_loader = DataLoader(TensorDataset(train_data, train_tte, train_pid_seg),
                              batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_data, val_tte, val_pid_seg),
                            batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TTEPredictorIN(config["latent_dim"], config["hidden_dim"], config["dropout"]).to(device)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | {device}")

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"]*0.01)
    loss_fn = nn.SmoothL1Loss()

    pbar = tqdm(total=config["epochs"], desc="v21f IN+rawRR")
    best_val_mae = float("inf"); patience_ctr = 0

    for epoch in range(config["epochs"]):
        model.train(); t_loss, t_n = 0.0, 0
        t_preds, t_actual, t_pids_ep = [], [], []

        for rr, tte, pid in train_loader:
            rr, tte = rr.to(device), tte.to(device)
            optimizer.zero_grad()
            pred = model(rr); loss = loss_fn(pred, tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            bsz = rr.size(0); t_loss += loss.item()*bsz; t_n += bsz
            t_preds.append(pred.detach().cpu().numpy())
            t_actual.append(tte.cpu().numpy()); t_pids_ep.append(pid.numpy())

        scheduler.step()
        t_preds = np.concatenate(t_preds); t_actual = np.concatenate(t_actual); t_pids_ep = np.concatenate(t_pids_ep)
        t_mae = np.mean(np.abs(t_preds - t_actual))
        t_rho = float(np.nan_to_num(spearmanr(t_preds, t_actual)[0], nan=0.0))

        model.eval(); v_preds, v_actual, v_pids_ep = [], [], []
        with torch.no_grad():
            for rr, tte, pid in val_loader:
                v_preds.append(model(rr.to(device)).cpu().numpy())
                v_actual.append(tte.numpy()); v_pids_ep.append(pid.numpy())
        v_preds = np.concatenate(v_preds); v_actual = np.concatenate(v_actual); v_pids_ep = np.concatenate(v_pids_ep)
        v_mae = np.mean(np.abs(v_preds - v_actual))
        v_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
        ss_res = np.sum((v_actual - v_preds)**2); ss_tot = np.sum((v_actual - v_actual.mean())**2)
        v_r2 = 1 - ss_res / max(ss_tot, 1e-8)
        v_pp = compute_per_patient_rho(v_preds, v_actual, v_pids_ep)
        v_mean_pp = float(np.mean(list(v_pp.values()))) if v_pp else 0.0

        if v_mae < best_val_mae: best_val_mae = v_mae; patience_ctr = 0; torch.save(model.state_dict(), "v21f_best.pth")
        else: patience_ctr += 1

        run.log({"epoch": epoch+1,
                 "train/loss": t_loss/max(t_n,1), "train/mae": t_mae, "train/rho": t_rho,
                 "val/mae": v_mae, "val/rho": v_rho, "val/r2": v_r2, "val/pp_rho": v_mean_pp})

        if (epoch+1) % 10 == 0 or epoch == config["epochs"]-1:
            pbar.write(f"Ep {epoch+1:>3} | MAE {t_mae:.2f}/{v_mae:.2f} | ρ {t_rho:.3f}/{v_rho:.3f} | R² {v_r2:.3f}")
        pbar.update(1)
        if patience_ctr >= config["patience"]:
            pbar.write(f"Early stop ep {epoch+1} | best MAE {best_val_mae:.2f}"); break

    pbar.close()

    model.load_state_dict(torch.load("v21f_best.pth", weights_only=True)); model.eval()
    v_preds, v_actual, v_pids_all = [], [], []
    with torch.no_grad():
        for rr, tte, pid in val_loader:
            v_preds.append(model(rr.to(device)).cpu().numpy())
            v_actual.append(tte.numpy()); v_pids_all.append(pid.numpy())
    v_preds = np.concatenate(v_preds); v_actual = np.concatenate(v_actual); v_pids_all = np.concatenate(v_pids_all)
    g_mae = np.mean(np.abs(v_preds - v_actual))
    g_rho = float(np.nan_to_num(spearmanr(v_preds, v_actual)[0], nan=0.0))
    ss_r = np.sum((v_actual - v_preds)**2); ss_t = np.sum((v_actual - v_actual.mean())**2)
    g_r2 = 1 - ss_r / max(ss_t, 1e-8)
    pp = compute_per_patient_rho(v_preds, v_actual, v_pids_all)
    g_pp = float(np.mean(list(pp.values()))) if pp else 0.0

    logger.info(f"\n{'='*65}")
    logger.info(f"FINAL v21f | MAE={g_mae:.2f} | ρ={g_rho:.4f} | R²={g_r2:.4f} | pp_ρ={g_pp:.4f}")
    logger.info(f"{'='*65}")

    cd = centroid_diagnostic(model, train_loader, val_loader, train_pid_set, val_pid_set, device)
    logger.info(f"Centroid probe: seen_acc={cd['seen_acc']:.4f} | unseen_auroc={cd['unseen_auroc']:.4f}")

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(f"v21f IN+rawRR | MAE={g_mae:.2f} | ρ={g_rho:.3f} | pp_ρ={g_pp:.3f} | R²={g_r2:.3f}", fontsize=13)
    ax = fig.add_subplot(1, 3, 1); lim = max(v_actual.max(), v_preds.max())*1.05
    ax.scatter(v_actual, v_preds, alpha=.4, s=10); ax.plot([0,lim],[0,lim],"r--",lw=1)
    ax.set_xlabel("Actual TTE (min)"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Pred"); ax.grid(alpha=.3)
    ax = fig.add_subplot(1, 3, 2); res = v_preds - v_actual
    ax.hist(res, bins=50, alpha=.7); ax.axvline(0, color="k", ls="--"); ax.set_title("Residuals"); ax.grid(alpha=.3)
    ax = fig.add_subplot(1, 3, 3)
    if pp:
        sp = sorted(pp.keys(), key=lambda p: pp[p])
        ax.barh(range(len(sp)), [pp[p] for p in sp], height=.7)
        ax.set_yticks(range(len(sp))); ax.set_yticklabels([f"P{p}" for p in sp], fontsize=7)
    ax.set_title("Per-patient ρ"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("v21f_results.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    run.log({"final/mae": g_mae, "final/rho": g_rho, "final/r2": g_r2, "final/pp_rho": g_pp,
             "final/centroid_seen_acc": cd["seen_acc"], "final/centroid_unseen_auroc": cd["unseen_auroc"]})
    logger.info("Done v21f.")

except Exception as e:
    if "logger" in locals(): logger.error(f"Error: {e}", exc_info=True)
    else: print(f"Error: {e}")
    if "pbar" in locals(): pbar.close()
finally:
    if "run" in locals(): run.finish()
