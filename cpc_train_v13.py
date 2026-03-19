from Model.CPC import CPC
from Model.PredictionHead.TimePredictor import TimePredictor
from Model.CoxHead.Base import CoxHead
from Loss.DeepSurvLoss import DeepSurvLoss
from Loss.TemporalRankingLoss import TemporalRankingLoss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Utils.Dataset.CPCDataset import CPCDataset
from Utils.Dataset.CPCCoxDataset import CPCCoxDataset
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

def compute_baseline_hazard(risk, time, event):
    order = torch.argsort(time)
    time = time[order]
    event = event[order]
    risk = risk[order]

    unique_times = torch.unique(time[event == 1])

    hazards = []
    for t in unique_times:
        d = ((time == t) & (event == 1)).sum()
        risk_set = torch.exp(risk[time >= t]).sum()
        hazards.append(d / risk_set)

    hazards = torch.stack(hazards)
    cumhaz = torch.cumsum(hazards, dim=0)

    return unique_times, cumhaz

def predict_median_survival(risk, times, baseline_cumhaz):
    surv = torch.exp(-baseline_cumhaz * torch.exp(risk))

    idx = torch.where(surv <= 0.5)[0]
    if len(idx) == 0:
        return times[-1]

    return times[idx[0]]

def concordance_index(pred, target):
    """
    pred, target: [B, 1] normalized time-to-event
    Penalizes pairs where the predicted ordering disagrees with actual ordering.
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(0) - target.unsqueeze(1)

    mask = (target_diff.abs() > 1e-4)

    concordant = ((pred_diff * target_diff) > 0).float()

    c_index = concordant[mask].mean()
    return c_index

def compute_validation_spearman(embeddings, context, tte_values, n_subsample=2000):
    """Spearman rho between pairwise cosine similarity and TTE proximity."""
    n = len(tte_values)
    if n < 10:
        return 0.0

    n_sub = min(n_subsample, n)
    idx = np.random.choice(n, n_sub, replace=False)

    combined = np.concatenate([embeddings[idx], context[idx]], axis=1)
    tte_sub = tte_values[idx]

    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    z_norm = combined / norms
    cos_sim = z_norm @ z_norm.T

    tte_dist = np.abs(tte_sub[:, None] - tte_sub[None, :])

    tri = np.triu_indices(n_sub, k=1)
    sim_flat = cos_sim[tri]
    dist_flat = tte_dist[tri]

    rho, _ = spearmanr(sim_flat, -dist_flat)
    return rho if not np.isnan(rho) else 0.0

try:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    config={
        "afib_length": 60*60,
        "sr_length": int(1.5*60*60),
        "number_of_windows_in_segment": 10,
        "stride": 20,
        "window_size": 100,
        "validation_split": 0.15,
        "cpc_epochs": 20,
        "cox_epochs": 20,
        "dropout": 0.2,
        "temperature": 0.2,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 6,
        "batch_size": 512,
        "cpc_lr": 1e-3,
        "cox_lr": 1e-3,
        # Temporal ranking loss parameters
        "temporal_loss_weight": 1.0,
        "temporal_temperature": 0.07,
        "temporal_sigma": 0.1,
        "temporal_projection_dim": 64,
        "temporal_warmup_epochs": 2,
    }
    run = wandb.init(
        entity="eml-labs",
        project="CPC-New-Temporal-Ranking",
        config=config,
    )


    torch.manual_seed(42)
    np.random.seed(42)

    afib_length = config["afib_length"]
    sr_length = config["sr_length"]
    number_of_windows_in_segment = config["number_of_windows_in_segment"]
    stride = config["stride"]
    window_size = config["window_size"]
    validation_split = config["validation_split"]
    cpc_epochs = config["cpc_epochs"]
    cox_epochs = config["cox_epochs"]
    processed_dataset_path = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"

    train_dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True
    )

    logger.info(f"Loaded {len(train_dataset)} training segments.")

    validation_dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False
    )

    logger.info(f"Loaded {len(validation_dataset)} validation segments.")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── CPC model ──
    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"]
    ).to(device)

    # ── Temporal ranking components ──
    # Projection head: maps (z_last ∥ c_last) → compact space for ranking loss.
    # Prevents the ranking objective from distorting the main representation.
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

    # Optimizer covers both CPC encoder/AR and the temporal projection head
    cpc_optimizer = optim.AdamW(
        list(cpc.parameters()) + list(temporal_proj.parameters()),
        lr=config["cpc_lr"],
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        cpc_optimizer, T_max=cpc_epochs, eta_min=config["cpc_lr"] * 0.1
    )

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1 — CPC + Temporal Ranking Joint Training
    # ══════════════════════════════════════════════════════════════════════
    pbar = tqdm(total=cpc_epochs, desc="Training CPC + Temporal Ranking")

    for cpc_epoch in range(cpc_epochs):
        cpc.train()
        temporal_proj.train()

        total_samples = 0
        total_cpc_loss = 0.0
        total_temporal_loss = 0.0
        total_accuracy = 0.0
        total_sim_gap = 0.0
        n_batches = 0

        # Warmup: linearly ramp temporal loss weight over first N epochs
        warmup = max(1, config["temporal_warmup_epochs"])
        temporal_alpha = config["temporal_loss_weight"] * min(1.0, (cpc_epoch + 1) / warmup)

        for rr, label, tte in train_data_loader:
            rr  = rr.to(device, non_blocking=True)
            tte = tte.to(device, non_blocking=True).float()
            tte = torch.nan_to_num(tte, nan=0.0).clamp(min=0)

            cpc_optimizer.zero_grad()

            # CPC forward: InfoNCE contrastive prediction loss
            loss_cpc, accuracy, z_seq, c_seq = cpc(rr)

            # Temporal ranking loss on last-timestep representations
            z_last = z_seq[:, -1, :]                        # [B, latent_dim]
            c_last = c_seq[:, -1, :]                        # [B, context_dim]
            h = torch.cat([z_last, c_last], dim=-1)         # [B, latent+context]
            h_proj = temporal_proj(h)                        # [B, proj_dim]

            loss_temporal, t_metrics = temporal_loss_fn(h_proj, tte)

            # Combined objective
            loss = loss_cpc + temporal_alpha * loss_temporal

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cpc.parameters()) + list(temporal_proj.parameters()),
                max_norm=1.0,
            )
            cpc_optimizer.step()

            total_cpc_loss      += loss_cpc.item()
            total_temporal_loss += loss_temporal.item()
            total_accuracy      += accuracy * rr.size(0)
            total_samples       += rr.size(0)
            total_sim_gap       += t_metrics.get("sim_gap", 0.0)
            n_batches           += 1

        scheduler.step()

        # ── Validation ──
        cpc.eval()
        temporal_proj.eval()

        val_cpc_loss = 0.0
        val_temporal_loss = 0.0
        val_accuracy = 0.0
        val_samples = 0
        val_sim_gap = 0.0
        val_batches = 0

        embeddings_list = []
        context_list = []
        label_list = []
        tte_list = []

        with torch.no_grad():
            for rr, label, tte in validation_data_loader:
                rr  = rr.to(device, non_blocking=True)
                tte_gpu = tte.to(device, non_blocking=True).float()
                tte_gpu = torch.nan_to_num(tte_gpu, nan=0.0).clamp(min=0)

                loss_cpc, accuracy, z_seq, c_seq = cpc(rr)

                z_last = z_seq[:, -1, :]
                c_last = c_seq[:, -1, :]
                h = torch.cat([z_last, c_last], dim=-1)
                h_proj = temporal_proj(h)

                loss_temporal, t_metrics = temporal_loss_fn(h_proj, tte_gpu)

                val_cpc_loss      += loss_cpc.item()
                val_temporal_loss += loss_temporal.item()
                val_accuracy      += accuracy * rr.size(0)
                val_samples       += rr.size(0)
                val_sim_gap       += t_metrics.get("sim_gap", 0.0)
                val_batches       += 1

                embeddings_list.append(z_seq[:, -1, :].cpu().numpy())
                context_list.append(c_seq[:, -1, :].cpu().numpy())
                label_list.extend(label.cpu().numpy().tolist())
                tte_list.extend(tte.cpu().numpy().tolist())

        # ── PCA Visualization: 2×2 grid (label + TTE coloring) ──
        embeddings_all = np.concatenate(embeddings_list, axis=0)
        context_all    = np.concatenate(context_list, axis=0)
        tte_all        = np.nan_to_num(np.array(tte_list), nan=0.0).clip(min=0)

        pca_emb = PCA(n_components=2, random_state=42)
        pca_ctx = PCA(n_components=2, random_state=42)
        embeddings_pca = pca_emb.fit_transform(embeddings_all)
        context_pca    = pca_ctx.fit_transform(context_all)

        color_map = {-1: 'blue', 0: 'green', 1: 'red'}
        label_colors = [color_map[l] for l in label_list]

        fig = plt.figure(figsize=(18, 16))
        fig.suptitle(f"Epoch {cpc_epoch+1}  |  α_temporal = {temporal_alpha:.2f}", fontsize=14)
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)

        # Top-left: embeddings by label
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                    c=label_colors, alpha=0.6, s=8)
        ax1.set_title("Latent Embeddings — by label")
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

        # Top-right: embeddings by TTE
        ax2 = fig.add_subplot(gs[0, 1])
        sc2 = ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                          c=tte_all, cmap='RdYlGn', alpha=0.6, s=8)
        ax2.set_title("Latent Embeddings — by TTE")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        plt.colorbar(sc2, ax=ax2, label='TTE (s)', shrink=0.8)

        # Bottom-left: context by label
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(context_pca[:, 0], context_pca[:, 1],
                    c=label_colors, alpha=0.6, s=8)
        ax3.set_title("Context Vectors — by label")
        ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")

        # Bottom-right: context by TTE
        ax4 = fig.add_subplot(gs[1, 1])
        sc4 = ax4.scatter(context_pca[:, 0], context_pca[:, 1],
                          c=tte_all, cmap='RdYlGn', alpha=0.6, s=8)
        ax4.set_title("Context Vectors — by TTE")
        ax4.set_xlabel("PC1"); ax4.set_ylabel("PC2")
        plt.colorbar(sc4, ax=ax4, label='TTE (s)', shrink=0.8)

        legend_handles = [
            mpatches.Patch(color=color_map[k], label=lbl)
            for k, lbl in zip([-1, 0, 1], ['SR', 'Mixed', 'AFIB'])
        ]
        fig.legend(handles=legend_handles, loc='upper right', title='Label')

        # ── Spearman ρ (pairwise sim vs TTE proximity on subsample) ──
        spearman_rho = compute_validation_spearman(
            embeddings_all, context_all, tte_all
        )

        # ── Logging ──
        avg_train_cpc      = total_cpc_loss / max(n_batches, 1)
        avg_train_temporal  = total_temporal_loss / max(n_batches, 1)
        avg_train_acc       = (total_accuracy / total_samples) if total_samples > 0 else 0
        avg_train_sim_gap   = total_sim_gap / max(n_batches, 1)

        avg_val_cpc         = val_cpc_loss / max(val_batches, 1)
        avg_val_temporal    = val_temporal_loss / max(val_batches, 1)
        avg_val_acc         = (val_accuracy / val_samples) if val_samples > 0 else 0
        avg_val_sim_gap     = val_sim_gap / max(val_batches, 1)

        pbar.update(1)
        pbar.set_description(f"Epoch {cpc_epoch+1}")
        pbar.write(
            f"Epoch {cpc_epoch+1} | "
            f"CPC loss {avg_train_cpc:.4f} / {avg_val_cpc:.4f} | "
            f"Temporal loss {avg_train_temporal:.4f} / {avg_val_temporal:.4f} | "
            f"Acc {avg_train_acc:.2f}% / {avg_val_acc:.2f}% | "
            f"SimGap {avg_train_sim_gap:.4f} / {avg_val_sim_gap:.4f} | "
            f"Spearman ρ {spearman_rho:.4f}"
        )
        run.log({
            "cpc_epoch": cpc_epoch+1,
            "cpc_lr": cpc_optimizer.param_groups[0]['lr'],
            "temporal_alpha": temporal_alpha,
            # Train
            "cpc_train_loss": avg_train_cpc,
            "temporal_train_loss": avg_train_temporal,
            "cpc_train_accuracy": avg_train_acc,
            "temporal_train_sim_gap": avg_train_sim_gap,
            # Validation
            "cpc_validation_loss": avg_val_cpc,
            "temporal_validation_loss": avg_val_temporal,
            "cpc_validation_accuracy": avg_val_acc,
            "temporal_validation_sim_gap": avg_val_sim_gap,
            # Global temporal quality
            "spearman_rho": spearman_rho,
            "pca": wandb.Image(fig),
        })
        plt.close(fig)

    pbar.close()

    torch.save(cpc.state_dict(), "cpc_model.pth")
    torch.save(temporal_proj.state_dict(), "temporal_proj.pth")
    artifact = wandb.Artifact("cpc_model", type="model")
    artifact.add_file("cpc_model.pth")
    artifact.add_file("temporal_proj.pth")
    run.log_artifact(artifact)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — Cox Survival Head (frozen encoder)
    # ══════════════════════════════════════════════════════════════════════
    cox_train_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True
    )

    logger.info(f"Loaded {len(cox_train_dataset)} training segments.")

    cox_validation_dataset = CPCCoxDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False
    )

    logger.info(f"Loaded {len(cox_validation_dataset)} validation segments.")

    cox_train_data_loader = DataLoader(
        cox_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    cox_validation_data_loader = DataLoader(
        cox_validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
        )

    cox = CoxHead(
        context_dim=config['context_dim'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout']
    ).to(device)

    cox_optimizer = optim.AdamW(cox.parameters(), lr=config["cox_lr"], weight_decay=1e-2)
    loss_fn = DeepSurvLoss()

    cpc.eval()
    for param in cpc.parameters():
        param.requires_grad = False

    pbar = tqdm(total=cox_epochs, desc="Training Cox Model")
    for cox_epoch in range(cox_epochs):
        total_loss = 0
        validation_loss = 0
        train_risk = []
        train_time = []
        train_event = []
        cpc.eval() # Freeze CPC model
        cox.train() # Train Cox model
        for rr, label, time, event in cox_train_data_loader:

            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)

            cox_optimizer.zero_grad()

            with torch.no_grad():
                _, _, embeddings, context = cpc(rr)

            logits = cox(context[:, -1, :], embeddings[:, -1, :])
            loss = loss_fn(logits, time, event)

            total_loss += loss.item()
            loss.backward()
            cox_optimizer.step()

            train_risk.append(logits.detach().cpu())
            train_time.append(time.cpu())
            train_event.append(event.cpu())

        actual_times = []
        predicted_times = []

        train_event = torch.cat(train_event)
        train_time = torch.cat(train_time)
        train_risk = torch.cat(train_risk)

        times, baseline_cumhuz = compute_baseline_hazard(
            train_risk,
            train_time,
            train_event
        )

        val_risk = []
        val_time = []
        val_event = []

        cpc.eval() # Freeze CPC model
        cox.eval() # Evaluate Cox model
        for rr, label, time, event in cox_validation_data_loader:

            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)

            actual_times.append(time.cpu().numpy())

            with torch.no_grad():
                _, _, embeddings, context = cpc(rr)
                logits = cox(context[:, -1, :], embeddings[:, -1, :])

                loss = loss_fn(logits, time, event)
                validation_loss += loss.item()

                val_risk.append(logits.detach().cpu())
                val_time.append(time.cpu())
                val_event.append(event.cpu())

                local_pred_times = []

                for r in logits.cpu():
                    predicted_time = predict_median_survival(r, times, baseline_cumhuz)
                    local_pred_times.append(predicted_time.cpu().item())
                predicted_times.append(local_pred_times)

        val_time = torch.cat(val_time)
        val_event = torch.cat(val_event)
        val_risk = torch.cat(val_risk)
        predictions = torch.tensor([item for sublist in predicted_times for item in sublist])
        concordance = concordance_index(predictions.unsqueeze(1), val_time.unsqueeze(1))

        np_actual_times = np.concatenate(actual_times, axis=0)
        np_predicted_times = np.concatenate(predicted_times, axis=0)

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f"Cox Model - Epoch {cox_epoch+1}", fontsize=16)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.scatter(np_actual_times, np_predicted_times, alpha=0.5)
        ax1.plot([0, np.max(np_actual_times)], [0, np.max(np_actual_times)], 'r--')
        ax1.set_xlabel("Actual Time to Event (hours)")
        ax1.set_ylabel("Predicted Time to Event (hours)")
        ax1.set_title(f"Actual vs Predicted Time to Event - Epoch {cox_epoch+1}")
        ax1.set_xlim(0, np.max(np_actual_times)*1.1)
        ax1.set_ylim(0, np.max(np_actual_times)*1.1)
        ax1.grid()

        ax2 = fig.add_subplot(gs[0, 1])
        hb = ax2.hexbin(np_actual_times, np_predicted_times, gridsize=50, cmap='Blues', mincnt=1)
        ax2.set_xlabel("Actual Time to Event (hours)")
        ax2.set_ylabel("Predicted Time to Event (hours)")
        ax2.set_title(f"Prediction Density - Epoch {cox_epoch+1}")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(val_risk.cpu().numpy(), bins=50, color='blue', alpha=0.7)
        ax3.set_xlabel("Predicted Risk Score")
        ax3.set_title(f"Predicted Risk Score Distribution - Epoch {cox_epoch+1}")
        ax3.grid()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        run.log({
            "cox_epoch": cox_epoch+1,
            "cox_training_loss": total_loss / len(cox_train_data_loader),
            "cox_validation_loss": validation_loss / len(cox_validation_data_loader),
            "concordance_index": concordance.item(),
            "cox_scatter_plot": wandb.Image(fig)
        })
        plt.close(fig)
        pbar.update(1)
        pbar.set_description(f"Epoch {cox_epoch+1}")
        pbar.write(f"Epoch {cox_epoch+1}, Cox Model Loss: {total_loss / len(cox_train_data_loader):.4f}, Validation Loss: {validation_loss / len(cox_validation_data_loader):.4f} Concordance Index: {concordance.item():.4f}")

    pbar.close()
    torch.save(cox.state_dict(), "cox_model.pth")
    artifact = wandb.Artifact("cox_model", type="model")
    artifact.add_file("cox_model.pth")
    run.log_artifact(artifact)

except Exception as e:
    if 'logger' in locals():
        logger.error(f"An error occurred: {e}", exc_info=True)
    else:
        print(f"An error occurred: {e}")
    if 'pbar' in locals():
        pbar.close()
finally:
    if 'run' in locals():
        run.finish()
