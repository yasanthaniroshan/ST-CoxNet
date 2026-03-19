import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import wandb

from Model.CPC import CPC
from Model.ClassificationHead import AFibClassificationHead
from Utils.Dataset.CPCDataset import CPCDataset
from Utils.Dataset.CPCClassificationDataset import CPCClassificationDataset

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)


def extract_features(cpc: CPC, rr: torch.Tensor):
    b, t, w = rr.shape
    z_seq = cpc.encoder(rr.view(-1, w)).view(b, t, -1)
    z_seq = F.normalize(z_seq, dim=-1)
    c_seq = cpc.ar_block(z_seq)
    return z_seq, c_seq


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean())
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, prec, rec, f1, auc, cm


def plot_cpc_pca(embeddings, contexts, labels, epoch):
    pca_e = PCA(n_components=2, random_state=42)
    pca_c = PCA(n_components=2, random_state=42)
    e2d = pca_e.fit_transform(embeddings)
    c2d = pca_c.fit_transform(contexts)

    cmap = {-1: "blue", 0: "green", 1: "red"}
    names = {-1: "SR", 0: "Mixed", 1: "AFIB"}
    colors = [cmap[int(l)] for l in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.scatter(e2d[:, 0], e2d[:, 1], c=colors, alpha=0.6, s=8)
    ax1.set_title(f"Latent Embeddings PCA (epoch {epoch})")
    ax2.scatter(c2d[:, 0], c2d[:, 1], c=colors, alpha=0.6, s=8)
    ax2.set_title(f"Context PCA (epoch {epoch})")
    handles = [mpatches.Patch(color=cmap[k], label=names[k]) for k in [-1, 0, 1]]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_cls_epoch(cm, y_prob, y_true, epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["SR", "AFIB"]); ax1.set_yticklabels(["SR", "AFIB"])
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title(f"Confusion Matrix (epoch {epoch})")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    for label, name in [(0, "SR"), (1, "AFIB")]:
        vals = y_prob[y_true == label]
        if len(vals) > 0:
            ax2.hist(vals, bins=30, alpha=0.6, label=f"{name} (true={label})")
    ax2.set_title(f"P(AFIB) distribution (epoch {epoch})")
    ax2.set_xlabel("P(AFIB)"); ax2.set_ylabel("Count")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_curves(epochs, train_vals, val_vals, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_vals, label="Train")
    ax.plot(epochs, val_vals, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    return fig


try:
    config = {
        "afib_length": 60 * 60,
        "sr_length": int(1.5 * 60 * 60),
        "number_of_windows_in_segment": 10,
        "stride": 20,
        "window_size": 100,
        "validation_split": 0.15,
        "dropout": 0.1,
        "temperature": 0.07,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 3,
        # --- Stage 1: CPC (cross-batch InfoNCE) ---
        "cpc_epochs": 50,
        "cpc_batch_size": 512,
        "cpc_lr": 1e-3,
        "cpc_patience": 8,
        # --- Stage 2: Classification (2-phase: head-only then fine-tune) ---
        "cls_epochs": 40,
        "cls_batch_size": 256,
        "cls_lr": 3e-4,
        "cls_backbone_lr_scale": 0.01,
        "cls_head_warmup_epochs": 8,
        "cls_patience": 10,
    }

    run = wandb.init(entity="eml-labs", project="CPC-AFib-Classification-v8", config=config)

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)

    processed_dataset_path = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # STAGE 1: CPC self-supervised pretraining (all segments: SR+Mixed+AFIB)
    # =====================================================================
    logger.info("=== STAGE 1: CPC Pretraining ===")

    cpc_train_ds = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        validation_split=config["validation_split"], train=True,
    )
    cpc_val_ds = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        validation_split=config["validation_split"], train=False,
    )
    logger.info(f"CPC train: {len(cpc_train_ds)}, val: {len(cpc_val_ds)}")

    cpc_train_loader = DataLoader(
        cpc_train_ds, batch_size=config["cpc_batch_size"], shuffle=True,
        drop_last=True, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    cpc_val_loader = DataLoader(
        cpc_val_ds, batch_size=config["cpc_batch_size"], shuffle=False,
        drop_last=False, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"],
        window_size=config["window_size"],
    ).to(device)

    cpc_opt = optim.AdamW(cpc.parameters(), lr=config["cpc_lr"], weight_decay=1e-2)
    cpc_sched = optim.lr_scheduler.CosineAnnealingLR(
        cpc_opt, T_max=config["cpc_epochs"], eta_min=config["cpc_lr"] * 0.1,
    )

    cpc_train_losses, cpc_val_losses = [], []
    cpc_train_accs, cpc_val_accs = [], []
    best_cpc_val_acc = 0.0
    cpc_patience_ctr = 0

    pbar = tqdm(total=config["cpc_epochs"], desc="Stage 1: CPC")
    for ep in range(config["cpc_epochs"]):
        cpc.train()
        t_loss, t_acc, t_n = 0.0, 0.0, 0

        for rr, _, _ in cpc_train_loader:
            rr = rr.to(device, non_blocking=True)
            cpc_opt.zero_grad()
            loss, acc, _, _ = cpc(rr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cpc.parameters(), 1.0)
            cpc_opt.step()
            t_loss += loss.item()
            t_acc += acc * rr.size(0)
            t_n += rr.size(0)
        cpc_sched.step()

        cpc.eval()
        v_loss, v_acc, v_n = 0.0, 0.0, 0
        emb_list, ctx_list, lbl_list = [], [], []

        for rr, lbl, _ in cpc_val_loader:
            rr = rr.to(device, non_blocking=True)
            lbl_list.extend(lbl.numpy().tolist())
            with torch.no_grad():
                loss, acc, emb, ctx = cpc(rr)
            v_loss += loss.item()
            v_acc += acc * rr.size(0)
            v_n += rr.size(0)
            emb_list.append(emb[:, -1, :].cpu().numpy())
            ctx_list.append(ctx[:, -1, :].cpu().numpy())

        tl = t_loss / max(len(cpc_train_loader), 1)
        ta = t_acc / max(t_n, 1)
        vl = v_loss / max(len(cpc_val_loader), 1)
        va = v_acc / max(v_n, 1)
        cpc_train_losses.append(tl); cpc_val_losses.append(vl)
        cpc_train_accs.append(ta); cpc_val_accs.append(va)

        if va > best_cpc_val_acc:
            best_cpc_val_acc = va
            cpc_patience_ctr = 0
            torch.save(cpc.state_dict(), "cpc_model_v8.pth")
        else:
            cpc_patience_ctr += 1

        pca_fig = None
        if emb_list:
            pca_fig = plot_cpc_pca(
                np.concatenate(emb_list), np.concatenate(ctx_list),
                np.array(lbl_list), ep + 1,
            )

        run.log({
            "cpc_epoch": ep + 1, "cpc_lr": cpc_opt.param_groups[0]["lr"],
            "cpc_train_loss": tl, "cpc_train_acc": ta,
            "cpc_val_loss": vl, "cpc_val_acc": va,
            "cpc_pca": wandb.Image(pca_fig) if pca_fig else None,
        })
        if pca_fig:
            plt.close(pca_fig)

        pbar.update(1)
        pbar.write(
            f"CPC {ep+1}: train_loss={tl:.4f} train_acc={ta:.2f} "
            f"val_loss={vl:.4f} val_acc={va:.2f} "
            f"[patience {cpc_patience_ctr}/{config['cpc_patience']}]"
        )

        if cpc_patience_ctr >= config["cpc_patience"]:
            logger.info(f"CPC early stop at epoch {ep+1}. Best val acc: {best_cpc_val_acc:.2f}")
            break
    pbar.close()

    cpc.load_state_dict(torch.load("cpc_model_v8.pth", weights_only=True))
    art = wandb.Artifact("cpc_model_v8", type="model")
    art.add_file("cpc_model_v8.pth")
    run.log_artifact(art)

    actual_cpc_epochs = len(cpc_train_accs)
    cpc_fig = plot_curves(
        list(range(1, actual_cpc_epochs + 1)),
        cpc_train_accs, cpc_val_accs,
        "Accuracy", "CPC Pretraining Accuracy",
        "plots/cpc_v8_accuracy.png",
    )
    run.log({"cpc_accuracy_curves": wandb.Image(cpc_fig)})
    plt.close(cpc_fig)

    # =====================================================================
    # STAGE 2: AFib vs SR classification
    #   Phase A: train head only (backbone frozen) for warmup
    #   Phase B: unfreeze backbone and fine-tune all with differential LR
    # =====================================================================
    logger.info("=== STAGE 2: AFib vs SR Classification ===")

    cls_train_ds = CPCClassificationDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        validation_split=config["validation_split"], train=True,
    )
    cls_val_ds = CPCClassificationDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=config["afib_length"], sr_length=config["sr_length"],
        number_of_windows_in_segment=config["number_of_windows_in_segment"],
        stride=config["stride"], window_size=config["window_size"],
        validation_split=config["validation_split"], train=False,
    )
    logger.info(f"CLS train: {len(cls_train_ds)}, val: {len(cls_val_ds)}")

    cls_train_loader = DataLoader(
        cls_train_ds, batch_size=config["cls_batch_size"], shuffle=True,
        drop_last=True, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    cls_val_loader = DataLoader(
        cls_val_ds, batch_size=config["cls_batch_size"], shuffle=False,
        drop_last=False, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    cls_head = AFibClassificationHead(
        context_dim=config["context_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    ).to(device)

    train_labels_np = cls_train_ds.labels.numpy()
    pos_w = float((train_labels_np == 0).sum()) / max(float((train_labels_np == 1).sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=device),
    )

    warmup_epochs = config["cls_head_warmup_epochs"]
    backbone_lr = config["cls_lr"] * config["cls_backbone_lr_scale"]

    for p in cpc.encoder.parameters():
        p.requires_grad = False
    for p in cpc.ar_block.parameters():
        p.requires_grad = False

    cls_opt = optim.AdamW(cls_head.parameters(), lr=config["cls_lr"], weight_decay=1e-2)
    cls_sched = optim.lr_scheduler.CosineAnnealingLR(
        cls_opt, T_max=config["cls_epochs"], eta_min=config["cls_lr"] * 0.01,
    )

    cls_t_losses, cls_v_losses = [], []
    cls_t_f1s, cls_v_f1s = [], []
    cls_t_accs, cls_v_accs = [], []
    best_val_f1 = 0.0
    patience_counter = 0

    pbar = tqdm(total=config["cls_epochs"], desc="Stage 2: Classification")
    for ep in range(config["cls_epochs"]):

        if ep == warmup_epochs:
            logger.info("Unfreezing backbone for fine-tuning")
            for p in cpc.encoder.parameters():
                p.requires_grad = True
            for p in cpc.ar_block.parameters():
                p.requires_grad = True
            cls_opt = optim.AdamW([
                {"params": cls_head.parameters(), "lr": config["cls_lr"]},
                {"params": cpc.encoder.parameters(), "lr": backbone_lr},
                {"params": cpc.ar_block.parameters(), "lr": backbone_lr},
            ], weight_decay=1e-2)
            cls_sched = optim.lr_scheduler.CosineAnnealingLR(
                cls_opt,
                T_max=config["cls_epochs"] - warmup_epochs,
                eta_min=backbone_lr * 0.01,
            )
            patience_counter = 0

        backbone_frozen = ep < warmup_epochs
        phase_tag = "head-only" if backbone_frozen else "fine-tune"

        cls_head.train()
        if backbone_frozen:
            cpc.eval()
        else:
            cpc.train()

        e_loss = 0.0
        t_tgt, t_prob = [], []

        for rr, label in cls_train_loader:
            rr = rr.to(device, non_blocking=True)
            tgt = label.to(device, non_blocking=True).float()

            cls_opt.zero_grad()
            if backbone_frozen:
                with torch.no_grad():
                    z, c = extract_features(cpc, rr)
            else:
                z, c = extract_features(cpc, rr)

            logits = cls_head(c[:, -1, :], z[:, -1, :])
            loss = criterion(logits, tgt)
            loss.backward()
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(cls_head.parameters(), max_norm)
            if not backbone_frozen:
                torch.nn.utils.clip_grad_norm_(cpc.encoder.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(cpc.ar_block.parameters(), max_norm)
            cls_opt.step()

            e_loss += loss.item()
            t_tgt.append(tgt.detach().cpu().numpy())
            t_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

        cls_sched.step()

        t_tgt = np.concatenate(t_tgt); t_prob = np.concatenate(t_prob)
        t_acc, t_prec, t_rec, t_f1, t_auc, _ = compute_binary_metrics(t_tgt, t_prob)
        tl = e_loss / max(len(cls_train_loader), 1)

        cls_head.eval(); cpc.eval()
        v_loss = 0.0
        v_tgt, v_prob = [], []
        with torch.no_grad():
            for rr, label in cls_val_loader:
                rr = rr.to(device, non_blocking=True)
                tgt = label.to(device, non_blocking=True).float()
                z, c = extract_features(cpc, rr)
                logits = cls_head(c[:, -1, :], z[:, -1, :])
                loss = criterion(logits, tgt)
                v_loss += loss.item()
                v_tgt.append(tgt.cpu().numpy())
                v_prob.append(torch.sigmoid(logits).cpu().numpy())

        v_tgt = np.concatenate(v_tgt); v_prob = np.concatenate(v_prob)
        v_acc, v_prec, v_rec, v_f1, v_auc, cm = compute_binary_metrics(v_tgt, v_prob)
        vl = v_loss / max(len(cls_val_loader), 1)

        cls_t_losses.append(tl); cls_v_losses.append(vl)
        cls_t_f1s.append(t_f1); cls_v_f1s.append(v_f1)
        cls_t_accs.append(t_acc); cls_v_accs.append(v_acc)

        cls_fig = plot_cls_epoch(cm, v_prob, v_tgt, ep + 1)
        run.log({
            "cls_epoch": ep + 1, "cls_phase": phase_tag,
            "cls_lr": cls_opt.param_groups[0]["lr"],
            "cls_train_loss": tl, "cls_val_loss": vl,
            "cls_train_acc": t_acc, "cls_val_acc": v_acc,
            "cls_train_f1": t_f1, "cls_val_f1": v_f1,
            "cls_train_prec": t_prec, "cls_val_prec": v_prec,
            "cls_train_rec": t_rec, "cls_val_rec": v_rec,
            "cls_train_auc": t_auc, "cls_val_auc": v_auc,
            "cls_epoch_plot": wandb.Image(cls_fig),
        })
        plt.close(cls_fig)

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            patience_counter = 0
            torch.save(cls_head.state_dict(), "cls_head_best_v8.pth")
            torch.save(cpc.state_dict(), "cpc_finetuned_best_v8.pth")
        else:
            patience_counter += 1

        pbar.update(1)
        pbar.write(
            f"CLS {ep+1} [{phase_tag}]: train_loss={tl:.4f} train_f1={t_f1:.4f} | "
            f"val_loss={vl:.4f} val_f1={v_f1:.4f} val_acc={v_acc:.4f} val_auc={v_auc:.4f} "
            f"[patience {patience_counter}/{config['cls_patience']}]"
        )

        if patience_counter >= config["cls_patience"]:
            logger.info(f"Early stopping at epoch {ep + 1}. Best val F1: {best_val_f1:.4f}")
            break
    pbar.close()

    art = wandb.Artifact("cls_head_best_v8", type="model")
    art.add_file("cls_head_best_v8.pth")
    run.log_artifact(art)

    ep_range = list(range(1, len(cls_t_f1s) + 1))
    f1_fig = plot_curves(ep_range, cls_t_f1s, cls_v_f1s, "F1", "Classification F1", "plots/cls_v8_f1.png")
    run.log({"cls_f1_curves": wandb.Image(f1_fig)})
    plt.close(f1_fig)
    acc_fig = plot_curves(ep_range, cls_t_accs, cls_v_accs, "Accuracy", "Classification Accuracy", "plots/cls_v8_acc.png")
    run.log({"cls_acc_curves": wandb.Image(acc_fig)})
    plt.close(acc_fig)
    loss_fig = plot_curves(ep_range, cls_t_losses, cls_v_losses, "Loss", "Classification Loss", "plots/cls_v8_loss.png")
    run.log({"cls_loss_curves": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    if "run" in locals():
        run.finish()
