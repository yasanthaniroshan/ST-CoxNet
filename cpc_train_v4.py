import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
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


def extract_backbone_features(cpc: CPC, rr: torch.Tensor):
    b, t, window_size = rr.shape
    z_seq = cpc.encoder(rr.view(-1, window_size)).view(b, t, -1)
    z_seq = F.normalize(z_seq, dim=-1)
    c_seq = cpc.ar_block(z_seq)
    return z_seq, c_seq


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean()) if len(y_true) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, precision, recall, f1, auc, cm


def plot_cpc_pca(embeddings_all, context_all, labels_all, epoch):
    pca_emb = PCA(n_components=2, random_state=42)
    pca_ctx = PCA(n_components=2, random_state=42)
    embeddings_pca = pca_emb.fit_transform(embeddings_all)
    context_pca = pca_ctx.fit_transform(context_all)

    color_map = {-1: "blue", 0: "green", 1: "red"}
    class_name = {-1: "SR", 0: "Mixed", 1: "AFIB"}
    colors = [color_map[int(l)] for l in labels_all]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=colors, alpha=0.7, s=10)
    ax1.set_title(f"CPC Latent Embeddings PCA (epoch {epoch})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    ax2.scatter(context_pca[:, 0], context_pca[:, 1], c=colors, alpha=0.7, s=10)
    ax2.set_title(f"CPC Context PCA (epoch {epoch})")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color_map[k], label=class_name[k]) for k in [-1, 0, 1]]
    fig.legend(handles=legend_handles, loc="upper right", title="Classes")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_classification_epoch(cm, y_prob, y_true, epoch):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["SR", "AFIB"])
    ax1.set_yticklabels(["SR", "AFIB"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"Confusion Matrix (epoch {epoch})")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    sr_probs = y_prob[y_true == 0]
    afib_probs = y_prob[y_true == 1]
    if len(sr_probs) > 0:
        ax2.hist(sr_probs, bins=30, alpha=0.6, label="SR (true=0)")
    if len(afib_probs) > 0:
        ax2.hist(afib_probs, bins=30, alpha=0.6, label="AFIB (true=1)")
    ax2.set_title(f"Predicted AFIB Probability (epoch {epoch})")
    ax2.set_xlabel("P(AFIB)")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_curves(
    epochs,
    train_loss,
    val_loss,
    train_metric,
    val_metric,
    metric_name,
    title,
    save_path,
):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Validation Loss")
    ax1.set_title(f"{title} Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_metric, label=f"Train {metric_name}")
    ax2.plot(epochs, val_metric, label=f"Validation {metric_name}")
    ax2.set_title(f"{title} {metric_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_name)
    ax2.grid(alpha=0.3)
    ax2.legend()

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
        "dropout": 0.2,
        "temperature": 0.2,
        "latent_dim": 64,
        "context_dim": 128,
        "number_of_prediction_steps": 6,
        "batch_size": 512,
        "cpc_epochs": 20,
        "classification_epochs": 20,
        "cpc_lr": 1e-3,
        "classification_lr": 1e-3,
        "freeze_backbone_for_classification": True,
    }

    run = wandb.init(
        entity="eml-labs",
        project="CPC-AFib-Classification",
        config=config,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("plots", exist_ok=True)

    afib_length = config["afib_length"]
    sr_length = config["sr_length"]
    number_of_windows_in_segment = config["number_of_windows_in_segment"]
    stride = config["stride"]
    window_size = config["window_size"]
    validation_split = config["validation_split"]
    cpc_epochs = config["cpc_epochs"]
    classification_epochs = config["classification_epochs"]
    processed_dataset_path = "/home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"

    cpc_train_dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True,
    )
    cpc_validation_dataset = CPCDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False,
    )

    logger.info(f"Loaded CPC train segments: {len(cpc_train_dataset)}")
    logger.info(f"Loaded CPC validation segments: {len(cpc_validation_dataset)}")

    cpc_train_loader = DataLoader(
        cpc_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    cpc_validation_loader = DataLoader(
        cpc_validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    cpc_optimizer = optim.AdamW(cpc.parameters(), lr=config["cpc_lr"], weight_decay=1e-2)
    cpc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        cpc_optimizer, T_max=cpc_epochs, eta_min=config["cpc_lr"] * 0.1
    )

    cpc_train_loss_hist, cpc_val_loss_hist = [], []
    cpc_train_acc_hist, cpc_val_acc_hist = [], []

    pbar = tqdm(total=cpc_epochs, desc="Stage 1/2: CPC pretraining")
    for cpc_epoch in range(cpc_epochs):
        cpc.train()
        train_total_loss = 0.0
        train_total_acc = 0.0
        train_total_samples = 0

        for rr, _, _ in cpc_train_loader:
            rr = rr.to(device, non_blocking=True)
            cpc_optimizer.zero_grad()
            loss, accuracy, _, _ = cpc(rr)
            loss.backward()
            cpc_optimizer.step()
            train_total_loss += loss.item()
            train_total_acc += accuracy * rr.size(0)
            train_total_samples += rr.size(0)

        cpc_scheduler.step()

        cpc.eval()
        val_total_loss = 0.0
        val_total_acc = 0.0
        val_total_samples = 0
        embeddings_list, context_list, label_list = [], [], []

        for rr, label, _ in cpc_validation_loader:
            rr = rr.to(device, non_blocking=True)
            label_list.extend(label.cpu().numpy().tolist())
            with torch.no_grad():
                loss, accuracy, embeddings, context = cpc(rr)
            val_total_loss += loss.item()
            val_total_acc += accuracy * rr.size(0)
            val_total_samples += rr.size(0)
            embeddings_list.append(embeddings[:, -1, :].cpu().numpy())
            context_list.append(context[:, -1, :].cpu().numpy())

        train_loss = train_total_loss / max(len(cpc_train_loader), 1)
        train_acc = train_total_acc / max(train_total_samples, 1)
        val_loss = val_total_loss / max(len(cpc_validation_loader), 1)
        val_acc = val_total_acc / max(val_total_samples, 1)

        cpc_train_loss_hist.append(train_loss)
        cpc_val_loss_hist.append(val_loss)
        cpc_train_acc_hist.append(train_acc)
        cpc_val_acc_hist.append(val_acc)

        pca_fig = None
        if len(embeddings_list) > 0:
            embeddings_all = np.concatenate(embeddings_list, axis=0)
            context_all = np.concatenate(context_list, axis=0)
            pca_fig = plot_cpc_pca(embeddings_all, context_all, np.array(label_list), cpc_epoch + 1)

        run.log(
            {
                "cpc_epoch": cpc_epoch + 1,
                "cpc_lr": cpc_optimizer.param_groups[0]["lr"],
                "cpc_train_loss": train_loss,
                "cpc_train_accuracy": train_acc,
                "cpc_validation_loss": val_loss,
                "cpc_validation_accuracy": val_acc,
                "cpc_pca": wandb.Image(pca_fig) if pca_fig is not None else None,
            }
        )

        if pca_fig is not None:
            plt.close(pca_fig)

        pbar.update(1)
        pbar.set_description(f"CPC epoch {cpc_epoch + 1}")
        pbar.write(
            f"CPC Epoch {cpc_epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}"
        )
    pbar.close()

    torch.save(cpc.state_dict(), "cpc_model.pth")
    cpc_artifact = wandb.Artifact("cpc_model", type="model")
    cpc_artifact.add_file("cpc_model.pth")
    run.log_artifact(cpc_artifact)

    cpc_curve_fig = plot_training_curves(
        epochs=list(range(1, cpc_epochs + 1)),
        train_loss=cpc_train_loss_hist,
        val_loss=cpc_val_loss_hist,
        train_metric=cpc_train_acc_hist,
        val_metric=cpc_val_acc_hist,
        metric_name="Accuracy",
        title="CPC",
        save_path="plots/cpc_training_curves.png",
    )
    run.log({"cpc_training_curves": wandb.Image(cpc_curve_fig)})
    plt.close(cpc_curve_fig)

    cls_train_dataset = CPCClassificationDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=True,
    )
    cls_validation_dataset = CPCClassificationDataset(
        processed_dataset_path=processed_dataset_path,
        afib_length=afib_length,
        sr_length=sr_length,
        number_of_windows_in_segment=number_of_windows_in_segment,
        stride=stride,
        window_size=window_size,
        validation_split=validation_split,
        train=False,
    )
    logger.info(f"Loaded classification train segments: {len(cls_train_dataset)}")
    logger.info(f"Loaded classification validation segments: {len(cls_validation_dataset)}")

    cls_train_loader = DataLoader(
        cls_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    cls_validation_loader = DataLoader(
        cls_validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    classifier_head = AFibClassificationHead(
        context_dim=config["context_dim"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    ).to(device)

    freeze_backbone = config["freeze_backbone_for_classification"]
    if freeze_backbone:
        for param in cpc.encoder.parameters():
            param.requires_grad = False
        for param in cpc.ar_block.parameters():
            param.requires_grad = False
        cpc.eval()
        backbone_params = []
    else:
        for param in cpc.encoder.parameters():
            param.requires_grad = True
        for param in cpc.ar_block.parameters():
            param.requires_grad = True
        backbone_params = list(cpc.encoder.parameters()) + list(cpc.ar_block.parameters())

    train_labels_np = cls_train_dataset.labels.numpy()
    pos_count = int((train_labels_np == 1).sum())
    neg_count = int((train_labels_np == 0).sum())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    cls_optimizer = optim.AdamW(
        list(classifier_head.parameters()) + backbone_params,
        lr=config["classification_lr"],
        weight_decay=1e-2,
    )
    cls_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        cls_optimizer, T_max=classification_epochs, eta_min=config["classification_lr"] * 0.1
    )

    cls_train_loss_hist, cls_val_loss_hist = [], []
    cls_train_acc_hist, cls_val_acc_hist = [], []
    cls_train_f1_hist, cls_val_f1_hist = [], []

    pbar = tqdm(total=classification_epochs, desc="Stage 2/2: AFib classification")
    for cls_epoch in range(classification_epochs):
        classifier_head.train()
        if freeze_backbone:
            cpc.eval()
        else:
            cpc.train()

        epoch_train_loss = 0.0
        train_targets, train_probs = [], []

        for rr, label in cls_train_loader:
            rr = rr.to(device, non_blocking=True)
            targets = label.to(device, non_blocking=True).float()

            cls_optimizer.zero_grad()
            if freeze_backbone:
                with torch.no_grad():
                    z_seq, c_seq = extract_backbone_features(cpc, rr)
            else:
                z_seq, c_seq = extract_backbone_features(cpc, rr)

            logits = classifier_head(c_seq[:, -1, :], z_seq[:, -1, :])
            loss = criterion(logits, targets)
            loss.backward()
            cls_optimizer.step()

            epoch_train_loss += loss.item()
            train_targets.append(targets.detach().cpu().numpy())
            train_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

        cls_scheduler.step()

        train_targets_np = np.concatenate(train_targets, axis=0)
        train_probs_np = np.concatenate(train_probs, axis=0)
        train_acc, train_precision, train_recall, train_f1, train_auc, _ = compute_binary_metrics(
            train_targets_np, train_probs_np
        )
        train_loss = epoch_train_loss / max(len(cls_train_loader), 1)

        classifier_head.eval()
        cpc.eval()
        epoch_val_loss = 0.0
        val_targets, val_probs = [], []
        with torch.no_grad():
            for rr, label in cls_validation_loader:
                rr = rr.to(device, non_blocking=True)
                targets = label.to(device, non_blocking=True).float()
                z_seq, c_seq = extract_backbone_features(cpc, rr)
                logits = classifier_head(c_seq[:, -1, :], z_seq[:, -1, :])
                loss = criterion(logits, targets)
                epoch_val_loss += loss.item()
                val_targets.append(targets.cpu().numpy())
                val_probs.append(torch.sigmoid(logits).cpu().numpy())

        val_targets_np = np.concatenate(val_targets, axis=0)
        val_probs_np = np.concatenate(val_probs, axis=0)
        val_acc, val_precision, val_recall, val_f1, val_auc, cm = compute_binary_metrics(
            val_targets_np, val_probs_np
        )
        val_loss = epoch_val_loss / max(len(cls_validation_loader), 1)

        cls_train_loss_hist.append(train_loss)
        cls_val_loss_hist.append(val_loss)
        cls_train_acc_hist.append(train_acc)
        cls_val_acc_hist.append(val_acc)
        cls_train_f1_hist.append(train_f1)
        cls_val_f1_hist.append(val_f1)

        cls_fig = plot_classification_epoch(cm, val_probs_np, val_targets_np, cls_epoch + 1)
        run.log(
            {
                "classification_epoch": cls_epoch + 1,
                "classification_lr": cls_optimizer.param_groups[0]["lr"],
                "classification_train_loss": train_loss,
                "classification_validation_loss": val_loss,
                "classification_train_accuracy": train_acc,
                "classification_validation_accuracy": val_acc,
                "classification_train_precision": train_precision,
                "classification_validation_precision": val_precision,
                "classification_train_recall": train_recall,
                "classification_validation_recall": val_recall,
                "classification_train_f1": train_f1,
                "classification_validation_f1": val_f1,
                "classification_train_auc": train_auc,
                "classification_validation_auc": val_auc,
                "classification_epoch_plots": wandb.Image(cls_fig),
            }
        )
        plt.close(cls_fig)

        pbar.update(1)
        pbar.set_description(f"Classification epoch {cls_epoch + 1}")
        pbar.write(
            f"CLS Epoch {cls_epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )
    pbar.close()

    torch.save(classifier_head.state_dict(), "afib_classifier_head.pth")
    cls_artifact = wandb.Artifact("afib_classifier_head", type="model")
    cls_artifact.add_file("afib_classifier_head.pth")
    run.log_artifact(cls_artifact)

    cls_curve_fig = plot_training_curves(
        epochs=list(range(1, classification_epochs + 1)),
        train_loss=cls_train_loss_hist,
        val_loss=cls_val_loss_hist,
        train_metric=cls_train_f1_hist,
        val_metric=cls_val_f1_hist,
        metric_name="F1",
        title="AFib Classification",
        save_path="plots/classification_training_curves.png",
    )
    # run.log({"classification_training_curves": wandb.Image(cls_curve_fig)})
    # plt.close(cls_curve_fig)

    logger.info("Training completed successfully.")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    if "run" in locals():
        run.finish()