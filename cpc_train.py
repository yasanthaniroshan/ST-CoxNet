from Model.CPC import CPC
from Model.PredictionHead.TimePredictor import TimePredictor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Utils.Dataset.CPCDataset import CPCDataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.login()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def concordance_loss(pred, target):
    """
    pred, target: [B, 1] normalized time-to-event
    Penalizes pairs where the predicted ordering disagrees with actual ordering.
    """
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    
    # All pairwise differences
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)      # [B, B]
    target_diff = target.unsqueeze(0) - target.unsqueeze(1) # [B, B]
    
    # Mask: only consider pairs where target ordering is clear (not ties)
    mask = (target_diff.abs() > 1e-4)
    
    # Concordant if signs agree
    concordant = ((pred_diff * target_diff) > 0).float()
    
    # C-index as a loss (maximize concordance = minimize 1 - concordance)
    c_index = concordant[mask].mean()
    return 1.0 - c_index

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
        "stride": 500,
        "window_size": 50,
        "validation_split": 0.15,
        "cpc_epochs": 100,
        "tp_epochs": 100,
        "dropout": 0.2,
        "temperature": 0.2,
        "latent_dim": 64,
        "context_dim": 256,
        "number_of_prediction_steps": 6,
        "batch_size": 512,
        "cpc_lr": 1e-3,
        "tp_lr": 1e-3
    }
    run = wandb.init(
        entity="eml-labs",
        project="CPC-New", 
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
    tp_epochs = config["tp_epochs"]
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


    cpc = CPC(
        latent_dim=config["latent_dim"],
        context_dim=config["context_dim"],
        number_of_prediction_steps=config["number_of_prediction_steps"],
        dropout=config["dropout"],
        temperature=config["temperature"]
    ).to(device)

    cpc_optimizer = optim.AdamW(cpc.parameters(), lr=config["cpc_lr"], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(cpc_optimizer, T_max=cpc_epochs, eta_min=config["cpc_lr"]*0.1)
    cpc.train()
    total_loader_size = len(train_data_loader)
    pbar = tqdm(total=cpc_epochs, desc="Training CPC Model")

    for cpc_epoch in range(cpc_epochs):
        total_samples = 0
        total_loss = 0
        total_accuracy = 0
        validation_loss = 0
        validation_accuracy = 0
        for rr,label,_ in train_data_loader:
            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            cpc_optimizer.zero_grad()
            loss, accuracy, _,_ = cpc(rr)
            total_loss += loss.item()
            total_accuracy += accuracy*rr.size(0)
            total_samples += rr.size(0)
            loss.backward()
            cpc_optimizer.step()
        scheduler.step()
        embeddings_list = []
        context_list = []
        label_list = []
        total_validation_samples = 0
        for rr,label,_ in validation_data_loader:
            rr = rr.to(device, non_blocking=True)
            label_list.extend(label.cpu().numpy().tolist())
            with torch.no_grad():
                loss, accuracy, embeddings, context = cpc(rr)
                validation_loss += loss.item()
                validation_accuracy += accuracy*rr.size(0)
                total_validation_samples += rr.size(0)
                embeddings_list.append(embeddings.cpu().numpy())
                context_list.append(context.cpu().numpy())

        pca_emb = PCA(n_components=2,random_state=42)
        pca_ctx = PCA(n_components=2,random_state=42)

        embeddings_all = np.concatenate(embeddings_list, axis=0)
        context_all = np.concatenate(context_list, axis=0)
        # For visualization, we take the last time step's embedding and context for each segment, which represents the most recent information before the event (AFIB start). This allows us to see how well the model is capturing the differences between SR, Mixed, and AFIB segments in the latent space. If we took all time steps, it would be harder to visualize and interpret the results, as each segment would contribute multiple points to the PCA plot.
        embeddings_all = embeddings_all[:, -1, :]
        context_all = context_all[:, -1, :]
        embeddings_pca = pca_emb.fit_transform(embeddings_all)
        context_pca = pca_ctx.fit_transform(context_all)
        # -1 - SR, 0 - Mixed, 1 - AFIB
        color_map = {-1: 'blue', 0: 'green', 1: 'red'}
        colors = [color_map[l] for l in label_list]
        fig = plt.figure(figsize=(12,12))
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        # ax3 = fig.add_subplot(gs[2:, 0])
        # ax4 = fig.add_subplot(gs[:, 0])

        scatter1 = ax1.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=colors, alpha=0.7,s=10)
        ax1.set_title("PCA of Latent Embeddings")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")

        scatter2 = ax2.scatter(context_pca[:,0], context_pca[:,1], c=colors, alpha=0.7,s=10)
        ax2.set_title("PCA of Context Vectors")
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")


        # Add color mapping legend
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color_map[k], label=lbl) for k, lbl in zip([-1, 0, 1], ['SR', 'Mixed', 'AFIB'])]
        fig.legend(handles=legend_handles, loc='upper right', title='Color Mapping')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pbar.update(1)
        pbar.set_description(f"Epoch {cpc_epoch+1}")
        pbar.write(f"Epoch {cpc_epoch+1}, Loss: {total_loss / len(train_data_loader):.4f}, Accuracy: {(total_accuracy / total_samples):.4f}, Validation Loss: {validation_loss / len(validation_data_loader):.4f}, Validation Accuracy: {(validation_accuracy / total_validation_samples):.4f}")
        run.log({
            "cpc_epoch": cpc_epoch+1,
            "cpc_lr": cpc_optimizer.param_groups[0]['lr'],
            "cpc_train_loss": total_loss / len(train_data_loader),
            "cpc_train_accuracy": (total_accuracy / total_samples) if total_samples > 0 else 0,
            "cpc_validation_loss": validation_loss / len(validation_data_loader),
            "cpc_validation_accuracy": (validation_accuracy / total_validation_samples) if total_validation_samples > 0 else 0,
            "pca": wandb.Image(fig)
        })
        plt.close(fig)


    torch.save(cpc.state_dict(), "cpc_model.pth")
    artifact = wandb.Artifact("cpc_model", type="model")
    artifact.add_file("cpc_model.pth")
    run.log_artifact(artifact)

    tp = TimePredictor(
        embedding_dim=config['latent_dim'],
        context_dim=config['context_dim'],
        dropout=config['dropout']
    ).to(device)

    tp_optimizer = optim.AdamW(tp.parameters(), lr=config["tp_lr"],weight_decay=1e-2)
    loss_fn = nn.HuberLoss(delta=1.0)

    cpc.eval()

    for param in cpc.parameters():
        param.requires_grad = False

    pbar = tqdm(total=tp_epochs, desc="Training Time Predictor")
    for tp_epoch in range(tp_epochs):
        total_loss = 0
        validation_loss = 0
        for rr,label,time in train_data_loader:
            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True).unsqueeze(1)
            tp_optimizer.zero_grad()
            with torch.no_grad():
                _,_, embeddings, context = cpc(rr)
            time_pred = tp(embeddings, context)
            loss = loss_fn(time_pred, time) + 0.5 * concordance_loss(time_pred, time)
            total_loss += loss.item()
            loss.backward()
            tp_optimizer.step()

        actual_times = []
        predicted_times = []
        for rr,label,time in validation_data_loader:
            rr = rr.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            time = time.to(device, non_blocking=True).unsqueeze(1)
            with torch.no_grad():
                _,_, embeddings, context = cpc(rr)
                time_pred = tp(embeddings, context)
                loss = loss_fn(time_pred, time) + 0.5 * concordance_loss(time_pred, time)
                validation_loss += loss.item()
                actual_times.append(time.cpu().numpy())
                predicted_times.append(time_pred.detach().cpu().numpy())

        pbar.update(1)
        pbar.set_description(f"Epoch {tp_epoch+1}")
        pbar.write(f"Epoch {tp_epoch+1}, Time Predictor Loss: {total_loss / len(train_data_loader):.4f}, Validation Loss: {validation_loss / len(validation_data_loader):.4f}")
        np_actual_times = np.concatenate(actual_times, axis=0)
        np_predicted_times = np.concatenate(predicted_times, axis=0)
        
        plt.figure(figsize=(6,6))
        plt.scatter(np_actual_times, np_predicted_times, alpha=0.5)
        plt.plot([0, np.max(np_actual_times)], [0, np.max(np_actual_times)], 'r--')  # Line for perfect predictions
        plt.xlabel("Actual Time to Event (hours)")
        plt.ylabel("Predicted Time to Event (hours)")
        plt.title(f"Actual vs Predicted Time to Event - Epoch {tp_epoch+1}")
        plt.xlim(0, np.max(np_actual_times)*1.1)
        plt.ylim(0, np.max(np_actual_times)*1.1)
        plt.grid()
        plt.tight_layout()

        run.log({
            "tp_epoch": tp_epoch+1,
            "tp_training_loss": total_loss / len(train_data_loader),
            "tp_validation_loss": validation_loss / len(validation_data_loader),
            "tp_scatter_plot": wandb.Image(plt)
        })
        plt.close()

    torch.save(tp.state_dict(), "time_predictor.pth")
    artifact = wandb.Artifact("time_predictor", type="model")
    artifact.add_file("time_predictor.pth")
    run.log_artifact(artifact)
    # run.finish()

except Exception as e:
    if 'logger' in locals():
        logger.error(f"An error occurred: {e}", exc_info=True)
    else:        
        print(f"An error occurred: {e}")
finally:
    if 'run' in locals():
        run.finish()