from Model.CPC import CPC
from Model.CoxHead.Base import CoxHead
from Utils.Dataset.CPCCoxDataset import CPCCoxDataset
from Utils.Dataset.CPCDataset import CPCDataset
from torch.utils.data import DataLoader
import torch

config={
    "afib_length": 60*60,
    "sr_length": int(1.5*60*60),
    "number_of_windows_in_segment": 10,
    "stride": 500,
    "window_size": 50,
    "validation_split": 0.15,
    "cpc_epochs": 100,
    "cox_epochs": 100,
    "dropout": 0.2,
    "temperature": 0.2,
    "latent_dim": 64,
    "context_dim": 128,
    "number_of_prediction_steps": 6,
    "batch_size": 1024,
    "cpc_lr": 1e-3,
    "cox_lr": 1e-3
}

afib_length = config["afib_length"]
sr_length = config["sr_length"]
number_of_windows_in_segment = config["number_of_windows_in_segment"]
stride = config["stride"]
window_size = config["window_size"]
validation_split = config["validation_split"]
cpc_epochs = config["cpc_epochs"]
cox_epochs = config["cox_epochs"]
processed_dataset_path = "//home/intellisense01/EML-Labs/ST-CoxNet/processed_datasets"

cpc_train_dataset = CPCDataset(
    processed_dataset_path=processed_dataset_path,
    afib_length=afib_length,
    sr_length=sr_length,
    number_of_windows_in_segment=number_of_windows_in_segment,
    stride=stride,
    window_size=window_size,
    validation_split=validation_split,
    train=True
)

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
cpc_train_dataloader = DataLoader(cpc_train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
cox_train_dataloader = DataLoader(cox_train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
print(f"Total CPC training samples: {len(cpc_train_dataset)}")
print(f"Total Cox training samples: {len(cox_train_dataset)}")
rr,labels,times = next(iter(cpc_train_dataloader))
print(f"CPC Train Batch - RR shape: {rr.shape}, Labels shape: {labels.shape}, Times shape: {times.shape}")
rr,labels,times,events = next(iter(cox_train_dataloader))
print(f"Cox Train Batch - RR shape: {rr.shape}, Labels shape: {labels.shape}, Times shape: {times.shape}, Events shape: {events.shape}")
print(torch.sum(events))
print(times)

cpc = CPC(
    latent_dim=config["latent_dim"],
    context_dim=config["context_dim"],
    number_of_prediction_steps=config["number_of_prediction_steps"],
    temperature=config["temperature"],
    dropout=config["dropout"]  
)

cox = CoxHead(
    context_dim=config["context_dim"],
    latent_dim=config["latent_dim"],
    dropout=config["dropout"]
)

avg_loss,avg_acc,latent,context = cpc(rr)
pred_danger = cox(context, latent)

print(f"CPC Output - Loss: {avg_loss}, Accuracy: {avg_acc}, Context shape: {context.shape}, Latent shape: {latent.shape}")
print(f"Cox Output - Predicted Risk shape: {pred_danger.shape}")