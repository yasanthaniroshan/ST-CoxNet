import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from Model.Classifier import Classifier
from Utils.Dataset.ClassifierDataset import ClassifierDataset
from tqdm import tqdm

random.seed(42)
torch.manual_seed(42)
dataset = pd.read_csv('Utils/Dataset/AFDB/CSV_Files/classification_dataset.csv')
segment_names = dataset['Segment_Name'].unique()
total_classes = dataset['label'].nunique()
print(f"Total unique segments: {len(segment_names)}, Total classes: {total_classes}")
print(f"Class distribution:\n{dataset['label'].value_counts()}")
train = random.sample(list(segment_names), int(0.8*len(segment_names)))
test = list(set(segment_names) - set(train))

train_df = dataset[dataset['Segment_Name'].isin(train)]
test_df = dataset[dataset['Segment_Name'].isin(test)]


feature_columns = ['RMSSD', 'pNN50', 'SDNN', 'alpha_1', 'ApEn']
label_column = 'label'

# Fit scaler only on training data
scaler = StandardScaler()
scaler.fit(train_df[feature_columns].values)  # only train

train_features = scaler.transform(train_df[feature_columns].values)
test_features = scaler.transform(test_df[feature_columns].values)

train_dataset = ClassifierDataset(train_features, train_df[label_column].values)
test_dataset = ClassifierDataset(test_features, test_df[label_column].values)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Classifier(number_of_features=len(feature_columns), number_of_classes=total_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 300
pbar = tqdm(range(num_epochs), desc="Training Epochs")
for epoch in pbar:
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    pbar.set_postfix({"Epoch": epoch + 1, "Loss": avg_loss})
    tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)              # raw logits
        predicted = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
