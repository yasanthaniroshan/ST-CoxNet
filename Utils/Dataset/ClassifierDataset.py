from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np


class ClassifierDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = features.astype(np.float32)
        self.y = labels.astype(np.int64)  # LongTensor expected by CrossEntropyLoss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)