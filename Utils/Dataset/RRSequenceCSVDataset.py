import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict

from rr_loader import load_csv_records
from rr_windowing import build_csv_index

class RRSequenceCSVDataset(Dataset):
    def __init__(self, 
                 rri_csv_path : str,
                 features_csv_path : str,
                 seq_len : int,
                 horizons : List[int],):
        self.seq_len = seq_len
        self.horizons = horizons
        
        self.records = load_csv_records(
            rri_csv_path=rri_csv_path,
            features_csv_path=features_csv_path
        )
        self.index = build_csv_index(
            self.records, 
            seq_len=self.seq_len,
            horizons=self.horizons
        )

        first_rec = next(iter(self.records.values()))
        self.feature_names = first_rec["feature_names"]
        self.window_size = first_rec["rri"].shape[1]
        self.num_features = first_rec["hrv"].shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seg_name, start = self.index[idx]
        rec = self.records[seg_name]
        
        # Shape : (seq_len, window_size)
        rri_windows = rec["rri"][start : start + self.seq_len]

        # Shape : (num_features,)
        current_hrv = rec["hrv"][start + self.seq_len - 1]

        # Shape : (len(horizons), num_features)
        future_hrv_indices = [start + self.seq_len - 1 + h for h in self.horizons]
        hrvs = rec["hrv"][future_hrv_indices]

        return (
            torch.tensor(rri_windows, dtype=torch.float32),
            torch.tensor(current_hrv, dtype=torch.float32),
            torch.tensor(hrvs, dtype=torch.float32),
        )

