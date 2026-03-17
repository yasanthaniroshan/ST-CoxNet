import torch
from torch.utils.data import Dataset
import pickle
import hashlib
import json
import os
import h5py
import numpy as np

class CPCCoxDataset(Dataset):
    def __init__(self,processed_dataset_path: str, afib_length:int,sr_length:int,number_of_windows_in_segment:int, stride:int, window_size:int,validation_split:float=0.15,train:bool=True):
        dataset_prop = {
            "dataset_name": "IRIDIA AFIB Dataset",
            "AFIB_length_seconds": afib_length,
            "SR_length_seconds": sr_length,
            "window_size": window_size,
            "segment_size": number_of_windows_in_segment*window_size,
            "stride": stride,
            "validation_split": validation_split,
            "scaler":"RobustScaler",
        }

        dataset_string = json.dumps(dataset_prop, sort_keys=True)
        file_name_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]

        try:
            if not os.path.exists(os.path.join(processed_dataset_path, f"{file_name_hash}.pkl")):
                raise FileNotFoundError("Dataset properties file not found.")
            dataset_prop_loaded = pickle.load(open(os.path.join(processed_dataset_path, f"{file_name_hash}.pkl"), "rb"))
            if dataset_prop_loaded != dataset_prop:
                raise ValueError("Dataset properties do not match the expected configuration.")
            file_path = os.path.join(processed_dataset_path, f"{file_name_hash}_train.h5") if train else os.path.join(processed_dataset_path, f"{file_name_hash}_validation.h5")
            data,labels,times = [],[],[]
            with h5py.File(file_path, "r") as f:
                data = f["segments"][:]
                labels = f["labels"][:]
                times = f["times"][:]
            mask = labels == -1
            data = data[mask]
            labels = labels[mask]
            times = times[mask]
            events = (times >=0).astype(int)
            
            self.data = torch.tensor(data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.times = torch.tensor(times, dtype=torch.float32)
            self.events = torch.tensor(events, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.times[idx], self.events[idx]