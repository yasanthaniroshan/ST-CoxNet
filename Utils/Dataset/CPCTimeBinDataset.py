import torch
from torch.utils.data import Dataset
import pickle
import hashlib
import json
import os
import h5py
import numpy as np


class CPCTimeBinDataset(Dataset):
    """SR-only segments with time-to-event binned into discrete classes.

    Bin edges define the boundaries: e.g. [0, 5, 10, 15, 20, 25, 30]
    creates 6 bins: [0,5), [5,10), [10,15), [15,20), [20,25), [25, inf).
    The last bin catches everything >= last-but-one edge.
    """

    def __init__(
        self,
        processed_dataset_path: str,
        afib_length: int,
        sr_length: int,
        number_of_windows_in_segment: int,
        stride: int,
        window_size: int,
        bin_edges: list,
        validation_split: float = 0.15,
        train: bool = True,
    ):
        dataset_prop = {
            "dataset_name": "IRIDIA AFIB Dataset",
            "AFIB_length_seconds": afib_length,
            "SR_length_seconds": sr_length,
            "window_size": window_size,
            "segment_size": number_of_windows_in_segment * window_size,
            "stride": stride,
            "validation_split": validation_split,
            "scaler": "RobustScaler",
        }

        dataset_string = json.dumps(dataset_prop, sort_keys=True)
        file_name_hash = hashlib.sha256(dataset_string.encode()).hexdigest()[:32]

        try:
            dataset_props_path = os.path.join(processed_dataset_path, f"{file_name_hash}.pkl")
            if not os.path.exists(dataset_props_path):
                raise FileNotFoundError("Dataset properties file not found.")

            dataset_prop_loaded = pickle.load(open(dataset_props_path, "rb"))
            if dataset_prop_loaded != dataset_prop:
                raise ValueError("Dataset properties do not match the expected configuration.")

            file_path = (
                os.path.join(processed_dataset_path, f"{file_name_hash}_train.h5")
                if train
                else os.path.join(processed_dataset_path, f"{file_name_hash}_validation.h5")
            )

            with h5py.File(file_path, "r") as f:
                data = f["segments"][:]
                labels = f["labels"][:]
                times = f["times"][:]

            mask = labels == -1
            data = data[mask]
            times = times[mask]

            bin_edges = np.array(bin_edges, dtype=np.float64)
            bin_labels = np.digitize(times, bin_edges) - 1
            num_bins = len(bin_edges) - 1
            bin_labels = np.clip(bin_labels, 0, num_bins - 1)

            self.data = torch.tensor(data, dtype=torch.float32)
            self.bin_labels = torch.tensor(bin_labels, dtype=torch.long)
            self.times = torch.tensor(times, dtype=torch.float32)
            self.num_classes = num_bins
            self.bin_edges = bin_edges

        except Exception as e:
            print(f"Error loading time-bin dataset: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.bin_labels[idx], self.times[idx]
