from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple

import numpy as np

from Metadata import CSVLoaderMetadata, FileLoaderMetadata
from Utils.Loader.FileLoader import FileLoader


def load_rr_records(
    sampling_rate: int,
    file_loader_metadata: FileLoaderMetadata,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load RR-interval sequences and associated patient IDs from raw QRS detections.
    """
    rr_records: List[np.ndarray] = []
    patient_ids: List[int] = []

    file_loader = FileLoader(file_loader_metadata)
    pid = 0
    for _, qrs in file_loader.load():
        if qrs is None:
            continue
        rr = np.diff(qrs.sample) / sampling_rate
        rr_records.append(rr)
        patient_ids.append(pid)
        pid += 1

    return rr_records, patient_ids

def load_csv_records(
    rri_csv_path: str,
    features_csv_path: str
) -> Dict[str, Dict]:
    rri_df = pd.read_csv(rri_csv_path)
    features_df = pd.read_csv(features_csv_path)

    rri_cols = [col for col in rri_df.columns if col.startswith("rri_")]
    meta_cols = ["Segment_Name", "start_idx", "end_idx"]
    feature_cols = [col for col in features_df.columns if col not in meta_cols]

    merged = pd.merge(
        rri_df[meta_cols + rri_cols], 
        features_df[meta_cols + feature_cols], 
        on=meta_cols)
    
    records = {}

    for seg_name, group in merged.groupby("Segment_Name"):
        group = group.sort_values("start_idx").reset_index(drop=True)
        records[seg_name] = {
            "rri" : group[rri_cols].values.astype(np.float32),
            "hrv": group[feature_cols].values.astype(np.float32), 
            "keys": list(zip(group["start_idx"], group["end_idx"])), 
            "feature_names": feature_cols,
        }
    return records


