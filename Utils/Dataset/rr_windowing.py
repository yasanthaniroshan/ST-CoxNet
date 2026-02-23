from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_index(
    rr_records: List[np.ndarray],
    window_size: int,
    stride: int,
    horizons: list[int],
    seq_len: int,
) -> List[Tuple[int, int]]:
    """
    Build (record_id, start_index) pairs for all valid RR sequences.
    """
    index: List[Tuple[int, int]] = []
    history_len = window_size + stride * (seq_len - 1)
    future_len = window_size + stride * (max(horizons) - 1)
    total_len = history_len + future_len

    for rid, rr in enumerate(rr_records):
        max_start = len(rr) - total_len
        for start in range(0, max_start + 1, stride):
            index.append((rid, start))

    return index

def build_csv_index(
    records : Dict[str, Dict],
    seq_len :int, 
    horizons: List[int],
)-> List[Tuple[str, int]]:
    
    index = []
    max_horizon = max(horizons)

    for seg_name, record in records.items():
        n_winows = len(record["keys"])
        max_start = n_winows - seq_len - max_horizon + 1
        for start in range(0, max_start + 1):
            index.append((seg_name, start))
    return index



def build_rr_windows(
    rr_seq: np.ndarray,
    window_size: int,
    stride: int,
    seq_len: int,
) -> np.ndarray:
    """
    Build a sequence of RR windows of shape [T, W] from a slice of RR values.
    """
    return np.stack(
        [
            rr_seq[j * stride : j * stride + window_size]
            for j in range(seq_len)
        ]
    )

