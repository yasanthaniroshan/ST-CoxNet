from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import wfdb
from abc import ABC, abstractmethod

class DatasetLoader(ABC):

    def __init__(self,
                 dataset_name: str, 
                 search_paths: Optional[List[str]], 
                 data_path: Optional[str] = None, 
                 use_physionet: bool = False):
        """
        Initialize Data loader

        Args:
            dataset_name: Name of the dataset (e.g., 'afdb')
            search_paths: List of paths to search for local data
            data_path: Path to local dataset
            use_physionet: If True, download from PhysioNet when needed
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.use_physionet = use_physionet
        self.search_paths = search_paths

        # Auto-discover data path
        self._discover_data_path()
        self._discover_record_ids()

    def _discover_data_path(self):
        """Auto-discover Data path"""
        if self.data_path and Path(self.data_path).exists():
            print(f"Using provided path: {self.data_path}")
            return

        for path in self.search_paths:
            path_obj = Path(path)
            if path_obj.exists() and list(path_obj.glob('*.hea')):
                self.data_path = str(path_obj)
                print(f"Auto-discovered data path: {self.data_path}")
                return

        print("Could not auto-discover data path. Will try PhysioNet.")
        self.data_path = None

    def _discover_record_ids(self):
        """Auto-discover record IDs"""
        if self.data_path:
            hea_files = sorted(Path(self.data_path).glob('*.hea'))
            self.record_ids = [f.stem for f in hea_files]
            print(f"Found {len(self.record_ids)} records")
        elif self.use_physionet:
            try:
                self.record_ids = wfdb.get_record_list(self.dataset_name)
                print(f"Found {len(self.record_ids)} records from PhysioNet")
            except Exception as e:
                print(f"Failed to get record list from PhysioNet: {e}")
                self.record_ids = []
        else:
            self.record_ids = []

    def load_record(self, 
                    record_id: str) -> Tuple[Optional[wfdb.Record], Optional[wfdb.Annotation]]:
        """Load record with fallback methods"""
        # Try local path
        if self.data_path:
            try:
                local_path = Path(self.data_path) / record_id
                record = wfdb.rdrecord(str(local_path.with_suffix('')))
                annotation = wfdb.rdann(str(local_path.with_suffix('')), 'atr')
                return record, annotation
            except Exception as e:
                print(f"Failed loading {record_id}: {e}")

        # Try PhysioNet
        if self.use_physionet:
            try:
                record = wfdb.rdrecord(record_id, pn_dir= self.dataset_name)
                annotation = wfdb.rdann(record_id, 'atr', pn_dir=self.dataset_name)
                return record, annotation
            except:
                try:
                    record = wfdb.rdrecord(f'{self.dataset_name}/{record_id}')
                    annotation = wfdb.rdann(f'{self.dataset_name}/{record_id}', 'atr')
                    return record, annotation
                except Exception as e:
                    print(f"Failed loading the record from physionet {record_id}: {e}")

        return None, None

    def load_qrs_annotation(self, record_id: str) -> Optional[wfdb.Annotation]:
        """Load QRS annotation file"""
        # Try local path
        if self.data_path:
            try:
                local_path = Path(self.data_path) / record_id
                qrs_ann = wfdb.rdann(str(local_path.with_suffix('')), 'qrs')
                return qrs_ann
            except Exception as e:
                print(f"Failed loading .qrs in the local file {record_id}: {e}")

        # Try PhysioNet
        if self.use_physionet:
            try:
                qrs_ann = wfdb.rdann(record_id, 'qrs', pn_dir=self.dataset_name)
                return qrs_ann
            except:
                try:
                    qrs_ann = wfdb.rdann(f'{self.dataset_name}/{record_id}', 'qrs')
                    return qrs_ann
                except Exception as e:
                    print(f"    Failed loading .qrs in physionet {record_id}: {e}")

        return None

    def get_rr_intervals_from_qrs(self, qrs_annotation: wfdb.Annotation, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract RR intervals from QRS annotations

        Returns:
            Tuple of (rr_intervals in seconds, beat_times in seconds)
        """
        if qrs_annotation is None or len(qrs_annotation.sample) < 2:
            return np.array([]), np.array([])

        # QRS annotation samples are the beat locations
        beat_samples = qrs_annotation.sample
        beat_times = beat_samples / fs

        # Calculate RR intervals
        rr_intervals = np.diff(beat_times)

        return rr_intervals, beat_times[1:]  # Times correspond to second beat of each RR

    @abstractmethod
    def parse_rhythm_segments(self, annotation: wfdb.Annotation, fs: int, total_samples: int):
        """
        Parse rhythm segments (AF and SR only) from the annotation file. 
        Dataset-specific rhythm interpretation
        Must be implemented by child class
        """
        raise NotImplementedError("parse_rhythm_segments must be implemented in dataset-specific loader")