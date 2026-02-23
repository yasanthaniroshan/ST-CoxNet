from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
from typing import Optional, Dict, Tuple

class SegmentExtractor:
    """Helper functions to extract pre-AF and AF data from extracted segments"""

    def __init__(self, extracted_segments_dir: str, extraction_report_path: str):

        self.segments_dir = Path(extracted_segments_dir)
        self.report_df = pd.read_csv(extraction_report_path)

    def get_segment_info(self, segment_name: str) -> Optional[Dict]:

        row = self.report_df[self.report_df['Segment_Name'] == segment_name]
        if row.empty:
            print(f"Segment {segment_name} not found in extraction report")
            return None
        return row.iloc[0].to_dict()

    def extract_data(self, segment_name: str, extract_type: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        # Get segment info from report
        info = self.get_segment_info(segment_name)
        if info is None:
            return None, None
        try:
            # Load the full record
            record_path = self.segments_dir / segment_name
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), 'atr')

            fs = int(record.fs)
            pre_af_minutes = info['Pre_AF_Minutes_Extracted']
            af_duration_minutes = info['AF_Duration_Minutes']

            # Calculate pre-AF samples (pre-AF is at the beginning of the segment)
            pre_af_samples = int(pre_af_minutes * 60 * fs)
            af_samples = int(af_duration_minutes * 60 * fs)

            af_start = pre_af_samples
            af_end = pre_af_samples + af_samples

            if extract_type == 'preaf':
                extracted_data = record.p_signal[:pre_af_samples, :]
                minutes_extracted = pre_af_minutes
                samples_extracted = pre_af_samples
            else:
                extracted_data = record.p_signal[af_start:af_end, :]
                minutes_extracted = af_duration_minutes
                samples_extracted = af_samples

            # Load QRS annotations
            try:
                qrs_ann = wfdb.rdann(str(record_path), 'qrs')
                if extract_type == 'preaf':
                    qrs_mask = qrs_ann.sample < pre_af_samples
                    qrs_samples = qrs_ann.sample[qrs_mask]
                else:
                    qrs_mask = (qrs_ann.sample >= af_start) & (qrs_ann.sample < af_end)
                    qrs_samples = qrs_ann.sample[qrs_mask] - af_start
                qrs_symbols = [qrs_ann.symbol[i] for i in range(len(qrs_ann.symbol))
                                  if qrs_mask[i]]
            except:
                qrs_samples = np.array([])
                qrs_symbols = []

            metadata = {
                'segment_name': segment_name,
                'record_id': info['Record_ID'],
                'episode_number': info['Episode_Number'],
                'duration_minutes': minutes_extracted,
                'duration_seconds': minutes_extracted * 60,
                'n_samples': samples_extracted,
                'sampling_frequency': fs,
                'n_channels': extracted_data.shape[1],
                'signal_names': record.sig_name[:extracted_data.shape[1]],
                'units': record.units[:extracted_data.shape[1]],
                'qrs_samples': qrs_samples,
                'qrs_symbols': qrs_symbols,
                'n_beats': len(qrs_samples)
            }

            if(extract_type == 'preaf'):
                print(f"✓ Extracted pre-AF data for {segment_name}: "
                  f"{minutes_extracted:.1f} min, {len(qrs_samples)} beats")
            else:
                print(f"✓ Extracted AF data for {segment_name}: "
                  f"{minutes_extracted:.1f} min, {len(qrs_samples)} beats")

            return extracted_data, metadata

        except Exception as e:
            print(f"Failed to extract data for {segment_name}: {e}")
            return None, None

    def extract_preaf_data(self, segment_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        return self.extract_data(segment_name, 'preaf')

    def extract_af_data(self, segment_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        return self.extract_data(segment_name, 'af')


    def extract_both(self, segment_name: str) -> Dict:
        preaf_ecg, preaf_meta = self.extract_preaf_data(segment_name)
        af_ecg, af_meta = self.extract_af_data(segment_name)

        return {
            'preaf': (preaf_ecg, preaf_meta),
            'af': (af_ecg, af_meta)
        }

    def get_rr_intervals(self, 
                         qrs_samples: np.ndarray, 
                         fs: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(qrs_samples) < 2:
            return np.array([]), np.array([])

        beat_times = qrs_samples / fs
        rr_intervals = np.diff(beat_times)

        return rr_intervals, beat_times[1:]
    
    def segment_rr_windows_rri(self, 
                               rr_intervals: np.ndarray, 
                               window_size_beats: int, 
                               stride_beats: int) -> Dict[Tuple[int, int], np.ndarray]:
        if len(rr_intervals) == 0:
            return {}

        segments = {}

        start_idx = 0
        while start_idx + window_size_beats <= len(rr_intervals):
            end_idx = start_idx + window_size_beats
            segments[(start_idx, end_idx)] = rr_intervals[start_idx:end_idx]
            start_idx += stride_beats

        return segments

    def segment_rr_windows_sec(self, 
                               rr_intervals: np.ndarray, 
                               beat_times: np.ndarray, 
                               window_size_sec: int, 
                               stride_sec: int) -> Dict[Tuple[int, int], np.ndarray]:
        segments = {}
        start_time = beat_times[0]

        while start_time + window_size_sec <= beat_times[-1]:
            end_time = start_time + window_size_sec
            mask = (beat_times >= start_time) & (beat_times < end_time)
            if len(rr_intervals[mask]) > 0:  
                segments[(start_time, end_time)] = rr_intervals[mask]
            start_time += stride_sec
        
        return segments
    
    def segment_rr_intervals_mins(self, 
                                  rr_intervals: np.ndarray, 
                                  beat_times: np.ndarray, 
                                  window_size_min: int,
                                  stride_min: int) -> Dict[Tuple[int, int], np.ndarray]:
        segments = {}
        start_time = beat_times[0]

        while start_time + window_size_min * 60 <= beat_times[-1]:
            end_time = start_time + window_size_min * 60
            start_time_min = start_time / 60.0
            end_time_min = start_time_min + window_size_min
            mask = (beat_times >= start_time) & (beat_times < end_time)
            if len(rr_intervals[mask]) > 0:  
                segments[(start_time_min, end_time_min)] = rr_intervals[mask]
            start_time += stride_min * 60
        
        return segments