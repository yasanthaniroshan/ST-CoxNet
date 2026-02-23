from Utils.Dataset.AFDB import AFDBDatasetLoader
from Utils.Dataset.AFExtractor import AFExtractor
from Utils.Dataset.SegmentExtractor import SegmentExtractor
import pandas as pd
import numpy as np

DATA_PATH = ""
OUPUT_DIR =""

loader = AFDBDatasetLoader(
    data_path=DATA_PATH,
    use_physionet=False
)

# Extract episodes with AF duration >= 60s and atleast pre-AF duration of 60min containing upto 120mins 
extractor = AFExtractor(
    loader=loader, 
    AF_low = 60, 
    preAF_low = 60, 
    preAF_max = 120, 
    output_dir=OUPUT_DIR
)

# Analyse all records and save summary report
summary_df = extractor.analyze_all_records()

# Extract the segment names
segment_names = summary_df['Segment_Name'].tolist()

segment_extractor = SegmentExtractor(
    extracted_segments_dir=OUPUT_DIR,
    extraction_report_path=OUPUT_DIR + '/EXTRACTION_REPORT.csv'
)
rri_csv = []

for segment_name in segment_names:
    preaf_ecg, preaf_meta = segment_extractor.extract_preaf_data(segment_name)
    qrs_samples = preaf_meta['qrs_samples']
    fs           = preaf_meta['sampling_frequency']
    rr_intervals, beat_times = segment_extractor.get_rr_intervals(qrs_samples, fs)
    # Extract 50rri windows with stride of 10rri
    seg_rri = segment_extractor.segment_rr_windows_rri(rr_intervals, 
                                                       window_size_beats=50, 
                                                       stride_beats=10)
    # Write 50 rri to the csv file
    for (start_idx, end_idx), rri_window in seg_rri.items():
        row = {
            "Segment_Name": segment_name,
            "start_idx" : start_idx,
            "end_idx" : end_idx,
        }

        for i, rri in enumerate(rri_window):
            row[f"rri_{i}"] = rri

        rri_csv.append(row)
    
    

