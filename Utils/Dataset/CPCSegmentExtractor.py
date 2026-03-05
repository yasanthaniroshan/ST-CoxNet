import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Utils.Dataset.AFDB.AFDBDatasetLoader import AFDBDatasetLoader
from Utils.Dataset.AFExtractor import AFExtractor
from Utils.Dataset.SegmentExtractor import SegmentExtractor
from Utils.FeatureExtractor.HRVMetrics.HRVFeatures import HRVFeatures
import pandas as pd
import numpy as np
import warnings

# Comprehensive warning suppression for NeuroKit2
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2")
warnings.filterwarnings("ignore", message=".*DFA_alpha2.*")
warnings.filterwarnings("ignore", message=".*long-term correlation.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar divide.*")
warnings.filterwarnings("ignore", message=".*entropy_multiscale.*")
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)



DATA_PATH = "Utils/Dataset/AFDB/physionet.org/files/afdb/1.0.0"
OUPUT_DIR = "Utils/Dataset/AFDB/Extracted_Segments"
CSV_DIR = "Utils/Dataset/AFDB/CSV_Files_3"

# loader = AFDBDatasetLoader(
#     data_path=DATA_PATH,
#     use_physionet=False
# )

# # Extract episodes with AF duration >= 60s and atleast pre-AF duration of 60min containing upto 120mins 
# extractor = AFExtractor(
#     loader=loader, 
#     AF_low = 60, 
#     preAF_low = 60, 
#     preAF_max = 120, 
#     output_dir=OUPUT_DIR
# )

# Analyse all records and save summary report

extractor_csv = OUPUT_DIR + '/EXTRACTION_REPORT.csv'
# Extract the segment names
summary_df = pd.read_csv(extractor_csv)
segment_names = summary_df['Segment_Name'].dropna().tolist()

print(f"Found {len(segment_names)} segments to process")
print(f"Processing segments in batches to avoid memory issues...")

# Process in batches to avoid memory issues
BATCH_SIZE = 5  # Process 5 segments at a time
total_batches = (len(segment_names) + BATCH_SIZE - 1) // BATCH_SIZE

segment_extractor = SegmentExtractor(
    extracted_segments_dir=OUPUT_DIR,
    extraction_report_path=OUPUT_DIR + '/EXTRACTION_REPORT.csv'
)

# Initialize CSV files 
rri_csv_file = CSV_DIR + '/rri_windows.csv'
features_csv_file = CSV_DIR + '/hrv_features.csv'

# Clear existing CSV files before starting
if os.path.exists(rri_csv_file):
    os.remove(rri_csv_file)
    print(f"Cleared existing {rri_csv_file}")

if os.path.exists(features_csv_file):
    os.remove(features_csv_file)
    print(f"Cleared existing {features_csv_file}")

first_write = True

for batch_num in range(total_batches):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min((batch_num + 1) * BATCH_SIZE, len(segment_names))
    batch_segments = segment_names[start_idx:end_idx]
    
    print(f"Processing batch {batch_num + 1}/{total_batches}: segments {start_idx + 1}-{end_idx}")
    
    rri_csv = []
    # features_csv = []

    for segment_name in batch_segments:
        try:
            print(f"  Processing {segment_name}...")
            preaf_ecg, preaf_meta = segment_extractor.extract_preaf_data(segment_name)
            qrs_samples = preaf_meta['qrs_samples']
            fs           = preaf_meta['sampling_frequency']
            rr_intervals, beat_times = segment_extractor.get_rr_intervals(qrs_samples, fs)
            # Extract 50rri windows with stride of 10rri
            seg_rri = segment_extractor.segment_rr_windows_rri(rr_intervals, 
                                                               window_size_beats=300, 
                                                               stride_beats=10)
    

            # Write 50 rri to the csv file
            for (start_idx, end_idx), rri_window in seg_rri.items():
                row = {
                    "Segment_Name": segment_name,
                    "start_idx" : start_idx,
                    "end_idx" : end_idx,
                }

                # features_row = {
                #     "Segment_Name": segment_name,
                #     "start_idx" : start_idx,
                #     "end_idx" : end_idx,
                # }

                for i, rri in enumerate(rri_window):
                    row[f"rri_{i}"] = rri
                
                # rri_ms  = np.array(rri_window) * 1000 
                # Trying AFDB sampled at 250Hz.
                # hrv_extractor = HRVFeatures(data=rri_ms, fs=250, rri_given=True)
                # features = hrv_extractor.compute_all()

                # for feature_name, feature_value in features.items():
                #     features_row[feature_name] = feature_value

                rri_csv.append(row)
                # features_csv.append(features_row)
                
        except Exception as e:
            print(f"    Error processing {segment_name}: {e}")
            continue
    
    # Save batch results
    # if rri_csv and features_csv:
    if rri_csv:
        print(f"  Saving batch {batch_num + 1} results...")
        
        # Save RRI windows
        rri_df = pd.DataFrame(rri_csv)
        rri_df.to_csv(rri_csv_file, mode='a', header=first_write, index=False)
        
        # Save HRV features  
        # features_df = pd.DataFrame(features_csv)
        # features_df.to_csv(features_csv_file, mode='a', header=first_write, index=False)
        
        first_write = False
        print(f"  Batch {batch_num + 1} completed: {len(rri_csv)} windows processed")
    
    # Clear memory
    # del rri_csv, features_csv
    del rri_csv
    # if 'rri_df' in locals():
    #     del rri_df, features_df

print(f"\n✓ Processing completed! Results saved to:")
print(f"  - {rri_csv_file}")
# print(f"  - {features_csv_file}")

