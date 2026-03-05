hrv_csv_file = "Utils/Dataset/AFDB/CSV_Files_2/hrv_features.csv"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Load the HRV features data
hrv_df = pd.read_csv(hrv_csv_file)

# Get all HRV feature columns (excluding metadata columns)
metadata_cols = ['Segment_Name', 'start_idx', 'end_idx']
hrv_feature_columns = [col for col in hrv_df.columns if col not in metadata_cols]

print(f"Found {len(hrv_feature_columns)} HRV features to plot:")
for i, feature in enumerate(hrv_feature_columns):
    print(f"{i+1:2d}. {feature}")

# Explore all available segments
print(f"\nAvailable segments in the CSV:")
unique_segments = hrv_df['Segment_Name'].unique()
for i, segment in enumerate(unique_segments):
    segment_count = len(hrv_df[hrv_df['Segment_Name'] == segment])
    segment_data = hrv_df[hrv_df['Segment_Name'] == segment]
    min_idx = segment_data['start_idx'].min()
    max_idx = segment_data['end_idx'].max()
    print(f"{i+1:2d}. {segment}: {segment_count} windows, range {min_idx}-{max_idx}")

# Create base plots directory
base_plots_dir = "Utils/Dataset/Tester/plots_by_feature"
os.makedirs(base_plots_dir, exist_ok=True)

# Create separate folders for each HRV feature
feature_dirs = {}
for feature in hrv_feature_columns:
    # Clean feature name for folder (remove HRV_ prefix and make filesystem-safe)
    folder_name = feature.replace('HRV_', '').replace('/', '_').replace('\\', '_')
    feature_dir = os.path.join(base_plots_dir, folder_name)
    os.makedirs(feature_dir, exist_ok=True)
    feature_dirs[feature] = feature_dir
    
print(f"\nCreated {len(feature_dirs)} feature directories in: {base_plots_dir}")

# Function to plot feature for a segment
def plot_feature_for_segment(segment, feature, feature_dir):
    """Plot last 3000 points of a specific feature for a segment"""
    
    # Get segment data
    seg_data = hrv_df[hrv_df['Segment_Name'] == segment].copy()
    
    if len(seg_data) == 0:
        return f"No data found for {segment}"
        
    # Get last 3000 indices
    max_end = seg_data['end_idx'].max()
    min_start_for_3000 = max_end - 3000
    
    last_3000_data = seg_data[seg_data['end_idx'] >= min_start_for_3000].copy()
    
    if len(last_3000_data) == 0:
        return f"No data in last 3000 indices for {segment}"
        
    # Extract feature values
    feature_values = last_3000_data[feature].values
    centers = last_3000_data['start_idx'].values + (last_3000_data['end_idx'] - last_3000_data['start_idx']).values / 2
    
    if len(feature_values) == 0:
        return f"No {feature} values found for {segment}"
    
    # Create plot
    plt.figure(figsize=(14, 8))
    plt.plot(centers, feature_values, 'bo-', linewidth=2, markersize=6)
    plt.title(f'{feature} - Last 3000 Indices for {segment}')
    plt.xlabel('Time Window Center (index)')
    plt.ylabel(f'{feature}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_base = f"{segment}_{feature}_last_3000_indices"
    png_file = os.path.join(feature_dir, f"{plot_base}.png")
    pdf_file = os.path.join(feature_dir, f"{plot_base}.pdf")
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    return f"Saved: {plot_base}"

# Main processing loop - Generate plots for all features and all segments
print(f"\n--- GENERATING PLOTS FOR ALL FEATURES AND SEGMENTS ---")
print(f"Total combinations to process: {len(unique_segments)} segments × {len(hrv_feature_columns)} features = {len(unique_segments) * len(hrv_feature_columns)}")

total_plots = 0
skipped_plots = 0

for feature_idx, feature in enumerate(hrv_feature_columns):
    print(f"\n[FEATURE {feature_idx+1}/{len(hrv_feature_columns)}] Processing {feature}...")
    feature_dir = feature_dirs[feature]
    
    feature_plots = 0
    feature_skipped = 0
    
    for segment_idx, segment in enumerate(unique_segments):
        result = plot_feature_for_segment(segment, feature, feature_dir)
        
        if "Saved:" in result:
            feature_plots += 1
            total_plots += 1
        else:
            feature_skipped += 1
            skipped_plots += 1
            print(f"  SKIP: {segment} - {result}")
    
    print(f"  {feature}: {feature_plots} plots saved, {feature_skipped} skipped")

# Summary
print(f"\n--- SUMMARY ---")
print(f"Total plots generated: {total_plots}")
print(f"Total plots skipped: {skipped_plots}")
print(f"Plots saved in: {base_plots_dir}")
print(f"\nFolder structure:")
for feature, folder in list(feature_dirs.items())[:5]:  # Show first 5
    folder_name = os.path.basename(folder)
    print(f"  {folder_name}/ - Contains {feature} plots for all segments")
if len(feature_dirs) > 5:
    print(f"  ... and {len(feature_dirs)-5} more feature folders")

# Quick statistics for first segment and feature as example
print(f"\n--- EXAMPLE STATISTICS ---")
if len(unique_segments) > 0 and len(hrv_feature_columns) > 0:
    example_segment = unique_segments[0]
    example_feature = hrv_feature_columns[0]
    
    seg_data = hrv_df[hrv_df['Segment_Name'] == example_segment].copy()
    max_end = seg_data['end_idx'].max()
    min_start_for_3000 = max_end - 3000
    last_3000_data = seg_data[seg_data['end_idx'] >= min_start_for_3000].copy()
    
    if len(last_3000_data) > 0:
        feature_values = last_3000_data[example_feature].values
        if len(feature_values) > 0:
            print(f"Example: {example_feature} for {example_segment} (last 3000 indices)")
            print(f"  Mean: {np.mean(feature_values):.4f}")
            print(f"  Std:  {np.std(feature_values):.4f}")
            print(f"  Min:  {np.min(feature_values):.4f}")
            print(f"  Max:  {np.max(feature_values):.4f}")
            print(f"  Points: {len(feature_values)}")

print(f"\nAll plots organized by feature type in: {base_plots_dir}")
print(f"Each folder contains plots for all segments showing that specific HRV feature.")
