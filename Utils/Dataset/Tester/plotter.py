hrv_csv_file = "Utils/Dataset/AFDB/CSV_Files_2/hrv_features.csv"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the HRV features data
hrv_df = pd.read_csv(hrv_csv_file)

# Explore all available segments
print("Available segments in the CSV:")
unique_segments = hrv_df['Segment_Name'].unique()
for i, segment in enumerate(unique_segments):
    segment_count = len(hrv_df[hrv_df['Segment_Name'] == segment])
    segment_data = hrv_df[hrv_df['Segment_Name'] == segment]
    min_idx = segment_data['start_idx'].min()
    max_idx = segment_data['end_idx'].max()
    print(f"{i+1:2d}. {segment}: {segment_count} windows, range {min_idx}-{max_idx}")

# Select a segment to analyze (you can change this)
Segment_Name = "04043_ep02"  # Specific patient segment
print(f"\nAnalyzing segment: {Segment_Name}")

# Filter data for the specific segment
segment_data = hrv_df[hrv_df['Segment_Name'] == Segment_Name].copy()

print(f"Found {len(segment_data)} time windows for segment: {Segment_Name}")
print(f"Full time range: {segment_data['start_idx'].min()} to {segment_data['end_idx'].max()}")

# Filter for indices 9500 to 11000
start_idx_filter = 9500
end_idx_filter = 11000

print(f"Filtering for indices {start_idx_filter} to {end_idx_filter}")

# Filter data within the specified range
filtered_data = segment_data[
    (segment_data['start_idx'] >= start_idx_filter) & (segment_data['end_idx'] <= end_idx_filter)
].copy()

print(f"Found {len(filtered_data)} windows in range {start_idx_filter}-{end_idx_filter}")
print(f"Filtered time range: {filtered_data['start_idx'].min()} to {filtered_data['end_idx'].max()}")

# Extract HRV_RMSSD values from filtered data
hrv_rmssd_values = filtered_data['HRV_RMSSD'].values
window_centers = filtered_data['start_idx'].values + (filtered_data['end_idx'] - filtered_data['start_idx']).values / 2

print(f"\nHRV_RMSSD values:")
if len(hrv_rmssd_values) > 0:
    for i, (center, rmssd) in enumerate(zip(window_centers, hrv_rmssd_values)):
        print(f"Window {i+1} (center: {center:.1f}): {rmssd:.4f} ms")

    # Plot the HRV_RMSSD transformation over time
    plt.figure(figsize=(14, 8))
    plt.plot(window_centers, hrv_rmssd_values, 'bo-', linewidth=2, markersize=6)
    plt.title(f'HRV RMSSD - Indices {start_idx_filter}-{end_idx_filter} for {Segment_Name}')
    plt.xlabel('Time Window Center (index)')
    plt.ylabel('HRV_RMSSD (ms)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plots_dir = "Utils/Dataset/Tester/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_filename_base = f"{Segment_Name}_RMSSD_{start_idx_filter}_{end_idx_filter}"
    png_path = os.path.join(plots_dir, f"{plot_filename_base}.png")
    pdf_path = os.path.join(plots_dir, f"{plot_filename_base}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved as:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    # Add statistics
    print(f"\nHRV_RMSSD Statistics for {Segment_Name} (Indices {start_idx_filter}-{end_idx_filter}):")
    print(f"Mean: {np.mean(hrv_rmssd_values):.4f} ms")
    print(f"Std:  {np.std(hrv_rmssd_values):.4f} ms")
    print(f"Min:  {np.min(hrv_rmssd_values):.4f} ms")
    print(f"Max:  {np.max(hrv_rmssd_values):.4f} ms")
    print(f"Range: {np.max(hrv_rmssd_values) - np.min(hrv_rmssd_values):.4f} ms")

    plt.show()
else:
    print("No data found in the last 3000 indices.")
    print("Available data ranges:")
    all_data = hrv_df[hrv_df['Segment_Name'] == Segment_Name]
    for i, row in all_data.iterrows():
        print(f"  Window: {row['start_idx']} to {row['end_idx']}")
    print("\nTry adjusting the range to match available data.")

# Summary of segment exploration
print(f"\n--- SEGMENT EXPLORATION SUMMARY ---")
print(f"Total segments available: {len(unique_segments)}")
print(f"Currently analyzing: {Segment_Name}")

# Generate and save plots for all segments
print(f"\n--- GENERATING PLOTS FOR ALL SEGMENTS ---")
plots_dir = "Utils/Dataset/Tester/plots"
os.makedirs(plots_dir, exist_ok=True)

for i, segment in enumerate(unique_segments):
    print(f"Processing {i+1}/{len(unique_segments)}: {segment}")
    
    # Get segment data
    seg_data = hrv_df[hrv_df['Segment_Name'] == segment].copy()
    
    if len(seg_data) == 0:
        print(f"  No data found for {segment}")
        continue
        
    # Get last 3000 indices
    max_end = seg_data['end_idx'].max()
    min_start_for_3000 = max_end - 3000
    
    last_3000_data = seg_data[seg_data['end_idx'] >= min_start_for_3000].copy()
    
    if len(last_3000_data) == 0:
        print(f"  No data in last 3000 indices for {segment}")
        continue
        
    # Extract values
    rmssd_vals = last_3000_data['HRV_RMSSD'].values
    centers = last_3000_data['start_idx'].values + (last_3000_data['end_idx'] - last_3000_data['start_idx']).values / 2
    
    # Create plot
    plt.figure(figsize=(14, 8))
    plt.plot(centers, rmssd_vals, 'bo-', linewidth=2, markersize=6)
    plt.title(f'HRV RMSSD - Last 3000 Indices for {segment}')
    plt.xlabel('Time Window Center (index)')
    plt.ylabel('HRV_RMSSD (ms)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_base = f"{segment}_RMSSD_last_3000_indices"
    png_file = os.path.join(plots_dir, f"{plot_base}.png")
    pdf_file = os.path.join(plots_dir, f"{plot_base}.pdf")
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    print(f"  Saved: {plot_base}.png and .pdf")

print(f"\nAll plots saved to: {plots_dir}")
print(f"To analyze a different segment, change 'Segment_Name' to one of:")
for segment in unique_segments[:10]:  # Show first 10
    print(f"  '{segment}'")
if len(unique_segments) > 10:
    print(f"  ... and {len(unique_segments)-10} more")

