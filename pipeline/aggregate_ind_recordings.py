"""
Step 0: combine the 33 inD recordings into three aggregated CSV files.

Reads every *_recordingMeta.csv, *_tracks.csv and *_tracksMeta.csv from the
original inD data folder and concatenates each type into one file:
aggregated_recordingMeta.csv, aggregated_tracks.csv, aggregated_tracksMeta.csv.

Run from the repository root. The output folder is the input of
downsample_to_5hz.py.
"""

import os
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "data/inD-dataset-v1/data"          # original inD CSV files
OUTPUT_DIR = "data/inD-dataset-v1/data_output"  # aggregated files go here
os.makedirs(OUTPUT_DIR, exist_ok=True)

recording_meta_list = []
tracks_list = []
tracks_meta_list = []

for file_name in sorted(os.listdir(DATA_DIR)):
    file_path = os.path.join(DATA_DIR, file_name)
    if file_name.endswith("recordingMeta.csv"):
        recording_meta_list.append(pd.read_csv(file_path))
    elif file_name.endswith("tracksMeta.csv"):
        tracks_meta_list.append(pd.read_csv(file_path))
    elif file_name.endswith("tracks.csv"):
        tracks_list.append(pd.read_csv(file_path))

if recording_meta_list:
    pd.concat(recording_meta_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "aggregated_recordingMeta.csv"), index=False)
    print("Saved aggregated_recordingMeta.csv")

if tracks_list:
    pd.concat(tracks_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "aggregated_tracks.csv"), index=False)
    print("Saved aggregated_tracks.csv")

if tracks_meta_list:
    pd.concat(tracks_meta_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "aggregated_tracksMeta.csv"), index=False)
    print("Saved aggregated_tracksMeta.csv")

print("Aggregation complete.")
