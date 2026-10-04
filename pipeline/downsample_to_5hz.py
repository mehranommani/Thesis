"""
================================================================================
STEP 0: Downsample inD Dataset from 25 Hz to 5 Hz
================================================================================

Run ONCE before the pipeline.

Reads aggregated_tracks.csv (25 Hz), keeps every 5th frame, renumbers frames.
Copies tracksMeta and recordingMeta unchanged.

All downstream scripts (merge, sampling, baking) read from the 5 Hz output.
================================================================================
"""

import os
import shutil
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBSAMPLE_FACTOR = 5  # 25 Hz → 5 Hz

ORIGINAL_DIR = "data/inD-dataset-v1/data_output"
OUTPUT_DIR = "data/inD-dataset-v1/data_output_5hz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1: DOWNSAMPLE AGGREGATED TRACKS
# ============================================================================

print("=" * 70)
print(f"DOWNSAMPLING inD FROM 25 Hz TO {25 // SUBSAMPLE_FACTOR} Hz")
print("=" * 70)

agg_path = f"{ORIGINAL_DIR}/aggregated_tracks.csv"
print(f"\nLoading {agg_path}...")
df = pd.read_csv(agg_path)
rows_before = len(df)
print(f"  Loaded: {rows_before:,} rows")

# Keep every 5th frame (globally consistent across all agents)
df = df[df['frame'] % SUBSAMPLE_FACTOR == 0].copy()

# Renumber: 0,5,10,15 → 0,1,2,3 (preserves diff==1 continuity)
df['frame'] = df['frame'] // SUBSAMPLE_FACTOR

rows_after = len(df)
out_path = f"{OUTPUT_DIR}/aggregated_tracks.csv"
df.to_csv(out_path, index=False)
print(f"  Saved: {rows_after:,} rows ({100 * rows_after / rows_before:.1f}% of original)")
print(f"  Output: {out_path}")

del df

# ============================================================================
# STEP 2: COPY META FILES UNCHANGED
# ============================================================================

print(f"\nCopying meta files unchanged...")
for meta_name in ['aggregated_tracksMeta.csv', 'aggregated_recordingMeta.csv']:
    src = f"{ORIGINAL_DIR}/{meta_name}"
    dst = f"{OUTPUT_DIR}/{meta_name}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✓ {meta_name}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 70}")
print(f"DONE")
print(f"{'=' * 70}")
print(f"  {rows_before:,} → {rows_after:,} rows ({SUBSAMPLE_FACTOR}× reduction)")
print(f"  Frame renumbering: original 0,5,10,15,... → 0,1,2,3,...")
print(f"  Velocities/accelerations: inD instantaneous values preserved")
print(f"\nNext: Run merge_and_analyze_csvs.py pointed at {OUTPUT_DIR}")
print(f"{'=' * 70}")