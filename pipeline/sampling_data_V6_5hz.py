"""
================================================================================
STEP 2: Stratified Sampling — V6 (5 Hz)
================================================================================

Changes from V5:
1. TARGET_TOTAL_TRAJECTORIES replaces TARGET_TOTAL_FRAMES.
   Sampling is now counted in trajectories, not raw frame rows, which is the
   natural unit that maps directly to baking sequence counts.

2. MIN_FOCAL_DISPLACEMENT_M = 2.0  — trajectories where the focal agent moves
   less than 2 metres are excluded from the pool entirely. This prevents
   stationary agents from dominating baked training windows.

3. Stationary scenario weight reduced from 0.10 → 0.05.  Combined with the
   displacement filter above, stationary data becomes a small minority rather
   than a silent majority.

4. OUTPUT_DIR renamed from "sampled_balanced_small_sample" →
   "sampled_balanced" to remove the misleading "small_sample" suffix.

5. TARGET_TOTAL_TRAJECTORIES sized so the baking pipeline can draw
   N_TRAIN=5000, N_VAL=1000, N_TEST=1000 without heavy upsampling:
   - Train pool target: 8000 trajectories (1.6× N_TRAIN for diversity)
   - Val/Test pool target: 2000 each
================================================================================
"""

import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from tqdm.auto import tqdm
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR  = "data/merged_analyzed"
OUTPUT_DIR = "data/sampled_balanced"          # ← fixed from "small_sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MERGED_CSV = f"{INPUT_DIR}/merged_full.csv"

# ============================================================================
# KEY PARAMETERS
# ============================================================================

FPS = 5
MIN_CONTINUOUS_LENGTH = 40      # 8 seconds × 5 FPS — minimum usable trajectory

# Minimum total displacement the focal agent must make over the whole trajectory.
# Trajectories below this threshold are excluded from the pool so models never
# learn a "stay still" prior from parked agents.
MIN_FOCAL_DISPLACEMENT_M = 2.0

MAX_UPSAMPLE_FACTOR = 3.0       # reduced from 5.0 to avoid over-repetition

# Target trajectory counts per split pool.
# These are intentionally larger than the baking N_TRAIN/N_VAL/N_TEST so that
# random.sample() in the baking step draws diverse, non-repeated trajectories.
TARGET_TRAIN_TRAJECTORIES = 8000
TARGET_VAL_TRAJECTORIES   = 2000
TARGET_TEST_TRAJECTORIES  = 2000

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
np.random.seed(SEED)

SIMPLIFIED_SCENARIOS = [
    'turning_maneuver', 'lane_change', 'straight',
    'stopping_behavior', 'stationary'
]
AGENT_CLASSES = ['car', 'pedestrian', 'bicycle', 'truck_bus']

# ============================================================================
# TARGET DISTRIBUTIONS
# Stationary weight reduced to 0.05 (was 0.10).
# All weights must sum to 1.0.
# ============================================================================

def get_scenario_weights():
    return {
        'turning_maneuver': 0.27,
        'straight':         0.27,
        'lane_change':      0.25,
        'stopping_behavior': 0.16,
        'stationary':       0.05,   # ← reduced from 0.10
    }

def get_class_weights():
    return {
        'car':       0.45,
        'pedestrian': 0.25,
        'bicycle':   0.22,
        'truck_bus': 0.08,
    }

# ============================================================================
# HELPERS
# ============================================================================

def normalize_class_names(class_name):
    s = str(class_name).lower()
    if 'car'       in s: return 'car'
    if 'truck' in s or 'bus' in s: return 'truck_bus'
    if 'pedestrian' in s or 'ped' in s: return 'pedestrian'
    if 'bicycle' in s or 'bike' in s:   return 'bicycle'
    return 'other'


def analyze_trajectory(frames_array, utm_xy_array, min_length=40,
                       min_displacement_m=2.0):
    """
    Analyse a single trajectory and return a dict with validity flags.

    Parameters
    ----------
    frames_array       : sorted int array of frame numbers
    utm_xy_array       : float array [N, 2] of UTM positions aligned to frames
    min_length         : minimum continuous segment length in frames
    min_displacement_m : minimum total displacement over trajectory in metres

    Returns
    -------
    dict with keys:
        has_valid_segment   bool
        passes_displacement bool  — agent moves at least min_displacement_m
        usable_frames       int
        estimated_samples   int
    """
    frames = np.asarray(frames_array)
    n = len(frames)

    result = {
        'has_valid_segment':   False,
        'passes_displacement': False,
        'usable_frames':       0,
        'estimated_samples':   0,
    }

    if n < min_length:
        return result

    # Displacement check — use full trajectory end-to-end distance
    if utm_xy_array is not None and len(utm_xy_array) == n:
        disp = float(np.linalg.norm(utm_xy_array[-1] - utm_xy_array[0]))
        result['passes_displacement'] = disp >= min_displacement_m
    else:
        result['passes_displacement'] = True  # cannot check, assume ok

    # Continuous segment analysis
    diffs = np.diff(frames)
    gap_idx = np.where(diffs != 1)[0] + 1

    segments = []
    prev = 0
    for gi in gap_idx:
        segments.append(gi - prev)
        prev = gi
    segments.append(n - prev)

    valid_segs = [s for s in segments if s >= min_length]
    if not valid_segs:
        return result

    result['has_valid_segment'] = True
    result['usable_frames']     = sum(valid_segs)
    result['estimated_samples'] = sum(max(0, s - min_length + 1)
                                      for s in valid_segs)
    return result


# ============================================================================
# STEP 1: PREPARE TRAJECTORY METADATA
# ============================================================================

def prepare_trajectory_data(merged_df):
    print(f"\n{'='*80}\nPREPARING TRAJECTORY DATA\n{'='*80}")

    if 'class_normalized' not in merged_df.columns:
        merged_df['class_normalized'] = merged_df['class'].apply(normalize_class_names)

    # Pre-build UTM columns if not present (needed for displacement check)
    if 'utm_x' not in merged_df.columns:
        # Cannot check displacement without UTM — mark all as passing
        merged_df['utm_x'] = 0.0
        merged_df['utm_y'] = 0.0

    traj_info = []
    grouped = merged_df.groupby(['recordingId', 'trackId'])

    for (rec_id, track_id), group in tqdm(grouped, desc="Analysing trajectories"):
        group_sorted   = group.sort_values('frame')
        frames_array   = group_sorted['frame'].values
        utm_xy_array   = group_sorted[['utm_x', 'utm_y']].values

        analysis = analyze_trajectory(
            frames_array, utm_xy_array,
            min_length=MIN_CONTINUOUS_LENGTH,
            min_displacement_m=MIN_FOCAL_DISPLACEMENT_M,
        )

        if not analysis['has_valid_segment']:
            continue

        # Trajectories that don't pass displacement are kept in the pool
        # only for the 'stationary' scenario bin (they are genuinely stationary).
        cls      = normalize_class_names(group['class'].iloc[0])
        scenario = str(group['scenario_simplified'].iloc[0])

        if not analysis['passes_displacement'] and scenario != 'stationary':
            continue   # skip non-stationary trajectories that barely move

        traj_info.append({
            'recordingId':        rec_id,
            'trackId':            track_id,
            'class':              cls,
            'scenario':           scenario,
            'n_frames':           len(frames_array),
            'usable_frames':      analysis['usable_frames'],
            'estimated_samples':  analysis['estimated_samples'],
            'passes_displacement': analysis['passes_displacement'],
            'stratify_key':       f"{cls}_{scenario}",
        })

    traj_df = pd.DataFrame(traj_info)
    print(f"Valid trajectories (all):       {len(traj_df):,}")
    moving = traj_df['passes_displacement'].sum()
    print(f"  of which moving (≥{MIN_FOCAL_DISPLACEMENT_M}m): {moving:,}")
    print(f"  stationary retained:          {len(traj_df) - moving:,}")
    print(f"\nPool by scenario:")
    print(traj_df['stratify_key'].value_counts().to_string())
    return merged_df, traj_df


# ============================================================================
# STEP 2: GLOBAL STRATIFIED SPLIT
# ============================================================================

def perform_global_split(traj_df):
    """
    Split the full trajectory pool into Train / Val / Test BEFORE sampling
    so each split's pool is independently balanced.
    """
    print(f"\n{'='*80}\nGLOBAL STRATIFIED SPLIT\n{'='*80}")

    train_dfs, val_dfs, test_dfs = [], [], []

    for key, group in traj_df.groupby('stratify_key'):
        n = len(group)
        group = group.sample(frac=1, random_state=SEED)

        if n >= 4:
            train, temp = train_test_split(group, train_size=TRAIN_RATIO,
                                           random_state=SEED)
            val_rel = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
            val, test = train_test_split(temp, train_size=val_rel,
                                         random_state=SEED)
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)
        elif n == 3:
            train_dfs.append(group.iloc[[0]])
            val_dfs.append(group.iloc[[1]])
            test_dfs.append(group.iloc[[2]])
        elif n == 2:
            train_dfs.append(group.iloc[[0]])
            test_dfs.append(group.iloc[[1]])
        else:
            train_dfs.append(group)

    train_pool = pd.concat(train_dfs).reset_index(drop=True)
    val_pool   = pd.concat(val_dfs).reset_index(drop=True)
    test_pool  = pd.concat(test_dfs).reset_index(drop=True) if test_dfs else pd.DataFrame()

    print(f"  Train pool: {len(train_pool):,} trajectories")
    print(f"  Val pool:   {len(val_pool):,}  trajectories")
    print(f"  Test pool:  {len(test_pool):,}  trajectories")
    return train_pool, val_pool, test_pool


# ============================================================================
# STEP 3: SAMPLE FROM POOL (count in trajectories, not frames)
# ============================================================================

def sample_from_pool(pool_df, target_trajectories, split_name):
    """
    Sample trajectories from a split pool targeting a balanced class×scenario
    distribution.  Counts in trajectories (not frames) to directly control
    how many baking windows the pool can yield.
    """
    print(f"\n  {split_name.upper()} — target {target_trajectories:,} trajectories")

    scenario_weights = get_scenario_weights()
    class_weights    = get_class_weights()

    available_bins = set(zip(pool_df['class'], pool_df['scenario']))

    # Build per-bin targets (normalised over available bins)
    bin_weights = {}
    for cls in AGENT_CLASSES:
        for scen in SIMPLIFIED_SCENARIOS:
            if (cls, scen) in available_bins:
                bin_weights[(cls, scen)] = class_weights[cls] * scenario_weights[scen]

    total_w = sum(bin_weights.values())
    selections = []
    repetition = defaultdict(int)

    for (cls, scen), w in bin_weights.items():
        bin_target = int(target_trajectories * (w / total_w))

        candidates = (
            pool_df[(pool_df['class'] == cls) & (pool_df['scenario'] == scen)]
            .sort_values('estimated_samples', ascending=False)
            .to_dict('records')
        )

        count  = 0
        rounds = 0
        while count < bin_target and rounds < len(candidates) * int(MAX_UPSAMPLE_FACTOR + 1):
            cand = candidates[rounds % len(candidates)]
            key  = (cand['recordingId'], cand['trackId'])
            if repetition[key] < MAX_UPSAMPLE_FACTOR:
                selections.append(cand)
                repetition[key] += 1
                count += 1
            rounds += 1

    print(f"    Selected {len(selections):,} trajectories "
          f"(target {target_trajectories:,})")
    return selections


# ============================================================================
# STEP 4: EXTRACT AND SAVE
# ============================================================================

def verify_and_save(merged_df, selections_map):
    print(f"\n{'='*80}\nEXTRACTING AND SAVING\n{'='*80}")
    stats = {}

    for split_name, selections in selections_map.items():
        frames_list = []

        print(f"\nExtracting {split_name} ({len(selections):,} selections)...")
        for item in tqdm(selections):
            traj = merged_df[
                (merged_df['recordingId'] == item['recordingId']) &
                (merged_df['trackId']     == item['trackId'])
            ].copy().sort_values('frame')

            # Stationary cap: keep at most 400 frames (80s at 5Hz) per stationary agent
            # so they don't dominate baking window counts
            if item['scenario'] == 'stationary' and len(traj) > 400:
                mid   = len(traj) // 2
                start = max(0, mid - 200)
                traj  = traj.iloc[start:start + 400]

            frames_list.append(traj)

        if not frames_list:
            continue

        split_df = pd.concat(frames_list, ignore_index=True)
        split_df['class_normalized'] = split_df['class'].apply(normalize_class_names)

        out_path = f"{OUTPUT_DIR}/{split_name}_balanced_v6.csv"
        split_df.to_csv(out_path, index=False)

        total = len(split_df)
        print(f"\n  {split_name.upper()}: {total:,} rows saved → {out_path}")
        print("  Scenario distribution:")
        for s in SIMPLIFIED_SCENARIOS:
            c   = (split_df['scenario_simplified'] == s).sum()
            pct = 100 * c / total if total else 0
            print(f"    {s:<22}: {pct:5.1f}%")

        stats[split_name] = {'rows': total, 'trajectories': len(selections)}

    with open(f"{OUTPUT_DIR}/split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved split_stats.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("SAMPLING PIPELINE V6 — 5 Hz (displacement-filtered, trajectory-counted)")
    print(f"  MIN_FOCAL_DISPLACEMENT_M = {MIN_FOCAL_DISPLACEMENT_M}m")
    print(f"  TARGET_TRAIN_TRAJECTORIES = {TARGET_TRAIN_TRAJECTORIES:,}")
    print(f"  Stationary weight = {get_scenario_weights()['stationary']:.0%}")
    print(f"  Output dir: {OUTPUT_DIR}")

    merged_df = pd.read_csv(MERGED_CSV)

    # If UTM columns absent but xCenter/yCenter present, compute UTM
    # (needed for displacement check; origin 0,0 is fine for relative distance)
    if 'utm_x' not in merged_df.columns and 'xCenter' in merged_df.columns:
        merged_df['utm_x'] = merged_df['xCenter']
        merged_df['utm_y'] = merged_df['yCenter']

    merged_df, traj_df = prepare_trajectory_data(merged_df)

    train_pool, val_pool, test_pool = perform_global_split(traj_df)

    train_sel = sample_from_pool(train_pool, TARGET_TRAIN_TRAJECTORIES, 'train')
    val_sel   = sample_from_pool(val_pool,   TARGET_VAL_TRAJECTORIES,   'val')
    test_sel  = sample_from_pool(test_pool,  TARGET_TEST_TRAJECTORIES,  'test')

    verify_and_save(merged_df, {
        'train': train_sel,
        'val':   val_sel,
        'test':  test_sel,
    })

    print("\nSampling complete.")


if __name__ == "__main__":
    main()
