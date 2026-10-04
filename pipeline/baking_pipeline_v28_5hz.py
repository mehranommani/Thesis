"""
================================================================================
BAKING PIPELINE V28 — MULTI-CONFIG EXPERIMENT (5 Hz)
================================================================================

Changes from V27 (V28):

  BUG FIX — HEADING UNIT CORRECTION (CRITICAL):
    The inD 'heading' column is stored in degrees (range ~1.5–358.7°).
    V27 used it directly without conversion, corrupting heading_rate
    (wrapping degrees as radians), turn_direction thresholds, and the
    Graphormer normalisation (dividing degrees by π gave ~0–114 instead
    of [-1, 1]).  Fix: compute_kinematics_for_recording() now converts
    heading from degrees to radians immediately after reading the inD
    column.  All downstream computations (heading_rate, curvature,
    classify_window thresholds, normalisation) are now unit-consistent.

  SIMPLIFICATION — engineer_features() and build_lookups_sampled():
    trajectory_lookup_sampled is used in find_valid_samples() only for
    data['frames'] and data['class'].  The full kinematic recomputation
    previously done by engineer_features() (velocity, heading, accel,
    curvature, all 15 features) was entirely wasted — model features
    always come from trajectory_lookup_original.
    engineer_features() now only computes UTM coordinates and preserves
    class/frame columns.  build_lookups_sampled() stores only frames
    and class, dropping kinematics and the unused frame_to_idx return.

  CLEANUP — stale FIX 6 comments removed from exporters.
    Both SGCN and STGCNN exporters had contradictory comments
    describing a normalization that was added and then reverted.

Changes from V26 (V27):
  WINDOW-LEVEL SCENARIO CLASSIFICATION:
    find_valid_samples() now classifies each candidate obs window
    directly from its kinematic slice (speed, heading_rate, is_stopping,
    displacement) rather than inheriting the track-level label.
    This fixes the granularity mismatch: a pre-turn straight window
    is now correctly labeled 'straight' instead of 'turning_maneuver'.
    Classification uses the same 5-category taxonomy and similar
    thresholds as merge_and_analyze_csvs.py but operates entirely
    on the precomputed kinematics array — no re-import needed.
    Config-aware: thresholds are applied to obs_len frames at 5 Hz.

  SAMPLE SIZES INCREASED:
    N_TRAIN = 5000  (was 1000) — benchmark models were below CV baseline
    N_VAL   = 1000  (was 250)  — 250 caused STGCNN early-stopping 20-40 epochs early
    N_TEST  = 1000  (was 250)  — larger test → tighter ADE/FDE confidence
    Split ≈ 71/14/14 — close to standard 70/15/15.

Changes from V24 (V26 — carried through):
  FIX 6 REVERTED: coordinate center-normalization removed from baking.
    SGCN and STGCNN operate on frame-to-frame displacements — subtracting
    a constant origin has zero effect on displacement values.
    Graphormer centers at runtime in the train loop (focal_pos subtraction)
    which is the correct place. Baking raw UTM avoids broken visualization
    and removes needless complexity (focal_utm columns, rel_to_abs helpers).
    Raw UTM stored directly — utm_to_pixel() works without any conversion.

Changes from V23 (V24):
1.  FIX 6 — Center-normalize all (x, y) coordinates per sample before writing.
    Origin = focal agent UTM position at end_obs_frame.
    SGCN and STGCNN feathers now store coordinates relative to that origin.
    ADE/FDE are translation-invariant — metrics unaffected.
    Removes raw UTM scale (~314000, 5627000) from model inputs entirely.

2.  FIX 4 — Graphormer node features normalized before saving .pt files.
    Raw kinematic dims span wildly different scales (speed 0-20 m/s,
    heading ±π, accel 0-5 m/s²). Normalization prevents large-magnitude dims
    from dominating transformer attention. Temporal displacement dims /10.
    Lane nodes have zeros in kinematic slots — safe to scale.

Changes from V22 (carried through V23):
1.  Stratified window sampling by scenario type.
    find_valid_samples() now tags each window with its scenario label.
    stratified_subsample() draws proportionally across scenario bins
    (turning=27%, straight=27%, lane_change=25%, stopping=16%, stationary=5%)
    with within-bin shuffling for trajectory diversity.
    Previously random.sample() over-represented turning/straight (long
    trajectories = many windows) and under-represented stopping behavior.

2.  focal_agent_id column added to ALL feather outputs (SGCN and STGCNN).
    Column value = the synthetic agent_id / ped_id of the focal agent for that
    sample_idx.  Evaluation and visualization can now look up the focal agent
    directly without UTM proximity heuristics or assuming ped=0.

3.  Focal agent written FIRST in agents_to_include list.
    The focal agent is always the first row for each sample_idx in the feather,
    so ped=0 works as a fallback when focal_agent_id is not used.

4.  Increased sample sizes:
        N_TRAIN = 5000  (was 1500)
        N_VAL   = 1000  (was 300)
        N_TEST  = 1000  (was 300)

5.  BALANCED_DATA_DIR updated to "data/sampled_balanced" (was "small_sample").

6.  All three models use the SAME sample draw per config (same seed) to ensure
    index alignment between feather files and Graphormer .pt files.
================================================================================
"""

import os
import sys
import time
import json
import random
import warnings
import gc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
from collections import defaultdict

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

GRAPHORMER_PYG_PATH = "script/graphormer-pyg"
sys.path.insert(0, GRAPHORMER_PYG_PATH)

from graphormer.functional import (
    precalculate_custom_attributes,
    precalculate_paths
)
print(f"✓ Imported graphormer-pyg from: {GRAPHORMER_PYG_PATH}")

try:
    from lane_graph_builder import LaneGraphBuilder
    LANE_GRAPH_AVAILABLE = True
except ImportError:
    LANE_GRAPH_AVAILABLE = False

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# EXPERIMENT CONFIGS
# ============================================================================

EXPERIMENT_CONFIGS = [
    (3, 1),
    (4, 2),
    (5, 3),
]

FPS = 5

NUM_TEMPORAL_SAMPLES = 5
GRAPHORMER_NODE_DIM  = 19 + NUM_TEMPORAL_SAMPLES * 2   # = 29

# Minimum displacement of the focal agent over the observation window (metres).
# Windows below this threshold are excluded from the baked pool entirely.
MIN_FOCAL_DISPLACEMENT_M = 2.0


def compute_tsn_temporal_offsets(obs_len, K=NUM_TEMPORAL_SAMPLES):
    offsets = []
    for k in range(1, K + 1):
        offset = round(obs_len * k / K)
        offsets.append(min(offset, obs_len - 1))
    return offsets


def config_tag(obs_sec, pred_sec):
    return f"obs{obs_sec}_pred{pred_sec}"


print(f"FPS: {FPS}  |  Node dim: {GRAPHORMER_NODE_DIM}")
print(f"MIN_FOCAL_DISPLACEMENT_M: {MIN_FOCAL_DISPLACEMENT_M}m")
print(f"Experiment configs: {EXPERIMENT_CONFIGS}")

# ============================================================================
# SAMPLE SIZES  (increased from V21)
# ============================================================================

N_TRAIN = 5000
N_VAL   = 1000
N_TEST  = 1000

# ============================================================================
# PATHS
# ============================================================================

BALANCED_DATA_DIR  = "data/sampled_balanced"       # ← fixed from "small_sample"
ROOT_DIR           = "data/inD-dataset-v1"
RECORDING_META_PATH = f"{ROOT_DIR}/data"
LANE_GRAPH_DIR     = "lane_graphs"

AGGREGATED_TRACKS_5HZ_PATH = f"{ROOT_DIR}/data_output_5hz/aggregated_tracks.csv"
AGGREGATED_META_5HZ_PATH   = f"{ROOT_DIR}/data_output_5hz/aggregated_tracksMeta.csv"

MULTICLASS_OUTPUT_DIR = "script/Multiclass-SGCN/datasets/inD_test2"
GRAPHORMER_OUTPUT_DIR = "script/graphormer-pyg/datasets/inD_graphs_test2"
STGCNN_OUTPUT_DIR     = "script/Social-STGCNN/datasets/inD/inD_benchmark_test2"

os.makedirs(MULTICLASS_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPHORMER_OUTPUT_DIR, exist_ok=True)
os.makedirs(STGCNN_OUTPUT_DIR,     exist_ok=True)

RECORDING_META_OUTPUT_PATH = f"{ROOT_DIR}/data_output_5hz/recording_meta.json"
BACKGROUND_IMAGE_ROOT      = f"{ROOT_DIR}/data"

MAX_IN_DEGREE     = 32
MAX_OUT_DEGREE    = 32
MAX_PATH_DISTANCE = 10

AGENT_EDGE_DISTANCE_THRESHOLD  = 30.0
AGENT_LANE_DISTANCE_THRESHOLD  = 30.0
LANE_PROXIMITY_THRESHOLD       = 50.0
NEIGHBOR_DISTANCE_THRESHOLD    = 50.0

# ============================================================================
# CLASS MAPPING
# ============================================================================

MULTICLASS_LABELS = ["Car", "Pedestrian", "Bicycle", "Truck_Bus"]

IND_TO_MULTICLASS_CLASS = {
    'car': 'Car', 'Car': 'Car',
    'pedestrian': 'Pedestrian', 'Pedestrian': 'Pedestrian',
    'bicycle': 'Bicycle', 'Bicycle': 'Bicycle', 'bike': 'Bicycle',
    'truck_bus': 'Truck_Bus', 'truck': 'Truck_Bus', 'bus': 'Truck_Bus',
    'Truck': 'Truck_Bus', 'Bus': 'Truck_Bus',
}

RECORDING_ID_TO_MAP_KEY = {}
for i in range(0, 33):
    if   0 <= i <=  6: RECORDING_ID_TO_MAP_KEY[i] = "aseag"
    elif 7 <= i <= 10: RECORDING_ID_TO_MAP_KEY[i] = "bendplatz"
    elif 11 <= i <= 17: RECORDING_ID_TO_MAP_KEY[i] = "bendplatz_construction"
    elif 18 <= i <= 29: RECORDING_ID_TO_MAP_KEY[i] = "frankenburg"
    elif 30 <= i <= 32: RECORDING_ID_TO_MAP_KEY[i] = "heckstrasse"

# ============================================================================
# UTILITIES  (unchanged from V21)
# ============================================================================

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_heading_smooth(vx_arr, vy_arr, min_speed=1.0):
    speeds = np.sqrt(vx_arr**2 + vy_arr**2)
    raw_headings = np.arctan2(vy_arr, vx_arr)
    valid_mask = speeds > min_speed
    if not np.any(valid_mask):
        return np.zeros_like(raw_headings)
    x  = np.arange(len(raw_headings))
    xp = x[valid_mask]
    fp = np.unwrap(raw_headings[valid_mask])
    return wrap_angle(np.interp(x, xp, fp))


# ============================================================================
# LOADERS  (unchanged from V21)
# ============================================================================

def get_origins_from_meta_files(meta_path):
    print("\n--- Loading Recording Metadata ---")
    location_origins      = {}
    recording_to_location = {}
    recording_meta_full   = {}

    FALLBACK_ORIGINS = {
        1: (314984.17, 5627720.76),
        2: (313788.40, 5630736.62),
        3: (313788.40, 5630736.62),
        4: (313618.02, 5628579.57),
        5: (313993.02, 5629899.76),
    }

    for rec_id in range(0, 33):
        meta_file = f"{meta_path}/{rec_id:02d}_recordingMeta.csv"
        if not os.path.exists(meta_file):
            continue
        try:
            meta_df      = pd.read_csv(meta_file)
            loc_id       = int(meta_df['locationId'].iloc[0])
            x_utm_origin = float(meta_df['xUtmOrigin'].iloc[0])
            y_utm_origin = float(meta_df['yUtmOrigin'].iloc[0])
            ortho_px2m   = float(meta_df['orthoPxToMeter'].iloc[0]) \
                           if 'orthoPxToMeter' in meta_df.columns else 0.0254

            location_origins[loc_id]      = (x_utm_origin, y_utm_origin)
            recording_to_location[rec_id] = loc_id
            recording_meta_full[rec_id]   = {
                'xUtmOrigin':           x_utm_origin,
                'yUtmOrigin':           y_utm_origin,
                'orthoPxToMeter':       ortho_px2m,
                'locationId':           loc_id,
                'background_image_path':
                    f"{BACKGROUND_IMAGE_ROOT}/{rec_id:02d}_background.png",
            }
        except Exception as exc:
            print(f"  WARNING: {meta_file}: {exc}")

    for loc, org in FALLBACK_ORIGINS.items():
        if loc not in location_origins:
            location_origins[loc] = org

    print(f"  Origins for {len(location_origins)} locations, "
          f"full meta for {len(recording_meta_full)} recordings")
    return location_origins, recording_to_location, recording_meta_full


def save_recording_metadata_json(recording_meta_full, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as fh:
        json.dump({str(k): v for k, v in recording_meta_full.items()},
                  fh, indent=2)
    print(f"  -> Saved recording metadata: {output_path} "
          f"({len(recording_meta_full)} recordings)")


def load_lane_graphs():
    lane_graphs = {}
    if not LANE_GRAPH_AVAILABLE:
        return {}
    for map_name in ["bendplatz", "frankenburg", "heckstrasse", "aseag",
                     "bendplatz_construction"]:
        path = os.path.join(LANE_GRAPH_DIR, f"{map_name}_lane_graph.pkl")
        if os.path.exists(path):
            try:
                lane_graphs[map_name] = LaneGraphBuilder.load(path)
            except Exception:
                pass
    print(f"  Loaded {len(lane_graphs)} lane graphs")
    return lane_graphs


def precompute_lane_data(lane_graphs):
    lane_cache = {}
    for map_name, lg in lane_graphs.items():
        graph_data = lg.lane_graph
        lane_cache[map_name] = {'lanes': {}, 'successors': {}}
        for lane_id, l_data in graph_data.items():
            lane_cache[map_name]['lanes'][lane_id] = {
                'start':   np.array(l_data['start'], dtype=np.float32),
                'heading': l_data['heading_start'],
                'length':  l_data['length'],
            }
            lane_cache[map_name]['successors'][lane_id] = l_data['successors']
    return lane_cache


def compute_kinematics_for_recording(df):
    if 'xVelocity' in df.columns:
        df['utm_vx'] = df['xVelocity']
        df['utm_vy'] = df['yVelocity']
    else:
        gb = df.groupby('trackId')
        df['dt'] = gb['frame'].diff().fillna(1.0) / float(FPS)
        df['dt'] = df['dt'].replace(0, 1.0 / float(FPS))
        df['utm_vx'] = gb['utm_x'].diff().fillna(0) / df['dt']
        df['utm_vy'] = gb['utm_y'].diff().fillna(0) / df['dt']

    df['speed'] = np.sqrt(df['utm_vx']**2 + df['utm_vy']**2)

    if 'heading' not in df.columns:
        # Fallback: derive heading from velocity direction (radians)
        smooth_h = []
        for _, group in df.groupby('trackId'):
            smooth_h.append(compute_heading_smooth(
                group['utm_vx'].values, group['utm_vy'].values))
        df['heading'] = np.concatenate(smooth_h)
    else:
        # inD heading column is in degrees — convert to radians so that
        # all downstream computations (heading_rate wrapping, turn_direction
        # threshold, classify_window thresholds, Graphormer /π normalisation)
        # operate in consistent units.
        df['heading'] = np.deg2rad(df['heading'])

    if 'xAcceleration' in df.columns:
        df['utm_ax'] = df['xAcceleration']
        df['utm_ay'] = df['yAcceleration']
    else:
        gb = df.groupby('trackId')
        df['dt'] = gb['frame'].diff().fillna(1.0) / float(FPS)
        df['dt'] = df['dt'].replace(0, 1.0 / float(FPS))
        df['utm_ax'] = gb['utm_vx'].diff().fillna(0) / df['dt']
        df['utm_ay'] = gb['utm_vy'].diff().fillna(0) / df['dt']

    gb = df.groupby('trackId')
    df['dt'] = gb['frame'].diff().fillna(1.0) / float(FPS)
    df['dt'] = df['dt'].replace(0, 1.0 / float(FPS))

    d_heading = gb['heading'].diff().fillna(0)
    d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
    df['heading_rate'] = d_heading / df['dt']

    eps = 1e-5
    cross = df['utm_vx'] * df['utm_ay'] - df['utm_vy'] * df['utm_ax']
    df['curvature'] = (cross / (df['speed']**3 + eps)).clip(-1.0, 1.0)

    decel = np.sqrt(df['utm_ax']**2 + df['utm_ay']**2)
    dot   = df['utm_vx'] * df['utm_ax'] + df['utm_vy'] * df['utm_ay']
    is_dec = dot < -0.1
    df['time_to_stop'] = np.where(
        is_dec & (decel > 0.1), df['speed'] / (decel + eps), 99.0
    ).clip(0, 10)
    df['is_stopping']  = ((df['time_to_stop'] < 5.0) & is_dec).astype(np.float32)
    df['is_low_speed'] = (df['speed'] < 1.0).astype(np.float32)
    df['turn_direction'] = np.where(
        df['heading_rate'].abs() > 0.15, np.sign(df['heading_rate']), 0.0)

    df['width']  = df.get('width',  pd.Series(1.8, index=df.index)).fillna(1.8)
    df['length'] = df.get('length', pd.Series(4.5, index=df.index)).fillna(4.5)

    class_map = {
        'car': 0, 'Car': 0,
        'truck_bus': 1, 'truck': 1, 'bus': 1, 'Truck': 1, 'Bus': 1,
        'bicycle': 2, 'bike': 2, 'Bicycle': 2,
        'pedestrian': 3, 'Pedestrian': 3,
    }
    df['class_int'] = df['class'].map(class_map).fillna(0).astype(np.float32) \
        if 'class' in df.columns else 0.0

    for col in ['utm_vx', 'utm_vy', 'speed', 'heading', 'heading_rate',
                'curvature', 'utm_ax', 'utm_ay', 'time_to_stop', 'is_stopping']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    df['heading_change_1s'] = 0.0
    df['in_turn'] = 0.0
    return df


def load_original_tracks_with_class(recording_ids, location_origins,
                                     recording_to_location):
    print("\n--- Loading Original 5Hz Track Data ---")
    all_tracks = {}
    class_map  = {}
    unique_recs = sorted(set(recording_ids))

    df_all      = pd.read_csv(AGGREGATED_TRACKS_5HZ_PATH)
    df_meta_all = pd.read_csv(AGGREGATED_META_5HZ_PATH)

    if 'class' in df_meta_all.columns:
        for _, row in df_meta_all.iterrows():
            class_map[(int(row['recordingId']), int(row['trackId']))] = str(row['class'])
        df_all = df_all.merge(
            df_meta_all[['recordingId', 'trackId', 'class']],
            on=['recordingId', 'trackId'], how='left')

    if 'class' not in df_all.columns:
        df_all['class'] = 'car'
    df_all['class'] = df_all['class'].fillna('car')
    df_all = df_all[df_all['recordingId'].isin(unique_recs)].copy()

    for rec_id in unique_recs:
        loc_id = recording_to_location.get(rec_id, 1)
        origin = location_origins.get(loc_id, (0, 0))
        mask = df_all['recordingId'] == rec_id
        df_all.loc[mask, 'utm_x'] = df_all.loc[mask, 'xCenter'] + origin[0]
        df_all.loc[mask, 'utm_y'] = df_all.loc[mask, 'yCenter'] + origin[1]

    for rec_id in tqdm(unique_recs, desc="Computing kinematics"):
        df_rec = df_all[df_all['recordingId'] == rec_id].copy()
        all_tracks[rec_id] = compute_kinematics_for_recording(df_rec)

    del df_all
    gc.collect()

    kin_cols = [
        'utm_vx', 'utm_vy', 'speed', 'heading', 'heading_rate',
        'curvature', 'utm_ax', 'utm_ay', 'time_to_stop', 'is_stopping',
        'is_low_speed', 'turn_direction', 'heading_change_1s', 'in_turn', 'class_int'
    ]

    frame_lookup          = defaultdict(list)
    trajectory_lookup     = {}
    frame_to_idx_lookup   = {}

    for rec_id, df in tqdm(all_tracks.items(), desc="Building lookups"):
        for _, row in df.iterrows():
            frame_lookup[(rec_id, int(row['frame']))].append(int(row['trackId']))

        for track_id, group in df.groupby('trackId'):
            group = group.sort_values('frame')
            cl    = class_map.get((rec_id, track_id), 'car')
            trajectory_lookup[(rec_id, track_id)] = {
                'frames':          group['frame'].values,
                'utm_xy':          group[['utm_x', 'utm_y']].values.astype(np.float32),
                'kinematics':      group[kin_cols].values.astype(np.float32),
                'static_features': group[['width', 'length']].iloc[0].values.astype(np.float32),
                'class':           cl,
            }
            frame_to_idx_lookup[(rec_id, track_id)] = {
                f: i for i, f in enumerate(group['frame'].values)
            }

    print(f"  Loaded {len(trajectory_lookup):,} agent trajectories")
    return frame_lookup, trajectory_lookup, frame_to_idx_lookup, class_map


def engineer_features(df, location_origins):
    """
    Minimal feature engineering for the sampled CSV.

    trajectory_lookup_sampled (built from this output) is used in
    find_valid_samples() only for data['frames'] and data['class'].
    All window-level kinematics come from trajectory_lookup_original.

    This function therefore only needs to:
      1. Add UTM coordinates (needed to build all_rec_ids list).
      2. Preserve frame, class, recordingId, trackId columns.
    """
    if 'locationId' not in df.columns:
        df['locationId'] = 1

    df['xUtmOrigin'] = df['locationId'].map(
        lambda x: location_origins.get(int(x), (0, 0))[0])
    df['yUtmOrigin'] = df['locationId'].map(
        lambda x: location_origins.get(int(x), (0, 0))[1])
    df['utm_x'] = df['xCenter'] + df['xUtmOrigin']
    df['utm_y'] = df['yCenter'] + df['yUtmOrigin']

    if 'class' not in df.columns:
        df['class'] = 'car'

    return df


def build_lookups_sampled(df):
    """
    Build a minimal lookup from the sampled CSV.

    find_valid_samples() only needs data['frames'] and data['class']
    from this lookup — all kinematic features come from
    trajectory_lookup_original.  Storing only what is used saves
    significant memory and eliminates redundant computation.
    """
    trajectory_lookup = {}
    for (rec_id, track_id), group in df.groupby(['recordingId', 'trackId']):
        cl = group['class'].iloc[0] if 'class' in group.columns else 'car'
        trajectory_lookup[(rec_id, track_id)] = {
            'frames': group.sort_values('frame')['frame'].values,
            'class':  cl,
        }
    return trajectory_lookup


# ============================================================================
# find_valid_samples  — V23: displacement filter + scenario tag
# ============================================================================


# ============================================================================
# WINDOW-LEVEL SCENARIO CLASSIFICATION  — V27
#
# Classifies each candidate obs window from its kinematic slice directly.
# Fixes track-level label inheritance: a 3s straight approach before a turn
# is now labeled 'straight' (correct) not 'turning_maneuver' (wrong).
#
# Uses same 5-category taxonomy as merge_and_analyze_csvs.py:
#   turning_maneuver  — accumulated heading change > threshold
#   lane_change       — lateral body-frame shift, low heading change
#   stopping_behavior — sustained low speed in obs window
#   straight          — default moving forward
#   stationary        — barely moves
#
# Kinematic array column indices (from kin_cols in load_original_tracks):
#   0: utm_vx   1: utm_vy   2: speed   3: heading   4: heading_rate
#   5: curvature  6: utm_ax  7: utm_ay  8: time_to_stop  9: is_stopping
#  10: is_low_speed  11: turn_direction  12: heading_change_1s
#  13: in_turn  14: class_int
# ============================================================================

KIN_IDX = {
    'speed': 2, 'heading': 3, 'heading_rate': 4,
    'is_stopping': 9, 'is_low_speed': 10,
    'utm_vx': 0, 'utm_vy': 1,
}

def classify_window(utm_xy_slice, kin_slice, agent_class='car'):
    """
    Classify a single obs window from its position and kinematic arrays.

    Parameters
    ----------
    utm_xy_slice : np.ndarray [obs_len, 2]  — UTM positions for obs window
    kin_slice    : np.ndarray [obs_len, 15] — precomputed kinematics
    agent_class  : str — 'car', 'pedestrian', 'bicycle', 'truck_bus'

    Returns
    -------
    str : one of 'straight', 'turning_maneuver', 'lane_change',
                 'stopping_behavior', 'stationary'

    Design notes
    ------------
    - Uses the same priority chain as merge_and_analyze_csvs.py:
        stationary → turning → lane_change → stopping → straight
    - Thresholds are slightly more permissive than track-level because
      a short window (3-5s) has less heading accumulation than a full track.
    - No dependency on external scripts — uses only the precomputed
      kinematics that are already stored in trajectory_lookup_original.
    """
    obs_len = len(utm_xy_slice)
    if obs_len < 3:
        return 'straight'

    speeds       = kin_slice[:, KIN_IDX['speed']]
    headings     = kin_slice[:, KIN_IDX['heading']]
    heading_rate = kin_slice[:, KIN_IDX['heading_rate']]
    is_stopping  = kin_slice[:, KIN_IDX['is_stopping']]

    # ── Per-class threshold adjustment ────────────────────────────────────
    # Cars: standard thresholds
    # VRUs (pedestrian/bicycle): higher heading threshold (more natural noise)
    # Trucks: slightly higher heading threshold (wider turns)
    if agent_class in ('pedestrian', 'bicycle'):
        turn_thresh_deg   = 60.0
        lc_heading_max    = np.deg2rad(25.0)
        lc_lateral_min    = 1.5    # metres — narrower agents
        stop_speed_thresh = 0.2    # m/s
        stat_disp_thresh  = 0.3    # metres
        stat_speed_thresh = 0.1    # m/s
    elif agent_class in ('truck_bus', 'truck', 'bus'):
        turn_thresh_deg   = 50.0
        lc_heading_max    = np.deg2rad(30.0)
        lc_lateral_min    = 2.5
        stop_speed_thresh = 0.3
        stat_disp_thresh  = 0.5
        stat_speed_thresh = 0.15
    else:  # car (default)
        turn_thresh_deg   = 40.0   # lower than track-level 45° — window is shorter
        lc_heading_max    = np.deg2rad(30.0)
        lc_lateral_min    = 2.0
        stop_speed_thresh = 0.3
        stat_disp_thresh  = 0.5
        stat_speed_thresh = 0.15

    turn_thresh_rad = np.deg2rad(turn_thresh_deg)

    # ── 1. STATIONARY ─────────────────────────────────────────────────────
    total_disp  = float(np.linalg.norm(utm_xy_slice[-1] - utm_xy_slice[0]))
    mean_speed  = float(np.mean(speeds))
    if total_disp < stat_disp_thresh and mean_speed < stat_speed_thresh:
        return 'stationary'

    # ── 2. TURNING: accumulated unsigned heading change ───────────────────
    # Wrap each step to [-π, π] before summing to avoid 2π jumps
    heading_diffs = []
    for t in range(1, obs_len):
        diff = headings[t] - headings[t - 1]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
        heading_diffs.append(abs(diff))
    total_unsigned_heading = float(np.sum(heading_diffs))

    if total_unsigned_heading > turn_thresh_rad:
        # Sub-classify: stopping while turning = stop_at_turn → stopping_behavior
        stop_frac = float(np.mean(is_stopping))
        if stop_frac > 0.25:   # >25% of obs frames flagged as stopping
            return 'stopping_behavior'
        return 'turning_maneuver'

    # ── 3. LANE CHANGE: lateral body-frame displacement ───────────────────
    # Only check if heading change is small (not a turn)
    if total_unsigned_heading <= lc_heading_max:
        if obs_len >= 6:
            initial_heading = float(headings[0])
            cos_h = np.cos(-initial_heading)
            sin_h = np.sin(-initial_heading)
            rel_pos = utm_xy_slice - utm_xy_slice[0]
            body_y  = rel_pos[:, 0] * sin_h + rel_pos[:, 1] * cos_h
            max_lat_left  = float(np.max(body_y))
            max_lat_right = float(-np.min(body_y))
            max_lat = max(max_lat_left, max_lat_right)
            if max_lat > lc_lateral_min:
                return 'lane_change'

    # ── 4. STOPPING BEHAVIOR: sustained low speed ─────────────────────────
    # At least 30% of obs frames must be in stopping state
    stop_frac = float(np.mean(is_stopping))
    if stop_frac > 0.30:
        return 'stopping_behavior'
    # Also check if agent slows significantly during obs window
    if mean_speed > 0.3 and float(np.min(speeds)) < stop_speed_thresh:
        speed_drop_ratio = (float(np.max(speeds)) - float(np.min(speeds))) / (mean_speed + 1e-5)
        if speed_drop_ratio > 0.6:
            return 'stopping_behavior'

    # ── 5. DEFAULT: STRAIGHT ──────────────────────────────────────────────
    return 'straight'


def find_valid_samples(trajectory_lookup, trajectory_lookup_original,
                       frame_to_idx_lookup_original,
                       obs_len, window_len,
                       min_displacement_m=MIN_FOCAL_DISPLACEMENT_M):
    """
    Find all valid windows and classify each one at WINDOW LEVEL — V27.

    V27 change: scenario label is now computed from the obs-window kinematics
    directly (using classify_window()), not inherited from the full-track label.
    This fixes the granularity mismatch where a straight pre-turn approach was
    incorrectly labeled 'turning_maneuver' based on the full track label.

    The classification is config-aware: it uses exactly obs_len frames at 5 Hz,
    so a 3s window and a 5s window of the same physical approach will be
    classified based on what actually happens in those frames.

    Returns 4-tuples: (rec_id, track_id, end_obs_frame, window_scenario)
    """
    valid_samples  = []
    skipped_static = 0

    for (rec_id, track_id), data in trajectory_lookup.items():
        frames      = data['frames']
        agent_class = data.get('class', 'car')

        if len(frames) < window_len:
            continue

        split_indices = np.where(np.diff(frames) != 1)[0] + 1
        segments = np.split(frames, split_indices)

        orig_traj   = trajectory_lookup_original.get((rec_id, track_id))
        idx_map     = frame_to_idx_lookup_original.get((rec_id, track_id), {})

        for segment in segments:
            if len(segment) < window_len:
                continue
            for i in range(len(segment) - window_len + 1):
                start_obs_frame = segment[i]
                end_obs_frame   = segment[i + obs_len - 1]

                if orig_traj is None:
                    continue

                idx_start = idx_map.get(start_obs_frame)
                idx_end   = idx_map.get(end_obs_frame)
                if idx_start is None or idx_end is None:
                    continue

                # ── Displacement filter (unchanged) ───────────────────────
                pos_start = orig_traj['utm_xy'][idx_start]
                pos_end   = orig_traj['utm_xy'][idx_end]
                disp = float(np.linalg.norm(pos_end - pos_start))
                if disp < min_displacement_m:
                    skipped_static += 1
                    continue

                # ── V27: window-level scenario classification ──────────────
                # Extract obs-window slices from precomputed arrays.
                # This gives the scenario label for exactly these obs_len frames,
                # not the full track — fixes track-level label inheritance bug.
                utm_xy_slice = orig_traj['utm_xy'][idx_start:idx_end + 1]
                kin_slice    = orig_traj['kinematics'][idx_start:idx_end + 1]

                window_scenario = classify_window(
                    utm_xy_slice, kin_slice, agent_class=agent_class
                )

                valid_samples.append(
                    (rec_id, track_id, end_obs_frame, window_scenario)
                )

    total = len(valid_samples) + skipped_static
    print(f"    Valid samples: {len(valid_samples):,}  |  "
          f"Skipped (static <{min_displacement_m}m): {skipped_static:,}  |  "
          f"Total windows checked: {total:,}  |  "
          f"Moving fraction: {100*len(valid_samples)/max(1,total):.1f}%")

    # Print window-level scenario distribution
    bin_counts = defaultdict(int)
    for _, _, _, sc in valid_samples:
        bin_counts[sc] += 1
    print(f"    Window scenario pool (obs_len={obs_len}f): " +
          "  ".join(f"{k}={v:,}" for k, v in sorted(bin_counts.items())))

    return valid_samples


# ============================================================================
# STRATIFIED SUBSAMPLE  — V23
#
# Draws n windows from the tagged pool respecting scenario weights.
# Each window is a 4-tuple (rec_id, track_id, end_obs_frame, scenario).
# Returns 3-tuples (scenario tag stripped) for exporters — no downstream
# changes needed.
# ============================================================================

# Target proportions for the final drawn set.
# Stationary gets 0 because displacement filter removes all stationary windows.
# Weights are normalised inside the function over bins that actually exist.
SCENARIO_DRAW_WEIGHTS = {
    'turning_maneuver':  0.27,
    'straight':          0.27,
    'lane_change':       0.25,
    'stopping_behavior': 0.16,
    'stationary':        0.05,
}

def stratified_subsample(pool, n, seed=SEED):
    """
    Draw n windows from pool with scenario-stratified sampling.

    Parameters
    ----------
    pool : list of (rec_id, track_id, end_obs_frame, scenario)
    n    : total number of windows to draw
    seed : random seed for reproducibility

    Returns
    -------
    list of (rec_id, track_id, end_obs_frame)  — scenario tag stripped
    """
    # Group windows by scenario
    bins = defaultdict(list)
    for item in pool:
        bins[item[3]].append(item)

    # Normalise weights over bins that are actually populated
    available = {sc: wins for sc, wins in bins.items() if wins}
    total_w   = sum(SCENARIO_DRAW_WEIGHTS.get(sc, 0.0) for sc in available)

    if total_w == 0:
        # Fallback: uniform draw (should never happen in practice)
        random.seed(seed)
        chosen = random.sample(pool, min(n, len(pool)))
        return [(r, t, f) for r, t, f, _ in chosen]

    selected = []
    rng = random.Random(seed)

    for sc, wins in sorted(available.items()):
        w         = SCENARIO_DRAW_WEIGHTS.get(sc, 0.0)
        bin_n     = round(n * w / total_w)
        bin_n     = min(bin_n, len(wins))   # can't draw more than available
        shuffled  = wins.copy()
        rng.shuffle(shuffled)               # within-bin shuffle for diversity
        selected.extend(shuffled[:bin_n])

    # Top-up or trim to exactly n (rounding may cause ±few)
    deficit = n - len(selected)
    if deficit > 0:
        # Fill from largest bins first
        remainder = [w for w in pool if w not in set(selected)]
        rng.shuffle(remainder)
        selected.extend(remainder[:deficit])
    elif deficit < 0:
        rng.shuffle(selected)
        selected = selected[:n]

    # Print final scenario distribution of drawn set
    drawn_counts = defaultdict(int)
    for item in selected:
        drawn_counts[item[3]] += 1
    print(f"    Stratified draw ({len(selected):,} total):")
    for sc in sorted(drawn_counts):
        pct = 100 * drawn_counts[sc] / len(selected)
        print(f"      {sc:<22}: {drawn_counts[sc]:>5,}  ({pct:.1f}%)")

    # Strip scenario tag — exporters expect 3-tuples
    return [(r, t, f) for r, t, f, _ in selected]


# ============================================================================
# EXPORTER 1: MULTICLASS-SGCN
# Feather columns: frame, agent_id, x, y, class, rec_id, sample_idx,
#                  focal_agent_id   ← NEW in V22
# Focal agent is always written FIRST in agents_to_include.
# ============================================================================

def export_for_multiclass_sgcnn(split_name, samples, frame_lookup_original,
                                 trajectory_lookup_original,
                                 frame_to_idx_lookup_original,
                                 obs_len, pred_len, cfg_tag):
    print(f"\n{'='*60}")
    print(f"[Exporter 1] Multiclass-SGCN  |  {cfg_tag}  |  {split_name}")
    print(f"  obs={obs_len}f  pred={pred_len}f  samples={len(samples):,}")
    print(f"{'='*60}")

    data = []
    samples_with_neighbors = 0
    total_agents_exported  = 0

    for sample_idx, (rec_id, focal_track_id, end_obs_frame) in enumerate(
            tqdm(samples, desc=f"SGCN {cfg_tag}/{split_name}")):

        start_frame = end_obs_frame - obs_len + 1
        end_frame   = end_obs_frame + pred_len

        focal_traj = trajectory_lookup_original.get((rec_id, focal_track_id))
        if focal_traj is None:
            continue

        focal_idx_map = frame_to_idx_lookup_original.get(
            (rec_id, focal_track_id), {})
        focal_obs_idx = focal_idx_map.get(end_obs_frame)
        if focal_obs_idx is None:
            continue
        focal_pos = focal_traj['utm_xy'][focal_obs_idx]

        all_agents = frame_lookup_original.get((rec_id, end_obs_frame), [])

        # Collect neighbours
        neighbors = []
        for agent_id in all_agents:
            if agent_id == focal_track_id:
                continue
            agent_traj = trajectory_lookup_original.get((rec_id, agent_id))
            if agent_traj is None:
                continue
            agent_idx_map = frame_to_idx_lookup_original.get(
                (rec_id, agent_id), {})
            idx = agent_idx_map.get(end_obs_frame)
            if idx is None:
                continue
            if np.linalg.norm(focal_pos - agent_traj['utm_xy'][idx]) \
                    > NEIGHBOR_DISTANCE_THRESHOLD:
                continue
            if all(f in agent_idx_map
                   for f in range(start_frame, end_frame + 1)):
                neighbors.append(agent_id)

        # FOCAL AGENT FIRST — guarantees ped=0 fallback works
        agents_to_include = [focal_track_id] + neighbors
        if len(agents_to_include) > 1:
            samples_with_neighbors += 1

        base_frame_id  = sample_idx * 1000
        focal_synthetic_id = sample_idx * 10000 + focal_track_id % 10000

        for agent_id in agents_to_include:
            agent_traj    = trajectory_lookup_original[(rec_id, agent_id)]
            agent_idx_map = frame_to_idx_lookup_original[(rec_id, agent_id)]
            class_label   = agent_traj.get('class', 'car')
            mapped_label  = IND_TO_MULTICLASS_CLASS.get(class_label, 'Car')
            unique_agent_id = sample_idx * 10000 + agent_id % 10000

            for rel_frame, orig_frame in enumerate(
                    range(start_frame, end_frame + 1)):
                idx  = agent_idx_map[orig_frame]
                x, y = agent_traj['utm_xy'][idx]
                # All agents in this sample share the same UTM origin so
                # pairwise distances (and thus GCN adjacency) are preserved.
                data.append([
                    base_frame_id + rel_frame,
                    unique_agent_id,
                    x, y,
                    mapped_label,
                    rec_id,
                    sample_idx,
                    focal_synthetic_id,
                ])
            total_agents_exported += 1

    df = pd.DataFrame(data, columns=[
        'frame', 'agent_id', 'x', 'y', 'class',
        'rec_id', 'sample_idx', 'focal_agent_id',
    ])
    out_dir = f"{MULTICLASS_OUTPUT_DIR}/{cfg_tag}/{split_name}"
    os.makedirs(out_dir, exist_ok=True)
    feather_path = f"{out_dir}/{split_name}.feather"
    df.to_feather(feather_path)
    print(f"  -> {len(df):,} rows → {feather_path}")
    print(f"  -> Total agents exported:   {total_agents_exported:,}")
    print(f"  -> Samples with neighbours: {samples_with_neighbors:,} "
          f"({100*samples_with_neighbors/max(1,len(samples)):.1f}%)")
    del data, df
    gc.collect()


# ============================================================================
# EXPORTER 2: SOCIAL-STGCNN
# Feather columns: frame, ped_id, x, y, rec_id, sample_idx, focal_agent_id
# Focal agent is written FIRST.
# ============================================================================

def export_for_stgcnn_benchmark(split_name, samples, frame_lookup_original,
                                 trajectory_lookup_original,
                                 frame_to_idx_lookup_original,
                                 obs_len, pred_len, cfg_tag):
    print(f"\n{'='*60}")
    print(f"[Exporter 2] Social-STGCNN  |  {cfg_tag}  |  {split_name}")
    print(f"  obs={obs_len}f  pred={pred_len}f  samples={len(samples):,}")
    print(f"{'='*60}")

    stgcnn_data = []
    samples_with_neighbors = 0
    total_agents_exported  = 0

    for sample_idx, (rec_id, focal_track_id, end_obs_frame) in enumerate(
            tqdm(samples, desc=f"STGCNN {cfg_tag}/{split_name}")):

        start_frame = end_obs_frame - obs_len + 1
        end_frame   = end_obs_frame + pred_len

        focal_traj = trajectory_lookup_original.get((rec_id, focal_track_id))
        if focal_traj is None:
            continue

        focal_idx_map = frame_to_idx_lookup_original.get(
            (rec_id, focal_track_id), {})
        focal_obs_idx = focal_idx_map.get(end_obs_frame)
        if focal_obs_idx is None:
            continue
        focal_pos = focal_traj['utm_xy'][focal_obs_idx]

        all_agents = frame_lookup_original.get((rec_id, end_obs_frame), [])

        neighbors = []
        for agent_id in all_agents:
            if agent_id == focal_track_id:
                continue
            agent_traj = trajectory_lookup_original.get((rec_id, agent_id))
            if agent_traj is None:
                continue
            agent_idx_map = frame_to_idx_lookup_original.get(
                (rec_id, agent_id), {})
            idx = agent_idx_map.get(end_obs_frame)
            if idx is None:
                continue
            if np.linalg.norm(focal_pos - agent_traj['utm_xy'][idx]) \
                    > NEIGHBOR_DISTANCE_THRESHOLD:
                continue
            if all(f in agent_idx_map
                   for f in range(start_frame, end_frame + 1)):
                neighbors.append(agent_id)

        # FOCAL AGENT FIRST
        agents_to_include = [focal_track_id] + neighbors
        if len(agents_to_include) > 1:
            samples_with_neighbors += 1

        base_frame_id      = sample_idx * 1000
        focal_synthetic_id = sample_idx * 10000 + focal_track_id % 10000

        for agent_id in agents_to_include:
            agent_traj    = trajectory_lookup_original[(rec_id, agent_id)]
            agent_idx_map = frame_to_idx_lookup_original[(rec_id, agent_id)]
            unique_agent_id = sample_idx * 10000 + agent_id % 10000

            for rel_frame, orig_frame in enumerate(
                    range(start_frame, end_frame + 1)):
                idx  = agent_idx_map[orig_frame]
                x, y = agent_traj['utm_xy'][idx]
                # Raw UTM stored; Graphormer centres at runtime,
                # SGCN/STGCNN use frame-to-frame displacements.
                stgcnn_data.append([
                    base_frame_id + rel_frame,
                    unique_agent_id,
                    x, y,
                    rec_id,
                    sample_idx,
                    focal_synthetic_id,
                ])
            total_agents_exported += 1

    df_stgcnn = pd.DataFrame(stgcnn_data, columns=[
        'frame', 'ped_id', 'x', 'y',
        'rec_id', 'sample_idx', 'focal_agent_id',
    ])
    df_stgcnn = df_stgcnn.sort_values(['frame', 'ped_id']).reset_index(drop=True)

    out_dir = f"{STGCNN_OUTPUT_DIR}/{cfg_tag}/{split_name}"
    os.makedirs(out_dir, exist_ok=True)
    feather_path = f"{out_dir}/{split_name}.feather"
    df_stgcnn.to_feather(feather_path)
    print(f"  -> {len(df_stgcnn):,} rows → {feather_path}")
    print(f"  -> Total agents exported:   {total_agents_exported:,}")
    print(f"  -> Samples with neighbours: {samples_with_neighbors:,} "
          f"({100*samples_with_neighbors/max(1,len(samples)):.1f}%)")
    del stgcnn_data, df_stgcnn
    gc.collect()


# ============================================================================
# EXPORTER 3: GRAPHORMER-PYG  (unchanged structure from V21)
# ============================================================================

def create_single_graphormer_graph(args):
    (graph_idx, rec_id, focal_track_id, end_obs_frame,
     frame_lookup_original, trajectory_lookup_original,
     frame_to_idx_lookup_original, lane_cache,
     temporal_sample_offsets, obs_len, pred_len) = args

    try:
        node_features = []
        node_type     = []
        edge_index    = [[], []]
        agent_id_to_idx = {}
        lane_id_to_idx  = {}
        current_node_idx = 0

        focal_traj = trajectory_lookup_original.get((rec_id, focal_track_id))
        if focal_traj is None:
            return (graph_idx, None)

        focal_idx_map = frame_to_idx_lookup_original.get(
            (rec_id, focal_track_id), {})
        focal_obs_idx = focal_idx_map.get(end_obs_frame)
        if focal_obs_idx is None:
            return (graph_idx, None)
        focal_pos = focal_traj['utm_xy'][focal_obs_idx]

        all_agents = frame_lookup_original.get((rec_id, end_obs_frame), [])

        # Build agent node list — focal agent first for consistent focal_idx
        ordered_agents = [focal_track_id] + [
            a for a in all_agents if a != focal_track_id
        ]

        for agent_id in ordered_agents:
            agent_traj = trajectory_lookup_original.get((rec_id, agent_id))
            if agent_traj is None:
                continue
            agent_idx_map = frame_to_idx_lookup_original.get(
                (rec_id, agent_id), {})
            idx = agent_idx_map.get(end_obs_frame)
            if idx is None:
                continue
            xy = agent_traj['utm_xy'][idx]
            if np.linalg.norm(xy - focal_pos) > NEIGHBOR_DISTANCE_THRESHOLD:
                continue

            kin  = agent_traj['kinematics'][idx]
            stat = agent_traj['static_features']

            temporal_displacements = []
            for dt in temporal_sample_offsets:
                past_idx = agent_idx_map.get(end_obs_frame - dt)
                if past_idx is not None:
                    temporal_displacements.extend(
                        (xy - agent_traj['utm_xy'][past_idx]).tolist())
                else:
                    temporal_displacements.extend([0.0, 0.0])

            node_features.append(
                np.concatenate([xy, kin, stat,
                                np.array(temporal_displacements, dtype=np.float32)]))
            node_type.append(0)
            agent_id_to_idx[agent_id] = current_node_idx
            current_node_idx += 1

        # Lane nodes
        map_name = RECORDING_ID_TO_MAP_KEY.get(rec_id)
        if map_name and map_name in lane_cache:
            cached = lane_cache[map_name]
            for lane_id, l_data in cached['lanes'].items():
                start_pt = l_data['start']
                if np.linalg.norm(start_pt - focal_pos) > LANE_PROXIMITY_THRESHOLD:
                    continue
                l_feats = np.zeros(GRAPHORMER_NODE_DIM, dtype=np.float32)
                l_feats[0:2] = start_pt
                l_feats[5]   = l_data['heading']
                l_feats[18]  = l_data['length']
                node_features.append(l_feats)
                node_type.append(1)
                lane_id_to_idx[lane_id] = current_node_idx
                current_node_idx += 1

            for lane_id in lane_id_to_idx:
                u = lane_id_to_idx[lane_id]
                for succ_id in cached['successors'].get(lane_id, []):
                    if succ_id in lane_id_to_idx:
                        edge_index[0].append(u)
                        edge_index[1].append(lane_id_to_idx[succ_id])

        for a_id, a_idx in agent_id_to_idx.items():
            a_pos = node_features[a_idx][:2]
            for l_id, l_idx in lane_id_to_idx.items():
                l_pos = node_features[l_idx][:2]
                if np.linalg.norm(a_pos - l_pos) < AGENT_LANE_DISTANCE_THRESHOLD:
                    edge_index[0].extend([a_idx, l_idx])
                    edge_index[1].extend([l_idx, a_idx])

        agent_indices = list(agent_id_to_idx.values())
        for i in range(len(agent_indices)):
            for j in range(i + 1, len(agent_indices)):
                ai, aj = agent_indices[i], agent_indices[j]
                dist = np.linalg.norm(
                    np.array(node_features[ai][:2]) -
                    np.array(node_features[aj][:2]))
                if dist < AGENT_EDGE_DISTANCE_THRESHOLD:
                    edge_index[0].extend([ai, aj])
                    edge_index[1].extend([aj, ai])

        start_fut_idx = focal_obs_idx + 1
        end_fut_idx   = start_fut_idx + pred_len
        if end_fut_idx > len(focal_traj['utm_xy']):
            return (graph_idx, None)
        future_xy = focal_traj['utm_xy'][start_fut_idx:end_fut_idx]

        if not node_features or focal_track_id not in agent_id_to_idx:
            return (graph_idx, None)

        node_features_arr = np.array(node_features, dtype=np.float32)

        # FIX 4: normalize kinematic and static features.
        # Position dims [0,1] store raw UTM — Graphormer centers at runtime
        # in train/validate/evaluate via focal_pos subtraction from batch.x.
        # Node feature layout (29 dims total):
        #   [0,1]   xy           (center-shifted above)
        #   [2]     utm_vx       → /20.0  m/s
        #   [3]     utm_vy       → /20.0  m/s
        #   [4]     speed        → /20.0  m/s
        #   [5]     heading      → /π     radians ±π
        #   [6]     heading_rate → /2.0   rad/s
        #   [7]     curvature    already clipped to [-1,1] — no scaling
        #   [8]     utm_ax       → /5.0   m/s²
        #   [9]     utm_ay       → /5.0   m/s²
        #   [10]    time_to_stop → /10.0  seconds (baked max=10)
        #   [11-15] binary/categorical flags — no scaling needed
        #   [16]    class_int    → /3.0   (0-3 → 0-1)
        #   [17]    width        → /5.0   metres
        #   [18]    length       → /20.0  metres
        #   [19:29] temporal (Δx,Δy)×5 → /10.0  metres over obs window
        # Lane nodes have zeros in kinematic slots → safe to scale all rows.
        _KIN_SCALES = {
            2: 20.0, 3: 20.0, 4: 20.0, 5: np.pi, 6: 2.0,
            8: 5.0,  9: 5.0, 10: 10.0, 16: 3.0, 17: 5.0, 18: 20.0,
        }
        for dim, scale in _KIN_SCALES.items():
            node_features_arr[:, dim] /= scale
        node_features_arr[:, 19:29] /= 10.0   # temporal displacement pairs

        x              = torch.tensor(node_features_arr, dtype=torch.float)
        edge_index_t   = torch.tensor(edge_index, dtype=torch.long)
        y              = torch.tensor(future_xy, dtype=torch.float)   # raw UTM
        focal_node_idx = agent_id_to_idx[focal_track_id]

        num_edges = edge_index_t.size(1)
        if num_edges > 0:
            src, dst  = edge_index_t[0], edge_index_t[1]
            # Edge distances use center-shifted coords — relative distances preserved
            distances = torch.norm(x[src, :2] - x[dst, :2], dim=1, keepdim=True)
            edge_attr = 1.0 / (distances + 1e-6)
        else:
            edge_attr = torch.zeros((0, 1), dtype=torch.float)

        graph = Data(
            x=x, edge_index=edge_index_t, edge_attr=edge_attr,
            y=y, focal_idx=focal_node_idx,
            node_type=torch.tensor(node_type, dtype=torch.long),
            recording_id=torch.tensor(rec_id, dtype=torch.long),
        )

        graph = precalculate_custom_attributes(
            graph, max_in_degree=MAX_IN_DEGREE, max_out_degree=MAX_OUT_DEGREE)
        _, _, npl, ept, epl = precalculate_paths(
            graph, max_path_distance=MAX_PATH_DISTANCE)
        graph.node_paths_length = npl
        graph.edge_paths_tensor = ept
        graph.edge_paths_length = epl

        return (graph_idx, graph)

    except Exception:
        return (graph_idx, None)


def export_for_graphormer(split_name, samples, frame_lookup_original,
                          trajectory_lookup_original,
                          frame_to_idx_lookup_original,
                          lane_cache, obs_len, pred_len,
                          temporal_sample_offsets, cfg_tag):
    out_dir = f"{GRAPHORMER_OUTPUT_DIR}/{cfg_tag}/{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Exporter 3] Graphormer-pyg  |  {cfg_tag}  |  {split_name}")
    print(f"  obs={obs_len}f  pred={pred_len}f  samples={len(samples):,}")
    print(f"  TSN offsets: {temporal_sample_offsets}")
    print(f"{'='*60}")

    existing = [f for f in os.listdir(out_dir)
                if f.startswith('graph_') and f.endswith('.pt')]
    if existing:
        for f in existing:
            os.remove(os.path.join(out_dir, f))

    args_list = [
        (idx, rec_id, track_id, frame,
         frame_lookup_original, trajectory_lookup_original,
         frame_to_idx_lookup_original, lane_cache,
         temporal_sample_offsets, obs_len, pred_len)
        for idx, (rec_id, track_id, frame) in enumerate(samples)
    ]

    successful = failed = total_nodes = total_edges = 0
    graphs_with_edges = multi_agent = 0
    BATCH_SIZE = 100

    if JOBLIB_AVAILABLE:
        for batch_start in tqdm(range(0, len(args_list), BATCH_SIZE),
                                desc="Batches"):
            batch = args_list[batch_start:batch_start + BATCH_SIZE]
            results = Parallel(n_jobs=4, prefer="threads")(
                delayed(create_single_graphormer_graph)(a) for a in batch)
            for graph_idx, graph in results:
                if graph is not None:
                    torch.save(graph, f"{out_dir}/graph_{graph_idx}.pt")
                    successful += 1
                    total_nodes += graph.x.shape[0]
                    ne = graph.edge_index.shape[1]
                    total_edges += ne
                    if ne > 0: graphs_with_edges += 1
                    if (graph.node_type == 0).sum() > 1: multi_agent += 1
                else:
                    failed += 1
            del results; gc.collect()
    else:
        for args in tqdm(args_list, desc="Graphs"):
            graph_idx, graph = create_single_graphormer_graph(args)
            if graph is not None:
                torch.save(graph, f"{out_dir}/graph_{graph_idx}.pt")
                successful += 1
                total_nodes += graph.x.shape[0]
                ne = graph.edge_index.shape[1]
                total_edges += ne
                if ne > 0: graphs_with_edges += 1
                if (graph.node_type == 0).sum() > 1: multi_agent += 1
            else:
                failed += 1
            if successful % 500 == 0: gc.collect()

    print(f"  Saved {successful:,} graphs  ({failed:,} failed)")
    print(f"  Multi-agent: {multi_agent:,}  Avg nodes: "
          f"{total_nodes/max(1,successful):.1f}  Avg edges: "
          f"{total_edges/max(1,successful):.1f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("BAKING PIPELINE V28  (5 Hz, heading fix + simplified sampled lookup)")
    print("=" * 80)
    print(f"  N_TRAIN={N_TRAIN}  N_VAL={N_VAL}  N_TEST={N_TEST}  "
          f"(split ≈ 71/14/14 — close to 70/15/15 standard)")
    print(f"  MIN_FOCAL_DISPLACEMENT_M={MIN_FOCAL_DISPLACEMENT_M}m")
    for obs_s, pred_s in EXPERIMENT_CONFIGS:
        print(f"  {config_tag(obs_s,pred_s)}: "
              f"obs={obs_s*FPS}f pred={pred_s*FPS}f "
              f"TSN={compute_tsn_temporal_offsets(obs_s*FPS)}")
    print("=" * 80)

    location_origins, recording_to_location, recording_meta_full = \
        get_origins_from_meta_files(RECORDING_META_PATH)
    save_recording_metadata_json(recording_meta_full, RECORDING_META_OUTPUT_PATH)

    lane_graphs = load_lane_graphs()
    lane_cache  = precompute_lane_data(lane_graphs)

    sample_sizes = {
        'train': N_TRAIN,
        'val':   N_VAL,
        'test':  N_TEST,
    }

    for split_name, csv_path in [
        ('train', f"{BALANCED_DATA_DIR}/train_balanced_v6.csv"),
        ('val',   f"{BALANCED_DATA_DIR}/val_balanced_v6.csv"),
        ('test',  f"{BALANCED_DATA_DIR}/test_balanced_v6.csv"),
    ]:
        t_split = time.time()
        print(f"\n{'='*80}\nSPLIT: {split_name.upper()}\n{'='*80}")

        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} rows")
        df = engineer_features(df, location_origins)
        trajectory_lookup_sampled = build_lookups_sampled(df)

        all_rec_ids = list({rec_id for (rec_id, _) in trajectory_lookup_sampled})
        print(f"  Unique recordings: {len(all_rec_ids)}")

        frame_lookup_original, trajectory_lookup_original, \
            frame_to_idx_lookup_original, class_map = \
            load_original_tracks_with_class(
                all_rec_ids, location_origins, recording_to_location)

        n_draw = sample_sizes[split_name]

        for obs_sec, pred_sec in EXPERIMENT_CONFIGS:
            obs_len    = obs_sec  * FPS
            pred_len   = pred_sec * FPS
            window_len = obs_len  + pred_len
            tsn        = compute_tsn_temporal_offsets(obs_len)
            cfg        = config_tag(obs_sec, pred_sec)

            t_cfg = time.time()
            print(f"\n  {'─'*68}")
            print(f"  {cfg}  obs={obs_len}f  pred={pred_len}f  window={window_len}f")

            all_samples = find_valid_samples(
                trajectory_lookup_sampled,
                trajectory_lookup_original,
                frame_to_idx_lookup_original,
                obs_len, window_len,
            )

            if not all_samples:
                print(f"  WARNING: no valid samples for {cfg}, skipping")
                continue

            # Stratified draw — balanced across scenario types
            samples = stratified_subsample(all_samples, n_draw)
            print(f"  Using {len(samples):,} samples (pool={len(all_samples):,})")

            export_for_multiclass_sgcnn(
                split_name, samples,
                frame_lookup_original, trajectory_lookup_original,
                frame_to_idx_lookup_original,
                obs_len, pred_len, cfg)

            export_for_stgcnn_benchmark(
                split_name, samples,
                frame_lookup_original, trajectory_lookup_original,
                frame_to_idx_lookup_original,
                obs_len, pred_len, cfg)

            export_for_graphormer(
                split_name, samples,
                frame_lookup_original, trajectory_lookup_original,
                frame_to_idx_lookup_original,
                lane_cache, obs_len, pred_len, tsn, cfg)

            print(f"  {cfg}/{split_name} done in "
                  f"{(time.time()-t_cfg)/60:.1f} min")

        del df, trajectory_lookup_sampled
        del frame_lookup_original, trajectory_lookup_original
        del frame_to_idx_lookup_original
        gc.collect()
        print(f"\nSplit {split_name.upper()} done in "
              f"{(time.time()-t_split)/60:.1f} min")

    print(f"\n{'='*80}")
    print(f"BAKING V27 COMPLETE — {(time.time()-t0)/60:.1f} min total")
    print(f"{'='*80}")
    print(f"  Feather columns (SGCN):   frame, agent_id, x, y, class, "
          f"rec_id, sample_idx, focal_agent_id")
    print(f"  Feather columns (STGCNN): frame, ped_id, x, y, "
          f"rec_id, sample_idx, focal_agent_id")
    print(f"  Graphormer .pt: focal_idx guaranteed = 0 (focal agent written first)")


if __name__ == "__main__":
    main()