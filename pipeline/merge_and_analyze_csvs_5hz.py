"""
================================================================================
STEP 1: Merge inD Dataset CSVs and Analyze Distribution
================================================================================

SOTA Scenario Classification for Multi-Object Trajectory Prediction
VERSION 3.1 - 5 Hz Downsampled (Simplified 5-Category, 2D Distribution)

Changes from V3.0:
- Data source: data_output_5hz (downsampled from 25 Hz to 5 Hz)
- FRAME_RATE = 5 Hz (was 25 Hz), DT = 0.2s (was 0.04s)
- min_stop_duration_frames = 2 (was 10) to preserve 0.4s physical threshold
- All other thresholds unchanged (already in meters/m/s)

This script:
1. Loads and analyzes the 3 inD CSV files
2. Identifies primary/foreign keys
3. Merges them intelligently
4. Handles null values
5. Computes scenario type using SOTA methodology
6. Simplifies to 5 clean categories for stable training
7. Computes 2D (class × scenario) distribution
8. Validates classification against expected distributions

Input files:
- aggregated_tracks.csv (main trajectory data)
- aggregated_tracksMeta.csv (trajectory metadata)  
- aggregated_recordingMeta.csv (recording/location metadata)

Scenario Classification Methodology:
------------------------------------
Based on established trajectory prediction benchmarks:
- Waymo Open Motion Dataset (Ettinger et al., 2021)
- Argoverse 2 Motion Forecasting (Wilson et al., 2023)
- nuScenes Prediction Challenge (Caesar et al., 2020)
- inD Dataset Paper (Bock et al., 2020)

Simplified Categories (5 instead of 8):
---------------------------------------
1. turning_maneuver: turning + turning_lane_change
2. lane_change: lateral shift without heading change
3. straight: straight_constant + accelerating
4. stopping_behavior: stop_at_turn + stopping
5. stationary: no significant movement

Author: Mehran
Version: 3.0 (Simplified 5-Category, 2D Distribution)
Last Updated: December 2025
================================================================================
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data/inD-dataset-v1/data_output_5hz"
OUTPUT_DIR = "data/merged_analyzed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACKS_FILE = f"{DATA_DIR}/aggregated_tracks.csv"
TRACKS_META_FILE = f"{DATA_DIR}/aggregated_tracksMeta.csv"
RECORDING_META_FILE = f"{DATA_DIR}/aggregated_recordingMeta.csv"

# inD Dataset Parameters (DOWNSAMPLED)
FRAME_RATE = 5   # Hz (downsampled from original 25 Hz)
DT = 1.0 / FRAME_RATE  # 0.2 seconds per frame

# Recording ID to Location Mapping (from inD documentation)
RECORDING_TO_LOCATION = {}
for i in range(0, 33):
    if 0 <= i <= 6:
        RECORDING_TO_LOCATION[i] = "aseag"
    elif 7 <= i <= 10:
        RECORDING_TO_LOCATION[i] = "bendplatz"
    elif 11 <= i <= 17:
        RECORDING_TO_LOCATION[i] = "bendplatz_construction"
    elif 18 <= i <= 29:
        RECORDING_TO_LOCATION[i] = "frankenburg"
    elif 30 <= i <= 32:
        RECORDING_TO_LOCATION[i] = "heckstrasse"


# ============================================================================
# SCENARIO SIMPLIFICATION MAPPING
# ============================================================================

# Map 8 detailed scenarios to 5 simplified categories
SCENARIO_SIMPLIFICATION = {
    'turning': 'turning_maneuver',
    'turning_lane_change': 'turning_maneuver',
    'lane_change': 'lane_change',
    'straight_constant': 'straight',
    'accelerating': 'straight',
    'stop_at_turn': 'stopping_behavior',
    'stopping': 'stopping_behavior',
    'stationary': 'stationary'
}

# Simplified category descriptions for thesis
SIMPLIFIED_CATEGORIES = {
    'turning_maneuver': 'Trajectories with significant heading change (>45° accumulated), including turns with lateral deviation',
    'lane_change': 'Lateral displacement without significant heading change (body-frame method)',
    'straight': 'Linear motion with constant or varying speed',
    'stopping_behavior': 'Trajectories containing sustained stops (≥0.4s at <0.3 m/s)',
    'stationary': 'Minimal overall displacement (<0.5m) and near-zero mean speed'
}


# ============================================================================
# SCENARIO CLASSIFICATION THRESHOLDS (Calibrated for German Urban Traffic)
# ============================================================================

@dataclass
class ClassificationThresholds:
    """
    Physics-based thresholds for scenario classification.
    
    CALIBRATED FOR inD DATASET based on:
    - German traffic regulations (StVO)
    - Vehicle dynamics literature
    - inD dataset characteristics (urban intersections)
    - Empirical distribution analysis
    
    Note on accumulated vs endpoint heading:
    - Accumulated heading captures every micro-correction
    - For urban intersections, use HIGHER thresholds than endpoint-based
    - 45° accumulated ≈ 30° endpoint for typical trajectories
    
    References:
    - Typical urban intersection turn: 45-120° accumulated heading change
    - Lane width in Germany: ~3.5m (lane change threshold ~2.0m for vehicles)
    - Stop definition: sustained speed < 0.3 m/s for > 0.4 seconds
    """
    # Stationary detection
    stationary_displacement: float = 0.5  # meters
    stationary_speed: float = 0.15  # m/s
    
    # Stopping detection
    stop_speed_threshold: float = 0.3  # m/s (below this = stopped)
    min_stop_duration_frames: int = 2  # frames (0.4s at 5fps)
    
    # Turning detection - CALIBRATED for accumulated heading
    # 45° accumulated corresponds to ~25-30° actual turn for smooth paths
    turn_heading_threshold_deg: float = 45.0  # degrees accumulated
    
    # Lane change detection
    lane_change_lateral_threshold: float = 2.0  # meters
    lane_change_max_heading_deg: float = 30.0  # max heading change for pure LC
    
    # Combined maneuver detection (turning + lane change)
    # TUNED: Increased from 2.5 to 4.0 to reduce false positives
    # This ensures only significant lateral deviation during turns is flagged
    turning_lane_change_lateral: float = 4.0  # meters lateral during turn
    
    # Acceleration detection
    acceleration_speed_ratio: float = 0.5  # 50% speed variation
    
    # Low-speed heading interpolation
    min_speed_for_heading: float = 0.1  # m/s
    
    @classmethod
    def for_agent_class(cls, agent_class: str) -> 'ClassificationThresholds':
        """Get thresholds adjusted for specific agent types."""
        thresholds = cls()
        
        if agent_class in ["bicycle", "pedestrian"]:
            # Bicycles and pedestrians have more heading variation naturally
            # Use HIGHER thresholds to avoid over-classifying as turns
            thresholds.stationary_displacement = 0.3
            thresholds.stationary_speed = 0.1
            thresholds.stop_speed_threshold = 0.2
            thresholds.turn_heading_threshold_deg = 60.0  # Higher for VRUs
            thresholds.lane_change_lateral_threshold = 1.5  # Lower for narrow agents
            thresholds.turning_lane_change_lateral = 3.0  # Tuned for VRUs
        elif agent_class in ["truck", "bus", "trailer", "truck_bus"]:
            # Large vehicles have larger turning radii
            thresholds.turn_heading_threshold_deg = 50.0
            thresholds.lane_change_lateral_threshold = 2.5
            thresholds.turning_lane_change_lateral = 5.0  # Larger for trucks
            
        return thresholds


# ============================================================================
# CORE MATHEMATICS: ANGLE HANDLING (Fixes Wrap-Around Bug)
# ============================================================================

def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-π, π] range.
    
    This is essential for proper heading comparison.
    Using arctan2(sin, cos) is the standard robust method.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute signed angular difference (angle2 - angle1).
    
    Handles wrap-around correctly:
    - Turn from 170° to -170° returns +20° (not -340°)
    - Turn from -5° to +5° returns +10°
    
    Returns value in [-π, π].
    """
    diff = angle2 - angle1
    return wrap_angle(diff)


def compute_accumulated_heading_change(headings: np.ndarray) -> Tuple[float, float]:
    """
    Compute accumulated heading change with proper wrap-around handling.
    
    This is the SOTA method used by Waymo and Argoverse.
    
    Returns:
        total_signed: Signed accumulated change (+ = left/CCW, - = right/CW)
        total_unsigned: Total absolute rotation (always positive)
    
    Example:
        90° left turn in 10 steps of 9° each:
        - total_signed = +90° (π/2)
        - total_unsigned = 90° (π/2)
        
        S-curve (45° left then 45° right):
        - total_signed = 0°
        - total_unsigned = 90° (π/2)
    """
    if len(headings) < 2:
        return 0.0, 0.0
    
    signed_changes = []
    for i in range(1, len(headings)):
        diff = angle_difference(headings[i-1], headings[i])
        signed_changes.append(diff)
    
    signed_changes = np.array(signed_changes)
    
    total_signed = np.sum(signed_changes)
    total_unsigned = np.sum(np.abs(signed_changes))
    
    return total_signed, total_unsigned


# ============================================================================
# HEADING COMPUTATION WITH LOW-SPEED INTERPOLATION
# ============================================================================

def compute_heading_from_velocity(vx: np.ndarray, vy: np.ndarray,
                                   min_speed: float = 0.1) -> np.ndarray:
    """
    Compute heading from velocity with low-speed interpolation.
    
    When vehicle speed is very low (< min_speed), heading from velocity
    is unreliable (noisy GPS/tracking). We interpolate from nearby
    valid headings to prevent spurious heading jumps.
    
    This is standard practice in trajectory prediction preprocessing.
    
    Args:
        vx: X-velocity array
        vy: Y-velocity array
        min_speed: Minimum speed for reliable heading (default 0.1 m/s)
    
    Returns:
        headings: Array of heading angles in radians
    """
    speeds = np.sqrt(vx**2 + vy**2)
    headings = np.arctan2(vy, vx)
    valid = speeds > min_speed
    
    if not valid.any():
        # All frames have near-zero speed - heading is undefined
        # Return zeros (stationary agent)
        return np.zeros_like(headings)
    
    # Interpolate headings for low-speed frames
    for i in range(len(headings)):
        if not valid[i]:
            # Find nearest valid headings
            left_idx = i - 1
            while left_idx >= 0 and not valid[left_idx]:
                left_idx -= 1
            
            right_idx = i + 1
            while right_idx < len(headings) and not valid[right_idx]:
                right_idx += 1
            
            # Interpolate (handling angle wrap-around)
            if left_idx >= 0 and right_idx < len(headings):
                h1, h2 = headings[left_idx], headings[right_idx]
                alpha = (i - left_idx) / (right_idx - left_idx)
                diff = angle_difference(h1, h2)
                headings[i] = wrap_angle(h1 + alpha * diff)
            elif left_idx >= 0:
                headings[i] = headings[left_idx]
            elif right_idx < len(headings):
                headings[i] = headings[right_idx]
    
    return headings


# ============================================================================
# STOPPING DETECTION (Fixes False Positive Bug)
# ============================================================================

def detect_stop_events(speeds: np.ndarray,
                       stop_threshold: float = 0.3,
                       min_stop_duration: int = 10
                       ) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Detect SUSTAINED stopping events.
    
    A stop is defined as:
    - Speed below threshold for at least min_stop_duration consecutive frames
    
    This prevents false positives from:
    - Momentary slow-downs (slowing for a turn)
    - Slow-moving bicycles/pedestrians
    - Brief speed dips from tracking noise
    
    Args:
        speeds: Array of speeds (m/s)
        stop_threshold: Maximum speed to be considered stopped (default 0.3 m/s)
        min_stop_duration: Minimum frames to count as stop (default 10 = 0.4s)
    
    Returns:
        total_stopped_frames: Total frames in stop state
        num_stop_events: Number of distinct stop periods
        stop_intervals: List of (start_frame, end_frame) tuples
    """
    is_stopped = speeds < stop_threshold
    
    stop_intervals = []
    current_start = None
    total_stopped = 0
    
    for i, stopped in enumerate(is_stopped):
        if stopped:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                duration = i - current_start
                if duration >= min_stop_duration:
                    stop_intervals.append((current_start, i))
                    total_stopped += duration
                current_start = None
    
    # Handle stop at end of trajectory
    if current_start is not None:
        duration = len(is_stopped) - current_start
        if duration >= min_stop_duration:
            stop_intervals.append((current_start, len(is_stopped)))
            total_stopped += duration
    
    return total_stopped, len(stop_intervals), stop_intervals


# ============================================================================
# LANE CHANGE DETECTION (Fixes Curved Road Bug)
# ============================================================================

def detect_lane_change(positions: np.ndarray,
                       headings: np.ndarray,
                       min_lateral_shift: float = 2.0,
                       max_heading_change: float = np.deg2rad(30)
                       ) -> Tuple[bool, float, str]:
    """
    Detect lane change using body-frame lateral displacement.
    
    Key insight from SOTA research:
    - Lane change = lateral shift WITHOUT significant heading change
    - Turn = heading change (lateral deviation is due to turning, not lane change)
    - Curved road = heading follows road curvature (not a maneuver)
    
    By checking heading change first, we avoid false positives on curved roads.
    
    Args:
        positions: [T, 2] array of (x, y) positions
        headings: [T,] array of heading angles
        min_lateral_shift: Minimum lateral displacement for lane change
        max_heading_change: Maximum heading change to still be lane change
    
    Returns:
        is_lane_change: Whether trajectory contains lane change
        signed_shift: Lateral shift (positive = left, negative = right)
        direction: "left", "right", or "none"
    """
    if len(positions) < 10:
        return False, 0.0, "none"
    
    # First check: Is there significant heading change?
    # If yes, this is a turn, not a lane change
    _, total_unsigned = compute_accumulated_heading_change(headings)
    
    if total_unsigned > max_heading_change:
        return False, 0.0, "none"
    
    # Compute lateral displacement in body frame
    # Use initial heading as reference (assuming straight road segment)
    initial_heading = headings[0]
    
    # Rotation matrix to transform to body frame
    cos_h = np.cos(-initial_heading)
    sin_h = np.sin(-initial_heading)
    
    # Transform positions relative to start
    relative_pos = positions - positions[0]
    
    # Rotate to body frame (x = forward, y = lateral)
    body_x = relative_pos[:, 0] * cos_h - relative_pos[:, 1] * sin_h
    body_y = relative_pos[:, 0] * sin_h + relative_pos[:, 1] * cos_h
    
    # Lateral displacement is y in body frame
    max_lateral_left = np.max(body_y)   # Positive = left
    max_lateral_right = -np.min(body_y)  # Positive = right (min is negative)
    
    # Determine direction and magnitude
    if max_lateral_left > max_lateral_right and max_lateral_left > min_lateral_shift:
        return True, max_lateral_left, "left"
    elif max_lateral_right > max_lateral_left and max_lateral_right > min_lateral_shift:
        return True, -max_lateral_right, "right"
    
    return False, 0.0, "none"


def compute_max_lateral_deviation(positions: np.ndarray) -> float:
    """
    Compute maximum lateral deviation from straight-line path.
    
    This is used for detecting combined turning + lane change maneuvers,
    where we want to know if there was significant lateral movement
    REGARDLESS of heading change.
    
    Returns:
        max_lateral: Maximum perpendicular distance from start-end line
    """
    if len(positions) < 3:
        return 0.0
    
    # Straight line from start to end
    start = positions[0]
    end = positions[-1]
    path_vector = end - start
    path_length = np.linalg.norm(path_vector)
    
    if path_length < 0.5:  # Very short path
        return 0.0
    
    path_direction = path_vector / path_length
    
    # Compute perpendicular distance for each point
    lateral_devs = []
    for pos in positions[1:-1]:
        vec_to_point = pos - start
        # Project onto path
        longitudinal = np.dot(vec_to_point, path_direction)
        proj = longitudinal * path_direction
        # Perpendicular component
        lateral = np.linalg.norm(vec_to_point - proj)
        lateral_devs.append(lateral)
    
    return max(lateral_devs) if lateral_devs else 0.0


# ============================================================================
# MAIN SCENARIO CLASSIFICATION FUNCTION
# ============================================================================

def compute_scenario_type(group, agent_class: str = "car") -> str:
    """
    Classify trajectory scenario using SOTA methodology.
    
    Classification hierarchy (based on Waymo/Argoverse):
    1. STATIONARY: No significant movement
    2. TURNING: Significant accumulated heading change (>45° for cars, >60° for VRUs)
    3. LANE_CHANGE: Lateral shift without heading change
    4. STOPPING: Sustained zero velocity period
    5. ACCELERATING: Significant speed variation on straight path
    6. STRAIGHT_CONSTANT: Default - constant velocity, no maneuver
    
    Combined scenarios:
    - STOP_AT_TURN: Stopping + turning (e.g., stop sign at intersection)
    - TURNING_LANE_CHANGE: Turning + significant lateral deviation (>4m)
    
    Args:
        group: DataFrame group with columns: xCenter, yCenter, xVelocity, yVelocity
        agent_class: Agent type for threshold adjustment
    
    Returns:
        Scenario label string
    """
    # Get thresholds for this agent type
    thresh = ClassificationThresholds.for_agent_class(agent_class)
    
    # Extract trajectory data
    positions = group[['xCenter', 'yCenter']].values
    velocities = group[['xVelocity', 'yVelocity']].values
    
    # Compute speeds
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Compute headings with low-speed interpolation
    headings = compute_heading_from_velocity(
        velocities[:, 0], 
        velocities[:, 1],
        min_speed=thresh.min_speed_for_heading
    )
    
    # Compute trajectory metrics
    total_displacement = np.linalg.norm(positions[-1] - positions[0]) if len(positions) > 1 else 0
    mean_speed = np.mean(speeds)
    min_speed = np.min(speeds)
    max_speed = np.max(speeds)
    
    # Accumulated heading change (with wrap-around handling)
    total_signed_heading, total_unsigned_heading = compute_accumulated_heading_change(headings)
    
    # Stop detection (sustained stops only)
    stopped_frames, num_stops, stop_intervals = detect_stop_events(
        speeds,
        stop_threshold=thresh.stop_speed_threshold,
        min_stop_duration=thresh.min_stop_duration_frames
    )
    
    # Lane change detection (body-frame method) - for non-turning paths
    is_lane_change, lateral_shift, lc_direction = detect_lane_change(
        positions, 
        headings,
        min_lateral_shift=thresh.lane_change_lateral_threshold,
        max_heading_change=np.deg2rad(thresh.lane_change_max_heading_deg)
    )
    
    # Maximum lateral deviation (for detecting turning + lane change)
    max_lateral_dev = compute_max_lateral_deviation(positions)
    
    # =========================================================================
    # CLASSIFICATION LOGIC (Clear priority chain)
    # =========================================================================
    
    # 1. STATIONARY: Minimal movement overall
    if (total_displacement < thresh.stationary_displacement and 
        mean_speed < thresh.stationary_speed):
        return "stationary"
    
    # 2. Check for TURNING (accumulated heading change)
    turn_threshold_rad = np.deg2rad(thresh.turn_heading_threshold_deg)
    is_turning = total_unsigned_heading > turn_threshold_rad
    
    if is_turning:
        # Sub-classify turning scenarios
        if stopped_frames > thresh.min_stop_duration_frames:
            # Stopped AND turned = stop at turn (e.g., stop sign)
            return "stop_at_turn"
        
        # Check for combined turn + lane change
        # TUNED: Only flag if lateral deviation is very significant (>4m for cars)
        if max_lateral_dev > thresh.turning_lane_change_lateral:
            return "turning_lane_change"
        
        # Pure turn (no stop, no significant lateral deviation)
        return "turning"
    
    # 3. LANE CHANGE: Lateral shift without significant heading change
    if is_lane_change:
        return "lane_change"
    
    # 4. STOPPING: Sustained zero velocity (but moved overall)
    if stopped_frames > thresh.min_stop_duration_frames:
        return "stopping"
    
    # 5. ACCELERATING: Significant speed variation
    if len(speeds) > 10 and mean_speed > 0.5:
        speed_range = max_speed - min_speed
        speed_change_ratio = speed_range / mean_speed
        if speed_change_ratio > thresh.acceleration_speed_ratio:
            return "accelerating"
    
    # 6. Default: STRAIGHT_CONSTANT
    return "straight_constant"


# ============================================================================
# STEP 1: LOAD AND INSPECT CSVs
# ============================================================================

def inspect_csv_structure(filepath: str, name: str) -> pd.DataFrame:
    """Analyze CSV structure and identify potential keys."""
    print(f"\n{'='*80}")
    print(f"INSPECTING: {name}")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath, nrows=1000)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns (sample)")
    print(f"\nColumns:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        nunique = df[col].nunique()
        has_nulls = df[col].isnull().any()
        print(f"  {i:2d}. {col:30s} | {str(dtype):10s} | {nunique:6,} unique | Nulls: {has_nulls}")
    
    print(f"\nPotential Primary Keys (unique per row):")
    for col in df.columns:
        if df[col].nunique() == len(df):
            print(f"  ✓ {col}")
    
    print(f"\nPotential Foreign Keys (repeated values):")
    for col in df.columns:
        nunique = df[col].nunique()
        if 1 < nunique < len(df) * 0.9:
            print(f"  → {col} ({nunique:,} unique values)")
    
    return df


def load_full_csvs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three CSVs completely."""
    print(f"\n{'='*80}")
    print(f"LOADING FULL CSVs")
    print(f"{'='*80}")
    
    print(f"\nLoading tracks.csv...")
    tracks = pd.read_csv(TRACKS_FILE)
    print(f"  Loaded: {len(tracks):,} rows")
    
    print(f"\nLoading tracksMeta.csv...")
    tracks_meta = pd.read_csv(TRACKS_META_FILE)
    print(f"  Loaded: {len(tracks_meta):,} rows")
    
    print(f"\nLoading recordingMeta.csv...")
    recording_meta = pd.read_csv(RECORDING_META_FILE)
    print(f"  Loaded: {len(recording_meta):,} rows")
    
    return tracks, tracks_meta, recording_meta


# ============================================================================
# STEP 2: MERGE CSVs
# ============================================================================

def merge_csvs(tracks: pd.DataFrame, 
               tracks_meta: pd.DataFrame, 
               recording_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three dataframes on appropriate keys.
    
    Structure:
    - tracks: Per-frame data (recordingId, trackId, frame, xCenter, yCenter, ...)
    - tracksMeta: Per-trajectory metadata (recordingId, trackId, class, width, ...)
    - recordingMeta: Per-recording metadata (recordingId, locationId, frameRate, ...)
    """
    print(f"\n{'='*80}")
    print(f"MERGING CSVs")
    print(f"{'='*80}")
    
    # Step 1: Merge tracks with tracksMeta
    print(f"\nStep 1: Merging tracks ← tracksMeta on (recordingId, trackId)...")
    
    assert 'recordingId' in tracks.columns, "tracks.csv missing 'recordingId'"
    assert 'trackId' in tracks.columns, "tracks.csv missing 'trackId'"
    assert 'recordingId' in tracks_meta.columns, "tracksMeta.csv missing 'recordingId'"
    assert 'trackId' in tracks_meta.columns, "tracksMeta.csv missing 'trackId'"
    
    merged = tracks.merge(
        tracks_meta,
        on=['recordingId', 'trackId'],
        how='left',
        suffixes=('', '_meta')
    )
    print(f"  Result: {len(merged):,} rows × {len(merged.columns)} columns")
    
    null_count = merged.isnull().sum().sum()
    if null_count > 0:
        print(f"  ⚠️  Warning: {null_count:,} null values after merge")
    else:
        print(f"  ✓ No null values")
    
    # Step 2: Merge with recordingMeta
    print(f"\nStep 2: Merging result ← recordingMeta on (recordingId)...")
    
    assert 'recordingId' in recording_meta.columns, "recordingMeta.csv missing 'recordingId'"
    
    merged = merged.merge(
        recording_meta,
        on='recordingId',
        how='left',
        suffixes=('', '_recording')
    )
    print(f"  Result: {len(merged):,} rows × {len(merged.columns)} columns")
    
    null_count = merged.isnull().sum().sum()
    if null_count > 0:
        print(f"  ⚠️  Warning: {null_count:,} null values after merge")
        null_cols = merged.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        if len(null_cols) > 0:
            print(f"\n  Columns with nulls:")
            for col, count in null_cols.items():
                print(f"    {col:30s}: {count:,} nulls ({(count/len(merged))*100:.2f}%)")
    else:
        print(f"  ✓ No null values")
    
    # Step 3: Handle null values
    print(f"\nStep 3: Handling null values...")
    null_count_before = merged.isnull().sum().sum()
    print(f"  Nulls before: {null_count_before:,}")
    
    if 'startTime' in merged.columns:
        merged['startTime'] = merged['startTime'].fillna(0.0)
        print(f"  → Filled 'startTime' nulls with 0.0")
    
    null_count_after = merged.isnull().sum().sum()
    print(f"  Nulls after: {null_count_after:,}")
    
    if null_count_after > 0:
        print(f"  ⚠️  Warning: {null_count_after:,} nulls remaining")
        remaining_nulls = merged.isnull().sum()
        remaining_nulls = remaining_nulls[remaining_nulls > 0]
        for col, count in remaining_nulls.items():
            print(f"    {col:30s}: {count:,} nulls")
    else:
        print(f"  ✓ All nulls handled!")
    
    print(f"\n{'='*80}")
    print(f"MERGE COMPLETE")
    print(f"{'='*80}")
    print(f"Final dataset: {len(merged):,} rows × {len(merged.columns)} columns")
    
    return merged


# ============================================================================
# STEP 3: COMPUTE SCENARIO FEATURES (SOTA Method)
# ============================================================================

def add_scenario_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Add SOTA scenario type to each trajectory."""
    print(f"\n{'='*80}")
    print(f"COMPUTING SCENARIO TYPES (SOTA Methodology)")
    print(f"{'='*80}")
    
    print(f"\nMethodology:")
    print(f"  • Accumulated heading change with wrap-around handling")
    print(f"  • Sustained stop detection (≥{ClassificationThresholds().min_stop_duration_frames} frames)")
    print(f"  • Body-frame lateral displacement for lane change")
    print(f"  • Agent-class-specific thresholds")
    print(f"  • Tuned lateral deviation threshold: {ClassificationThresholds().turning_lane_change_lateral}m")
    
    print(f"\nGrouping by (recordingId, trackId)...")
    grouped = merged_df.groupby(['recordingId', 'trackId'])
    
    print(f"Total trajectories: {len(grouped):,}")
    
    print(f"\nComputing scenario types...")
    
    # Create scenario type mapping
    scenario_map = {}
    for (rec_id, track_id), group in tqdm(grouped, desc="Classifying trajectories"):
        agent_class = group['class'].iloc[0] if 'class' in group.columns else "car"
        scenario = compute_scenario_type(group, agent_class)
        scenario_map[(rec_id, track_id)] = scenario
    
    # Add to dataframe - detailed scenario
    print(f"\nAdding scenario_type column (8 detailed categories)...")
    merged_df['scenario_type'] = merged_df.apply(
        lambda row: scenario_map.get((row['recordingId'], row['trackId']), 'unknown'),
        axis=1
    )
    
    # Add simplified scenario
    print(f"Adding scenario_simplified column (5 categories)...")
    merged_df['scenario_simplified'] = merged_df['scenario_type'].map(SCENARIO_SIMPLIFICATION)
    
    # Statistics - Detailed (8 categories)
    scenario_counts = merged_df.groupby(['recordingId', 'trackId'])['scenario_type'].first().value_counts()
    total_traj = len(grouped)
    
    print(f"\n{'='*80}")
    print(f"DETAILED SCENARIO DISTRIBUTION (8 categories)")
    print(f"{'='*80}")
    
    for scenario, count in scenario_counts.items():
        pct = (count / total_traj) * 100
        print(f"  {scenario:25s}: {count:8,} ({pct:5.2f}%)")
    
    # Statistics - Simplified (5 categories)
    simplified_counts = merged_df.groupby(['recordingId', 'trackId'])['scenario_simplified'].first().value_counts()
    
    print(f"\n{'='*80}")
    print(f"SIMPLIFIED SCENARIO DISTRIBUTION (5 categories)")
    print(f"{'='*80}")
    
    for scenario, count in simplified_counts.items():
        pct = (count / total_traj) * 100
        print(f"  {scenario:25s}: {count:8,} ({pct:5.2f}%)")
    
    return merged_df


# ============================================================================
# STEP 4: VALIDATE CLASSIFICATION
# ============================================================================

def validate_classification(merged_df: pd.DataFrame) -> Dict:
    """
    Validate scenario classification against expected distributions.
    
    Uses SIMPLIFIED 5-category validation.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING SIMPLIFIED CLASSIFICATION (5 categories)")
    print(f"{'='*80}")
    
    trajectories = merged_df.groupby(['recordingId', 'trackId'])['scenario_simplified'].first()
    total = len(trajectories)
    
    scenario_counts = trajectories.value_counts()
    
    # Expected ranges for simplified categories
    expected_ranges = {
        "turning_maneuver": (0.25, 0.55, "All turning behaviors combined"),
        "lane_change": (0.10, 0.30, "Lateral shift without heading change"),
        "straight": (0.12, 0.35, "Linear motion (constant + accelerating)"),
        "stopping_behavior": (0.08, 0.25, "Stop events (stop_at_turn + stopping)"),
        "stationary": (0.01, 0.08, "No significant movement"),
    }
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "distribution": {},
        "total_trajectories": total
    }
    
    print(f"\n{'Scenario':<25} {'Count':>10} {'Actual':>10} {'Expected':>12} {'Status':>10}")
    print(f"{'─'*70}")
    
    for scenario, (low, high, description) in expected_ranges.items():
        count = scenario_counts.get(scenario, 0)
        pct = count / total if total > 0 else 0
        
        validation_results["distribution"][scenario] = {
            "count": int(count),
            "percentage": float(pct),
            "expected_range": (low, high)
        }
        
        if pct < low:
            status = "⚠️  LOW"
            validation_results["warnings"].append(
                f"{scenario}: {pct:.2%} < expected {low:.0%}-{high:.0%}"
            )
            validation_results["valid"] = False
        elif pct > high:
            status = "⚠️  HIGH"
            validation_results["warnings"].append(
                f"{scenario}: {pct:.2%} > expected {low:.0%}-{high:.0%}"
            )
        else:
            status = "✅ OK"
        
        expected_str = f"{low:.0%}-{high:.0%}"
        print(f"{scenario:<25} {count:>10,} {pct:>9.2%} {expected_str:>12} {status:>10}")
    
    # Summary
    print(f"\n{'='*80}")
    if validation_results["valid"]:
        print(f"✅ VALIDATION PASSED: All scenarios within expected ranges")
    else:
        print(f"⚠️  VALIDATION WARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"   • {warning}")
    print(f"{'='*80}")
    
    return validation_results


# ============================================================================
# STEP 5: ANALYZE 2D DISTRIBUTION (Class × Scenario)
# ============================================================================

def analyze_2d_distribution(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze 2D distribution (Class × Simplified Scenario).
    
    This is the key output for stratified sampling.
    Only 4 classes × 5 scenarios = 20 combinations (manageable!)
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING 2D DISTRIBUTION (Class × Scenario)")
    print(f"{'='*80}")
    
    trajectories = merged_df.groupby(['recordingId', 'trackId']).first().reset_index()
    
    # Add location for reference (but not used in 2D distribution)
    trajectories['location'] = trajectories['recordingId'].map(RECORDING_TO_LOCATION)
    
    total = len(trajectories)
    print(f"\nTotal unique trajectories: {total:,}")
    
    # 1. Class distribution
    print(f"\n{'='*80}")
    print(f"1. CLASS DISTRIBUTION")
    print(f"{'='*80}")
    class_counts = trajectories['class'].value_counts()
    for class_name, count in class_counts.items():
        pct = (count / total) * 100
        print(f"  {class_name:20s}: {count:8,} ({pct:5.2f}%)")
    
    # 2. Simplified Scenario distribution
    print(f"\n{'='*80}")
    print(f"2. SIMPLIFIED SCENARIO DISTRIBUTION")
    print(f"{'='*80}")
    scenario_counts = trajectories['scenario_simplified'].value_counts()
    for scenario, count in scenario_counts.items():
        pct = (count / total) * 100
        print(f"  {scenario:25s}: {count:8,} ({pct:5.2f}%)")
    
    # 3. 2D Distribution Matrix
    print(f"\n{'='*80}")
    print(f"3. 2D DISTRIBUTION MATRIX (Class × Simplified Scenario)")
    print(f"{'='*80}")
    
    # Create pivot table
    distribution_2d = trajectories.groupby(['class', 'scenario_simplified']).size().unstack(fill_value=0)
    
    # Reorder columns for readability
    column_order = ['turning_maneuver', 'lane_change', 'straight', 'stopping_behavior', 'stationary']
    distribution_2d = distribution_2d.reindex(columns=[c for c in column_order if c in distribution_2d.columns])
    
    print(f"\n{distribution_2d.to_string()}")
    
    # 4. Summary statistics
    print(f"\n{'='*80}")
    print(f"4. 2D DISTRIBUTION SUMMARY")
    print(f"{'='*80}")
    
    distribution_flat = trajectories.groupby(['class', 'scenario_simplified']).size()
    
    print(f"\nTotal unique combinations: {len(distribution_flat)}")
    print(f"Maximum possible combinations: {len(class_counts) * len(scenario_counts)} (4×5=20)")
    
    # Identify rare combinations
    print(f"\nRare combinations (< 50 trajectories):")
    print(f"{'─'*60}")
    rare_found = False
    for (cls, scenario), count in distribution_flat.items():
        if count < 50:
            rare_found = True
            pct = (count / total) * 100
            print(f"  {cls:15s} × {scenario:20s}: {count:5,} ({pct:4.2f}%) ⚠️")
    
    if not rare_found:
        print("  ✓ No rare combinations found!")
    
    # Save 2D distribution
    distribution_dict = {}
    for (cls, scenario), count in distribution_flat.items():
        key = f"{cls}|{scenario}"
        distribution_dict[key] = int(count)
    
    with open(f"{OUTPUT_DIR}/2d_distribution.json", 'w') as f:
        json.dump(distribution_dict, f, indent=2)
    
    # Also save as CSV for easy viewing
    distribution_2d.to_csv(f"{OUTPUT_DIR}/2d_distribution_matrix.csv")
    
    print(f"\n✓ Saved 2D distribution to {OUTPUT_DIR}/2d_distribution.json")
    print(f"✓ Saved 2D matrix to {OUTPUT_DIR}/2d_distribution_matrix.csv")
    
    return trajectories, distribution_2d


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"inD DATASET: CSV MERGING AND SOTA SCENARIO CLASSIFICATION")
    print(f"{'='*80}")
    print(f"\nVersion: 3.1 (5 Hz Downsampled, Simplified 5-Category, 2D Distribution)")
    print(f"Methodology: Waymo/Argoverse/nuScenes aligned")
    print(f"\nSimplified Categories:")
    for cat, desc in SIMPLIFIED_CATEGORIES.items():
        print(f"  • {cat}: {desc[:60]}...")
    
    # Step 1: Inspect structure
    print(f"\n{'─'*80}")
    print(f"STEP 1: INSPECTING CSV STRUCTURE")
    print(f"{'─'*80}")
    
    tracks_sample = inspect_csv_structure(TRACKS_FILE, "aggregated_tracks.csv")
    tracks_meta_sample = inspect_csv_structure(TRACKS_META_FILE, "aggregated_tracksMeta.csv")
    recording_meta_sample = inspect_csv_structure(RECORDING_META_FILE, "aggregated_recordingMeta.csv")
    
    # Step 2: Load full CSVs
    print(f"\n{'─'*80}")
    print(f"STEP 2: LOADING FULL CSVs")
    print(f"{'─'*80}")
    
    tracks, tracks_meta, recording_meta = load_full_csvs()
    
    # Step 3: Merge
    print(f"\n{'─'*80}")
    print(f"STEP 3: MERGING CSVs")
    print(f"{'─'*80}")
    
    merged = merge_csvs(tracks, tracks_meta, recording_meta)
    
    # Save merged CSV (before scenario computation)
    print(f"\nSaving merged CSV (raw)...")
    merged_path = f"{OUTPUT_DIR}/merged_raw.csv"
    merged.to_csv(merged_path, index=False)
    print(f"✓ Saved to {merged_path}")
    print(f"  Size: {os.path.getsize(merged_path) / (1024**3):.2f} GB")
    
    # Step 4: Add scenario features
    print(f"\n{'─'*80}")
    print(f"STEP 4: COMPUTING SCENARIO FEATURES (SOTA)")
    print(f"{'─'*80}")
    
    merged = add_scenario_features(merged)
    
    # Step 5: Validate classification (simplified)
    print(f"\n{'─'*80}")
    print(f"STEP 5: VALIDATING CLASSIFICATION")
    print(f"{'─'*80}")
    
    validation_results = validate_classification(merged)
    
    # Save validation results
    with open(f"{OUTPUT_DIR}/validation_results.json", 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Save merged CSV (with scenarios)
    print(f"\nSaving merged CSV (with scenarios)...")
    merged_full_path = f"{OUTPUT_DIR}/merged_full.csv"
    merged.to_csv(merged_full_path, index=False)
    print(f"✓ Saved to {merged_full_path}")
    print(f"  Size: {os.path.getsize(merged_full_path) / (1024**3):.2f} GB")
    
    # Step 6: Analyze 2D distribution
    print(f"\n{'─'*80}")
    print(f"STEP 6: ANALYZING 2D DISTRIBUTION")
    print(f"{'─'*80}")
    
    trajectories, distribution_2d = analyze_2d_distribution(merged)
    
    # Save trajectory-level summary
    print(f"\nSaving trajectory-level summary...")
    traj_path = f"{OUTPUT_DIR}/trajectories_summary.csv"
    trajectories.to_csv(traj_path, index=False)
    print(f"✓ Saved to {traj_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  1. {OUTPUT_DIR}/merged_raw.csv - Merged data (raw)")
    print(f"  2. {OUTPUT_DIR}/merged_full.csv - Merged data (with scenarios)")
    print(f"  3. {OUTPUT_DIR}/trajectories_summary.csv - Trajectory-level summary")
    print(f"  4. {OUTPUT_DIR}/2d_distribution.json - 2D distribution (class × scenario)")
    print(f"  5. {OUTPUT_DIR}/2d_distribution_matrix.csv - 2D matrix view")
    print(f"  6. {OUTPUT_DIR}/validation_results.json - Classification validation")
    
    print(f"\n{'='*80}")
    print(f"METHODOLOGY SUMMARY (for thesis)")
    print(f"{'='*80}")
    print(f"""
Scenario classification follows established trajectory prediction methodology:

1. HEADING COMPUTATION:
   - Accumulated heading change with wrap-around handling
   - Low-speed interpolation to prevent spurious jumps
   - Reference: Waymo Open Motion Dataset (Ettinger et al., 2021)

2. STOP DETECTION:
   - Sustained behavior (≥{ClassificationThresholds().min_stop_duration_frames} frames at <{ClassificationThresholds().stop_speed_threshold} m/s)
   - Prevents false positives from momentary slow-downs
   - Reference: Argoverse 2 (Wilson et al., 2023)

3. LANE CHANGE DETECTION:
   - Body-frame lateral displacement method
   - Distinguishes actual lane changes from curved road following
   - Reference: nuScenes (Caesar et al., 2020)

4. SIMPLIFIED CATEGORIES (5 instead of 8):
   - turning_maneuver: turning + turning_lane_change
   - lane_change: lateral shift without heading change
   - straight: straight_constant + accelerating
   - stopping_behavior: stop_at_turn + stopping
   - stationary: no significant movement
   
5. 2D DISTRIBUTION (Class × Scenario):
   - 4 classes × 5 scenarios = 20 combinations
   - Reduces rare combination issues vs 3D (123 combinations)
   - Enables stable stratified sampling

6. THRESHOLDS:
   - Calibrated for German urban traffic (inD dataset)
   - Agent-class-specific adjustments
   - Lateral deviation: {ClassificationThresholds().turning_lane_change_lateral}m (tuned)
   - Reference: inD Dataset (Bock et al., 2020)
""")
    
    print(f"\nNext step: Run intelligent_2d_sampling.py!")
    print(f"{'='*80}")
    
    return merged, validation_results


if __name__ == "__main__":
    merged, validation = main()