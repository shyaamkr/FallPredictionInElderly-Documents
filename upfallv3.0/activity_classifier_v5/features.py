"""Feature engineering - derived features, wearable extraction, and column selection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity_classifier import apply_sliding_window

from .config import FEATURES_DIR, logger

WEARABLE_RAW_COLS = [
    "heart_rate_bpm", "spo2_pct", "skin_temp_c",
    "gsr_us", "bp_systolic_mmhg", "bp_diastolic_mmhg",
]

WEARABLE_FEATURE_NAMES = [
    "wearable_hr", "wearable_spo2", "wearable_skin_temp",
    "wearable_gsr", "wearable_sbp", "wearable_dbp",
]

DERIVED_FEATURES = [
    "cadence_proxy", "vertical_dominance", "impact_regularity",
    "gait_energy_ratio", "symmetry_x_oscillation", "stride_variability",
    "horizontal_motion_ratio", "has_imu", "leg_oscillation_energy",
    "oscillation_intensity", "normalized_ankle_variability",
    "has_wearable", "hr_spo2_coupling", "vital_instability",
    "gsr_hr_product", "vital_anomaly_score",
]


def window_wearable_stats(wearable_csv_path: str, n_windows: int) -> pd.DataFrame:
    """Compute per-window mean/std/max/min/range for the 6 wearable signals.

    Partitions the 1 Hz wearable CSV into n_windows segments using the same
    window-boundary slicing as merge_wearable_features() (train time), so
    predict-time wearable features are numerically identical to what the
    model was trained on. Returns n_windows rows of 30 columns, all-zero if
    the CSV is missing, unreadable, or malformed.
    """
    stat_cols = [f"{fn}_{stat}" for fn in WEARABLE_FEATURE_NAMES
                 for stat in ("mean", "std", "max", "min", "range")]
    zeros = pd.DataFrame(np.zeros((n_windows, len(stat_cols))), columns=stat_cols)

    try:
        wdf = pd.read_csv(wearable_csv_path)
    except Exception as e:
        logger.warning("Could not read wearable CSV %s: %s", wearable_csv_path, e)
        return zeros

    missing = [c for c in WEARABLE_RAW_COLS if c not in wdf.columns]
    if missing:
        logger.warning("Wearable CSV missing columns: %s", missing)
        return zeros

    wearable_vals = wdf[WEARABLE_RAW_COLS].values
    n_wearable = len(wearable_vals)
    if n_wearable == 0:
        return zeros

    out: dict[str, list[float]] = {}
    if n_wearable == 1:
        for i, feat_name in enumerate(WEARABLE_FEATURE_NAMES):
            val = float(wearable_vals[0, i])
            out[f"{feat_name}_mean"] = [val] * n_windows
            out[f"{feat_name}_std"] = [0.0] * n_windows
            out[f"{feat_name}_max"] = [val] * n_windows
            out[f"{feat_name}_min"] = [val] * n_windows
            out[f"{feat_name}_range"] = [0.0] * n_windows
    else:
        window_timestamps = np.linspace(0, n_wearable - 1, n_windows + 1)
        for i, feat_name in enumerate(WEARABLE_FEATURE_NAMES):
            signal = wearable_vals[:, i]
            means, stds, maxes, mins, ranges = [], [], [], [], []
            for w in range(n_windows):
                t_start = window_timestamps[w]
                t_end = window_timestamps[w + 1]
                idx_start = max(0, int(np.floor(t_start)))
                idx_end = min(n_wearable, int(np.ceil(t_end)) + 1)
                if idx_end <= idx_start:
                    idx_end = idx_start + 1
                seg = signal[idx_start:idx_end]
                means.append(float(np.mean(seg)))
                stds.append(float(np.std(seg)))
                maxes.append(float(np.max(seg)))
                mins.append(float(np.min(seg)))
                ranges.append(float(np.max(seg) - np.min(seg)))
            out[f"{feat_name}_mean"] = means
            out[f"{feat_name}_std"] = stds
            out[f"{feat_name}_max"] = maxes
            out[f"{feat_name}_min"] = mins
            out[f"{feat_name}_range"] = ranges

    return pd.DataFrame(out)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"subject_id", "action_id", "l1_category", "l2_subtype",
               "video_path", "side", "location_id"}
    cols = [c for c in df.select_dtypes(include="number").columns
            if c not in exclude]
    for d in DERIVED_FEATURES:
        if d in df.columns and d not in cols:
            cols.append(d)
    return cols


def _compute_sample_weight(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y])


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 16 derived features (11 from v4 + 5 wearable)."""
    eps = 1e-8

    # -- v4 derived features (11) --
    disp_mean = df.get("cog_displacement_rate_mean", pd.Series(0.0, index=df.index))
    osc_mean = df.get("vertical_oscillation_count_mean", pd.Series(0.0, index=df.index))
    vel_x_mean = df.get("cog_velocity_x_mean", pd.Series(0.0, index=df.index))
    vel_y_mean = df.get("cog_velocity_y_mean", pd.Series(0.0, index=df.index))
    peak_g_mean = df.get("peak_impact_g_mean", pd.Series(0.0, index=df.index))
    peak_g_std = df.get("peak_impact_g_std", pd.Series(0.0, index=df.index))
    jerk_mean = df.get("jerk_magnitude_mean", pd.Series(0.0, index=df.index))
    acc_mag_mean = df.get("imu_acc_magnitude_mean", pd.Series(0.0, index=df.index))
    sym_mean = df.get("step_symmetry_mean", pd.Series(0.0, index=df.index))
    stride_mean = df.get("stride_regularity_mean", pd.Series(0.0, index=df.index))
    stride_std = df.get("stride_regularity_std", pd.Series(0.0, index=df.index))

    df["cadence_proxy"] = osc_mean / (disp_mean + eps)
    df["vertical_dominance"] = vel_y_mean.abs() / (disp_mean + eps)
    df["impact_regularity"] = peak_g_mean / (peak_g_std + eps)
    df["gait_energy_ratio"] = jerk_mean / (acc_mag_mean + eps)
    df["symmetry_x_oscillation"] = sym_mean * osc_mean
    df["stride_variability"] = stride_std / (stride_mean.abs() + eps)

    df["horizontal_motion_ratio"] = vel_x_mean.abs() / (vel_y_mean.abs() + eps)

    df["has_imu"] = (acc_mag_mean.abs() > eps).astype(float)

    l_leg_std = df.get("left_leg_angle_std", pd.Series(0.0, index=df.index))
    r_leg_std = df.get("right_leg_angle_std", pd.Series(0.0, index=df.index))
    df["leg_oscillation_energy"] = (l_leg_std + r_leg_std) * osc_mean

    vel_y_range = df.get("cog_velocity_y_range", pd.Series(0.0, index=df.index))
    df["oscillation_intensity"] = osc_mean * vel_y_range / (disp_mean + eps)

    ankle_std = df.get("ankle_distance_std", pd.Series(0.0, index=df.index))
    ankle_mean = df.get("ankle_distance_mean", pd.Series(0.0, index=df.index))
    df["normalized_ankle_variability"] = ankle_std / (ankle_mean + eps)

    # -- v5 wearable derived features (5) --
    hr_mean = df.get("wearable_hr_mean", pd.Series(0.0, index=df.index))
    spo2_mean = df.get("wearable_spo2_mean", pd.Series(0.0, index=df.index))
    gsr_mean = df.get("wearable_gsr_mean", pd.Series(0.0, index=df.index))
    sbp_mean = df.get("wearable_sbp_mean", pd.Series(0.0, index=df.index))
    hr_std = df.get("wearable_hr_std", pd.Series(0.0, index=df.index))
    spo2_std = df.get("wearable_spo2_std", pd.Series(0.0, index=df.index))
    sbp_std = df.get("wearable_sbp_std", pd.Series(0.0, index=df.index))

    df["has_wearable"] = (hr_mean.abs() > eps).astype(float)

    # HR-SpO2 coupling: when HR drops and SpO2 drops together = syncope risk
    # Approximate via correlation of their std deviations
    df["hr_spo2_coupling"] = hr_std * spo2_std / (hr_mean.abs() + eps)

    # Vital instability: sum of coefficient-of-variation across key vitals
    hr_cv = hr_std / (hr_mean.abs() + eps)
    spo2_cv = spo2_std / (spo2_mean.abs() + eps)
    sbp_cv = sbp_std / (sbp_mean.abs() + eps)
    df["vital_instability"] = hr_cv + spo2_cv + sbp_cv

    # GSR x HR interaction: stress amplified by cardiac response
    df["gsr_hr_product"] = gsr_mean * hr_mean / 1000.0

    # Vital anomaly score: count of vitals outside normal range per window
    anomaly = pd.Series(0.0, index=df.index)
    if hr_mean.abs().max() > eps:
        anomaly += ((hr_mean < 55) | (hr_mean > 120)).astype(float)
    if spo2_mean.abs().max() > eps:
        anomaly += (spo2_mean < 93).astype(float)
    if sbp_mean.abs().max() > eps:
        anomaly += ((sbp_mean < 100) | (sbp_mean > 160)).astype(float)
    skin_temp_mean = df.get("wearable_skin_temp_mean", pd.Series(0.0, index=df.index))
    if skin_temp_mean.abs().max() > eps:
        anomaly += ((skin_temp_mean < 33.0) | (skin_temp_mean > 37.0)).astype(float)
    if gsr_mean.abs().max() > eps:
        anomaly += (gsr_mean > 20.0).astype(float)
    dbp_mean = df.get("wearable_dbp_mean", pd.Series(0.0, index=df.index))
    if dbp_mean.abs().max() > eps:
        anomaly += ((dbp_mean < 60) | (dbp_mean > 100)).astype(float)
    df["vital_anomaly_score"] = anomaly

    return df


def merge_wearable_features() -> None:
    """Merge wearable sensor data into the existing windowed features parquet.

    Reads windowed_features.parquet (from v4 extraction), looks up each
    video's *_wearable_sensors.csv, resamples to window level, and saves
    as windowed_features_v5.parquet.
    """
    src_path = FEATURES_DIR / "windowed_features.parquet"
    if not src_path.exists():
        logger.error("Source features not found: %s - run --extract first",
                     src_path)
        return

    df = pd.read_parquet(src_path)
    logger.info("Loaded %d rows from %s", len(df), src_path)

    if "video_path" not in df.columns:
        logger.error("No 'video_path' column in features - cannot locate "
                     "wearable CSVs")
        return

    for wf_name in WEARABLE_FEATURE_NAMES:
        for stat in ("mean", "std", "max", "min", "range"):
            col = f"{wf_name}_{stat}"
            if col not in df.columns:
                df[col] = 0.0

    video_groups = df.groupby("video_path")
    found = 0
    not_found = 0

    for video_path, group_idx in video_groups.groups.items():
        vid_dir = Path(video_path).parent
        wearable_csvs = list(vid_dir.glob("*_wearable_sensors.csv"))

        if not wearable_csvs:
            not_found += 1
            continue

        wearable_csv = str(wearable_csvs[0])
        n_windows = len(group_idx)
        stats = window_wearable_stats(wearable_csv, n_windows)
        if stats.abs().sum().sum() == 0:
            not_found += 1
            continue

        found += 1
        for col in stats.columns:
            df.loc[group_idx, col] = stats[col].values

    out_path = FEATURES_DIR / "windowed_features_v5.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved v5 features to %s (%d rows, %d cols)",
                out_path, len(df), len(df.columns))
    logger.info("Wearable data found for %d/%d videos (%d missing)",
                found, found + not_found, not_found)