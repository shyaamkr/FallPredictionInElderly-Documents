"""
Activity Classifier for FigShare VideoDataset - dual-modality (pose + IMU) pipeline.

Usage:
    python activity_classifier.py --extract [--max-videos N] [--data-root PATH]
    python activity_classifier.py --train [--model {xgboost,random_forest,lightgbm,svm}]
    python activity_classifier.py --evaluate
    python activity_classifier.py --predict <video>
    python activity_classifier.py --predict-detailed <video>
"""

import argparse
from collections import Counter
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

# MediaPipe pose landmark indices
LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16

# 32 feature columns used for windowed aggregation
FEATURE_COLUMNS: list[str] = [
    # Stability domain (5)
    "torso_inclination", "lateral_lean", "com_sway_area",
    "base_of_support_variability", "pose_confidence",
    # Gait domain (7)
    "cog_velocity_x", "cog_displacement_rate", "ankle_distance",
    "stride_regularity", "step_symmetry", "motion_periodicity",
    "vertical_oscillation_count",
    # Momentum domain (4)
    "cog_velocity_y", "cog_accel_y", "head_drop_rate", "limb_angular_momentum",
    # Transitions domain (5)
    "body_height_ratio", "body_height_change_rate", "aspect_ratio",
    "aspect_ratio_change", "post_window_stillness",
    # Additional pose (2)
    "left_leg_angle", "right_leg_angle",
    # Brightness (1)
    "frame_brightness",
    # IMU features (8)
    "imu_acc_magnitude", "imu_acc_magnitude_variance", "imu_orientation_stability",
    "imu_gyro_periodicity", "peak_impact_g", "jerk_magnitude",
    "orientation_change_rate", "vertical_acc_asymmetry",
]

assert len(FEATURE_COLUMNS) == 32, f"Expected 32, got {len(FEATURE_COLUMNS)}"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    defaults = {
        "data_root": r"C:\Data\FigShare\VideoDataset\VideoDataset",
        "output_dir": str(BASE_DIR / "output"),
        "extraction": {"window_size": 50, "overlap": 0.5},
        "activity_mapping": {
            "l1": {
                "fall": list(range(1, 13)),
                "ambulatory": list(range(13, 18)),
                "transitional": list(range(18, 22)),
                "stationary": list(range(22, 26)),
            },
            "l2": {
                "fall": {
                    "forward_fall": [1, 2], "backward_fall": [3, 4],
                    "lateral_fall": [5, 6], "tripping": [7, 8],
                    "syncope": [9, 10], "slip": [11, 12],
                },
                "ambulatory": {
                    "walking": [13], "slow_walking": [14], "jogging": [15],
                    "stairs_up": [16], "stairs_down": [17],
                },
                "transitional": {
                    "sit_to_stand": [18], "stand_to_sit": [19],
                    "bending": [20], "picking_up": [21],
                },
                "stationary": {
                    "sitting": [22], "lying_down": [23],
                    "standing_still": [24], "leaning": [25],
                },
            },
        },
    }
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            user_cfg = json.load(f)
        for key in defaults:
            if key in user_cfg:
                if isinstance(defaults[key], dict) and isinstance(user_cfg[key], dict):
                    defaults[key].update(user_cfg[key])
                else:
                    defaults[key] = user_cfg[key]
    return defaults


CONFIG = load_config()

_OUTPUT_DIR = Path(CONFIG["output_dir"])
MODEL_DIR   = _OUTPUT_DIR / "models"
PLOTS_DIR   = _OUTPUT_DIR / "plots"
FEATURES_DIR = _OUTPUT_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity mapping helpers
# ---------------------------------------------------------------------------

def _action_to_l1(action_id: int) -> str:
    for cat, ids in CONFIG["activity_mapping"]["l1"].items():
        if action_id in ids:
            return cat
    return "unknown"


def _action_to_l2(action_id: int) -> str:
    for subtypes in CONFIG["activity_mapping"]["l2"].values():
        for subtype, ids in subtypes.items():
            if action_id in ids:
                return subtype
    return "unknown"


# ---------------------------------------------------------------------------
# Video / IMU discovery
# ---------------------------------------------------------------------------

def discover_videos(data_root: Path, max_videos: Optional[int] = None) -> list[dict]:
    """Walk data_root and return a list of trial dicts.

    Each dict has: video_path, imu_path, subject_id, action_id, l1_category,
    l2_subtype, location_id, side.
    """
    records: list[dict] = []

    for top_dir in [data_root / "Fall", data_root / "ADL"]:
        if not top_dir.exists():
            logger.warning("Category directory not found: %s", top_dir)
            continue

        for subj_dir in sorted(top_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            for trial_dir in sorted(subj_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue

                mp4_files = list(trial_dir.glob("*.mp4"))
                txt_files = list(trial_dir.glob("*.txt"))
                metadata_file = trial_dir / "metadata.json"

                if not mp4_files or not txt_files or not metadata_file.exists():
                    continue

                try:
                    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Skipping %s - bad metadata: %s", trial_dir, exc)
                    continue

                action_id = int(meta["actionId"])
                records.append({
                    "video_path": mp4_files[0],
                    "imu_path": txt_files[0],
                    "subject_id": int(meta["subjectId"]),
                    "action_id": action_id,
                    "l1_category": _action_to_l1(action_id),
                    "l2_subtype": _action_to_l2(action_id),
                    "location_id": int(meta["locationId"]),
                    "side": meta.get("side", ""),
                })

    if max_videos is not None:
        # Balanced sample across L1 categories, sorted by subject for diversity
        by_cat: dict[str, list[dict]] = {}
        for r in records:
            by_cat.setdefault(r["l1_category"], []).append(r)

        n_cats = len(by_cat)
        per_cat = max(1, max_videos // n_cats)
        sampled: list[dict] = []
        for cat_records in by_cat.values():
            cat_records.sort(key=lambda r: r["subject_id"])
            sampled.extend(cat_records[:per_cat])
        records = sampled[:max_videos]

    return records


# ---------------------------------------------------------------------------
# IMU parsing
# ---------------------------------------------------------------------------

def _parse_imu_timestamp(ts_str: str) -> float:
    """Parse '2024-9-11 13:51:8:73' -> seconds (float). Returns nan on failure."""
    try:
        date_part, time_part = ts_str.strip().split(" ", 1)
        y, mo, d = date_part.split("-")
        segments = time_part.split(":")
        h, mi, s = int(segments[0]), int(segments[1]), int(segments[2])
        ms = int(segments[3]) if len(segments) > 3 else 0
        return int(h) * 3600 + int(mi) * 60 + int(s) + ms / 1000.0
    except (ValueError, IndexError):
        return np.nan


def parse_imu_file(txt_path: str) -> pd.DataFrame:
    """Parse IMU .txt file and return a cleaned DataFrame.

    Returns columns: timestamp_s, AccX, AccY, AccZ, GyroX, GyroY, GyroZ,
    AngleX, AngleY, AngleZ.
    """
    try:
        raw = pd.read_csv(txt_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:
        logger.error("Cannot read IMU file %s: %s", txt_path, exc)
        return pd.DataFrame()

    col_map: dict[str, str] = {}
    for col in raw.columns:
        lo = col.lower()
        if "accx" in lo:
            col_map[col] = "AccX"
        elif "accy" in lo:
            col_map[col] = "AccY"
        elif "accz" in lo:
            col_map[col] = "AccZ"
        elif "asx" in lo:
            col_map[col] = "GyroX"
        elif "asy" in lo:
            col_map[col] = "GyroY"
        elif "asz" in lo:
            col_map[col] = "GyroZ"
        elif "anglex" in lo:
            col_map[col] = "AngleX"
        elif "angley" in lo:
            col_map[col] = "AngleY"
        elif "anglez" in lo:
            col_map[col] = "AngleZ"
        elif lo == "time" or lo.startswith("time"):
            col_map[col] = "time"

    df = raw.rename(columns=col_map)

    required_imu = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    for c in required_imu:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["AngleX", "AngleY", "AngleZ"]:
        if c not in df.columns:
            df[c] = np.nan

    if "time" in df.columns:
        timestamps = df["time"].apply(lambda t: _parse_imu_timestamp(str(t)))
        first_valid = timestamps.dropna().iloc[0] if not timestamps.dropna().empty else 0.0
        df["timestamp_s"] = timestamps - first_valid
    else:
        df["timestamp_s"] = np.arange(len(df)) / 50.0  # assume 50 Hz if no timestamps

    cols = ["timestamp_s", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
            "AngleX", "AngleY", "AngleZ"]
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pose feature extraction
# ---------------------------------------------------------------------------

_MODEL_PATH = str(BASE_DIR / "pose_landmarker_full.task")
_ROLLING_WIN = 30  # frames for rolling-window pose features


class PoseFeatureExtractor:
    """Extract 23 pose features per frame using MediaPipe PoseLandmarker (VIDEO mode)."""

    def __init__(self, model_path: str = _MODEL_PATH):
        self._model_path = model_path
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process_video(self, video_path: str) -> pd.DataFrame:
        """Return per-frame feature DataFrame.  df.attrs['fps'] is set."""
        # Recreate landmarker to avoid timestamp issues across videos
        self._landmarker.close()
        self._landmarker = self._create_landmarker()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return pd.DataFrame()

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_features: list[dict] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_features.append(self._extract_frame_features(result, frame, frame_idx, fps))
            frame_idx += 1

        cap.release()

        if not frame_features:
            return pd.DataFrame()

        df = pd.DataFrame(frame_features)
        df = self._compute_temporal_features(df, fps)
        df.attrs["fps"] = fps
        return df

    def _extract_frame_features(self, result, frame, frame_idx: int, fps: float) -> dict:
        features: dict = {"frame_idx": frame_idx, "timestamp": frame_idx / fps}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features["frame_brightness"] = float(np.mean(gray))

        _nan_keys = [
            "torso_inclination", "lateral_lean", "pose_confidence",
            "left_leg_angle", "right_leg_angle", "aspect_ratio", "ankle_distance",
            "cog_x", "cog_y", "nose_y", "body_height",
            "l_ankle_y", "r_ankle_y",
            "l_wrist_x", "l_wrist_y", "r_wrist_x", "r_wrist_y",
            "limb_dist_l_wrist", "limb_dist_r_wrist",
            "limb_dist_l_ankle", "limb_dist_r_ankle",
        ]
        if not result.pose_landmarks:
            for k in _nan_keys:
                features[k] = np.nan
            return features

        lm = result.pose_landmarks[0]

        nose    = np.array([lm[LM_NOSE].x,           lm[LM_NOSE].y])
        l_sh    = np.array([lm[LM_LEFT_SHOULDER].x,  lm[LM_LEFT_SHOULDER].y])
        r_sh    = np.array([lm[LM_RIGHT_SHOULDER].x, lm[LM_RIGHT_SHOULDER].y])
        l_hip   = np.array([lm[LM_LEFT_HIP].x,       lm[LM_LEFT_HIP].y])
        r_hip   = np.array([lm[LM_RIGHT_HIP].x,      lm[LM_RIGHT_HIP].y])
        l_knee  = np.array([lm[LM_LEFT_KNEE].x,      lm[LM_LEFT_KNEE].y])
        r_knee  = np.array([lm[LM_RIGHT_KNEE].x,     lm[LM_RIGHT_KNEE].y])
        l_ankle = np.array([lm[LM_LEFT_ANKLE].x,     lm[LM_LEFT_ANKLE].y])
        r_ankle = np.array([lm[LM_RIGHT_ANKLE].x,    lm[LM_RIGHT_ANKLE].y])
        l_wrist = np.array([lm[LM_LEFT_WRIST].x,     lm[LM_LEFT_WRIST].y])
        r_wrist = np.array([lm[LM_RIGHT_WRIST].x,    lm[LM_RIGHT_WRIST].y])

        mid_sh    = (l_sh + r_sh) / 2
        mid_hip   = (l_hip + r_hip) / 2
        mid_ankle = (l_ankle + r_ankle) / 2

        torso_vec = mid_sh - mid_hip
        cos_a = np.dot(torso_vec, [0.0, -1.0]) / (np.linalg.norm(torso_vec) + 1e-8)
        features["torso_inclination"] = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        features["lateral_lean"]      = float(abs(mid_sh[0] - mid_hip[0]))
        features["pose_confidence"]   = float(np.mean([lm[i].visibility for i in range(33)]))
        features["left_leg_angle"]    = self._angle(l_hip, l_knee, l_ankle)
        features["right_leg_angle"]   = self._angle(r_hip, r_knee, r_ankle)

        vis_idx = [i for i in range(33) if lm[i].visibility > 0.5]
        if len(vis_idx) >= 2:
            xs = [lm[i].x for i in vis_idx]
            ys = [lm[i].y for i in vis_idx]
            features["aspect_ratio"] = float((max(xs) - min(xs)) / (max(ys) - min(ys) + 1e-8))
        else:
            features["aspect_ratio"] = np.nan

        features["ankle_distance"] = float(np.linalg.norm(l_ankle - r_ankle))
        features["cog_x"]          = float(mid_hip[0])
        features["cog_y"]          = float(mid_hip[1])
        features["nose_y"]         = float(nose[1])
        features["body_height"]    = float(np.linalg.norm(nose - mid_ankle))
        features["l_ankle_y"]      = float(l_ankle[1])
        features["r_ankle_y"]      = float(r_ankle[1])
        features["l_wrist_x"]      = float(l_wrist[0])
        features["l_wrist_y"]      = float(l_wrist[1])
        features["r_wrist_x"]      = float(r_wrist[0])
        features["r_wrist_y"]      = float(r_wrist[1])

        cog = mid_hip
        features["limb_dist_l_wrist"] = float(np.linalg.norm(l_wrist - cog))
        features["limb_dist_r_wrist"] = float(np.linalg.norm(r_wrist - cog))
        features["limb_dist_l_ankle"] = float(np.linalg.norm(l_ankle - cog))
        features["limb_dist_r_ankle"] = float(np.linalg.norm(r_ankle - cog))

        return features

    def _compute_temporal_features(self, df: pd.DataFrame, fps: float) -> pd.DataFrame:
        dt  = 1.0 / fps
        win = _ROLLING_WIN

        def _fill(col: str) -> np.ndarray:
            return df[col].ffill().bfill().values

        cog_x       = _fill("cog_x")
        cog_y       = _fill("cog_y")
        nose_y      = _fill("nose_y")
        body_height = _fill("body_height")
        aspect_r    = _fill("aspect_ratio")
        l_ankle_y   = _fill("l_ankle_y")
        r_ankle_y   = _fill("r_ankle_y")

        vel_x = np.gradient(cog_x, dt)
        vel_y = np.gradient(cog_y, dt)
        df["cog_velocity_x"]        = vel_x
        df["cog_velocity_y"]        = vel_y
        df["cog_accel_y"]           = np.gradient(vel_y, dt)
        df["cog_displacement_rate"] = np.sqrt(vel_x ** 2 + vel_y ** 2)
        df["head_drop_rate"]        = np.gradient(nose_y, dt)

        baseline = float(df["body_height"].iloc[:30].mean()) if len(df) >= 5 else 1.0
        df["body_height_ratio"]       = df["body_height"] / (baseline + 1e-8)
        df["body_height_change_rate"] = np.gradient(body_height, dt)
        df["aspect_ratio_change"]     = np.gradient(aspect_r, dt)

        cog_x_s = pd.Series(cog_x)
        cog_y_s = pd.Series(cog_y)
        df["com_sway_area"] = (
            cog_x_s.rolling(win, min_periods=5).std() *
            cog_y_s.rolling(win, min_periods=5).std()
        ).values

        ankle_dist = df["ankle_distance"].ffill().bfill()
        df["base_of_support_variability"] = ankle_dist.rolling(win, min_periods=5).std().values

        ankle_diff = pd.Series(l_ankle_y - r_ankle_y)
        df["stride_regularity"] = ankle_diff.rolling(win, min_periods=10).apply(
            _autocorr_lag5, raw=True
        ).values

        l_a_s = pd.Series(l_ankle_y)
        r_a_s = pd.Series(r_ankle_y)
        l_std = l_a_s.rolling(win, min_periods=5).std()
        r_std = r_a_s.rolling(win, min_periods=5).std()
        df["step_symmetry"] = (
            np.minimum(l_std, r_std) / (np.maximum(l_std, r_std) + 1e-8)
        ).values

        frame_disp = np.sqrt(np.diff(cog_x, prepend=cog_x[0]) ** 2 +
                             np.diff(cog_y, prepend=cog_y[0]) ** 2)
        disp_s = pd.Series(frame_disp)
        df["motion_periodicity"] = disp_s.rolling(win, min_periods=10).apply(
            _fft_dominant_magnitude, raw=True
        ).values

        df["vertical_oscillation_count"] = pd.Series(vel_y).rolling(win, min_periods=5).apply(
            _zero_crossing_count, raw=True
        ).values

        df["post_window_stillness"] = 1.0 / (disp_s.rolling(win, min_periods=5).std() + 1e-8)

        # limb_angular_momentum: sum of (limb_distance * limb_speed) for 4 limbs
        vel_lwx = np.gradient(_fill("l_wrist_x"), dt)
        vel_lwy = np.gradient(_fill("l_wrist_y"), dt)
        vel_rwx = np.gradient(_fill("r_wrist_x"), dt)
        vel_rwy = np.gradient(_fill("r_wrist_y"), dt)
        vel_la  = np.gradient(l_ankle_y, dt)
        vel_ra  = np.gradient(r_ankle_y, dt)

        df["limb_angular_momentum"] = (
            _fill("limb_dist_l_wrist") * np.sqrt(vel_lwx ** 2 + vel_lwy ** 2) +
            _fill("limb_dist_r_wrist") * np.sqrt(vel_rwx ** 2 + vel_rwy ** 2) +
            _fill("limb_dist_l_ankle") * np.abs(vel_la) +
            _fill("limb_dist_r_ankle") * np.abs(vel_ra)
        )

        return df

    def process_frame(self, frame, frame_idx: int, fps: float) -> dict:
        """Process a single frame (for real-time use)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(frame_idx * 1000 / fps))
        return self._extract_frame_features(result, frame, frame_idx, fps)

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


# Rolling-window helpers (module-level for pickle / apply compat)

def _autocorr_lag5(x: np.ndarray) -> float:
    if len(x) < 6:
        return 0.0
    lag = min(5, len(x) // 4)
    s = pd.Series(x)
    result = s.autocorr(lag=lag)
    return float(result) if not np.isnan(result) else 0.0


def _fft_dominant_magnitude(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    x_demean = x - x.mean()
    magnitudes = np.abs(np.fft.rfft(x_demean))
    return float(magnitudes[1:].max()) if len(magnitudes) > 1 else 0.0


def _zero_crossing_count(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(((x[:-1] * x[1:]) < 0).sum())


# ---------------------------------------------------------------------------
# IMU feature extraction
# ---------------------------------------------------------------------------

def extract_imu_features(imu_df: pd.DataFrame, n_frames: int, fps: float) -> pd.DataFrame:
    """Align IMU data to n_frames and return 8 IMU features per frame."""
    imu_cols = ["imu_acc_magnitude", "imu_acc_magnitude_variance",
                "imu_orientation_stability", "imu_gyro_periodicity",
                "peak_impact_g", "jerk_magnitude",
                "orientation_change_rate", "vertical_acc_asymmetry"]

    empty = pd.DataFrame({c: np.full(n_frames, np.nan) for c in imu_cols})

    if imu_df.empty or n_frames == 0:
        return empty

    n_imu = len(imu_df)
    indices = np.linspace(0, n_imu - 1, n_frames).astype(int)
    aligned = imu_df.iloc[indices].reset_index(drop=True)

    def to_f(col: str) -> np.ndarray:
        if col not in aligned.columns:
            return np.zeros(n_frames)
        return pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).values

    acc_x  = to_f("AccX")
    acc_y  = to_f("AccY")
    acc_z  = to_f("AccZ")
    gyro_x = to_f("GyroX")
    gyro_y = to_f("GyroY")
    gyro_z = to_f("GyroZ")
    ang_x  = to_f("AngleX")
    ang_y  = to_f("AngleY")
    ang_z  = to_f("AngleZ")

    acc_mag  = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    gyro_mag = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)
    ang_mag  = np.sqrt(ang_x ** 2 + ang_y ** 2 + ang_z ** 2)
    dt       = 1.0 / fps
    win      = min(_ROLLING_WIN, max(1, n_frames // 2))

    mag_s = pd.Series(acc_mag)
    ang_s = pd.Series(ang_mag)

    result = pd.DataFrame()
    result["imu_acc_magnitude"]          = acc_mag
    result["imu_acc_magnitude_variance"] = mag_s.rolling(win, min_periods=1).var().values
    result["peak_impact_g"]              = mag_s.rolling(win, min_periods=1).max().values
    result["jerk_magnitude"]             = np.abs(np.gradient(acc_mag, dt))
    result["imu_orientation_stability"]  = (
        1.0 / (ang_s.rolling(win, min_periods=1).std() + 1e-8)
    ).values
    result["orientation_change_rate"]    = np.abs(np.gradient(ang_mag, dt))
    result["imu_gyro_periodicity"]       = pd.Series(gyro_mag).rolling(win, min_periods=10).apply(
        _fft_dominant_magnitude, raw=True
    ).values
    result["vertical_acc_asymmetry"]     = pd.Series(acc_y).rolling(win, min_periods=5).apply(
        _vertical_asymmetry, raw=True
    ).values

    return result


def _vertical_asymmetry(x: np.ndarray) -> float:
    pos = float(x[x > 0].sum())
    neg = float((-x[x < 0]).sum())
    return (pos - neg) / (pos + neg + 1e-8)


# ---------------------------------------------------------------------------
# Sliding window aggregation
# ---------------------------------------------------------------------------

def apply_sliding_window(
    frame_df: pd.DataFrame,
    window_size: int = 50,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """Compute mean/std/max/min/range over sliding windows for FEATURE_COLUMNS.

    Returns a DataFrame with 32 x 5 = 160 windowed feature columns.
    """
    step = max(1, int(window_size * (1 - overlap)))
    windows: list[dict] = []

    for start in range(0, len(frame_df) - window_size + 1, step):
        window = frame_df.iloc[start : start + window_size]
        row: dict = {}
        for col in FEATURE_COLUMNS:
            s = window[col].dropna() if col in frame_df.columns else pd.Series(dtype=float)
            if len(s) == 0:
                row[f"{col}_mean"]  = np.nan
                row[f"{col}_std"]   = np.nan
                row[f"{col}_max"]   = np.nan
                row[f"{col}_min"]   = np.nan
                row[f"{col}_range"] = np.nan
            else:
                row[f"{col}_mean"]  = float(s.mean())
                row[f"{col}_std"]   = float(s.std()) if len(s) > 1 else 0.0
                row[f"{col}_max"]   = float(s.max())
                row[f"{col}_min"]   = float(s.min())
                row[f"{col}_range"] = float(s.max() - s.min())
        windows.append(row)

    return pd.DataFrame(windows)


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def run_extraction(data_root: Path, max_videos: Optional[int] = None) -> None:
    """Discover trials, extract pose + IMU features, save windowed_features.parquet."""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_cfg = CONFIG.get("extraction", {})
    window_size = int(ext_cfg.get("window_size", 50))
    overlap     = float(ext_cfg.get("overlap", 0.5))

    trials = discover_videos(data_root, max_videos)
    logger.info("Found %d trials to process", len(trials))

    extractor    = PoseFeatureExtractor()
    all_windowed: list[pd.DataFrame] = []

    for i, trial in enumerate(trials):
        label = f"[{i+1}/{len(trials)}] {Path(trial['video_path']).parent.name}"
        logger.info(label)

        pose_df = extractor.process_video(str(trial["video_path"]))
        if pose_df.empty:
            logger.warning("  Skipping - empty pose DataFrame")
            continue

        fps      = pose_df.attrs.get("fps", 20.0)
        n_frames = len(pose_df)

        imu_df       = parse_imu_file(str(trial["imu_path"]))
        imu_features = extract_imu_features(imu_df, n_frames, fps)

        # Merge IMU features into pose DataFrame
        for col in imu_features.columns:
            pose_df[col] = imu_features[col].values

        # Apply sliding window
        windowed = apply_sliding_window(pose_df, window_size, overlap)
        if windowed.empty:
            logger.warning("  Skipping - too few frames for windowing (%d < %d)", n_frames, window_size)
            continue

        windowed["subject_id"]  = trial["subject_id"]
        windowed["action_id"]   = trial["action_id"]
        windowed["l1_category"] = trial["l1_category"]
        windowed["l2_subtype"]  = trial["l2_subtype"]
        windowed["location_id"] = trial["location_id"]
        windowed["side"]        = trial["side"]
        windowed["video_path"]  = str(trial["video_path"])
        all_windowed.append(windowed)

    extractor.close()

    if all_windowed:
        result_df = pd.concat(all_windowed, ignore_index=True)
        out_path  = output_dir / "windowed_features.parquet"
        result_df.to_parquet(out_path, index=False)
        logger.info(
            "Saved %d windowed samples (%d videos) -> %s",
            len(result_df), len(all_windowed), out_path,
        )
    else:
        logger.warning("No windowed features produced - output not saved")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_FEATURE_STAT_SUFFIXES = ("_mean", "_std", "_max", "_min", "_range")


def create_model(model_name: str):
    """Return an unfitted sklearn-compatible estimator for the given model name."""
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced", random_state=42,
        )
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            objective="multiclass", random_state=42, verbose=-1,
        )
    if model_name == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=42), ensemble=False)
    if model_name == "ensemble":
        return _create_ensemble()
    raise ValueError(f"Unknown model: {model_name}")


def _create_ensemble() -> StackingClassifier:
    estimators = [
        ("xgboost",       create_model("xgboost")),
        ("random_forest", create_model("random_forest")),
        ("lightgbm",      create_model("lightgbm")),
        ("svm",           create_model("svm")),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
    )


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.endswith(s) for s in _FEATURE_STAT_SUFFIXES)]


def _compute_sample_weight(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    total = len(y)
    weights = np.array([total / (len(classes) * freq[cls]) for cls in y])
    return weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name: str, level: str = "both") -> None:
    feat_path = FEATURES_DIR / "windowed_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found at %s - run --extract first", feat_path)
        return

    df = pd.read_parquet(feat_path)
    feature_cols = _get_feature_cols(df)
    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X = np.nan_to_num(X, nan=0.0)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")
    logger.info("Saved scaler and feature columns (%d cols)", len(feature_cols))

    groups = df["subject_id"].values
    use_weight = model_name in ("xgboost", "lightgbm")

    if level in ("l1", "both"):
        le_l1 = LabelEncoder()
        y_l1 = le_l1.fit_transform(df["l1_category"].values)
        n_unique_groups = len(np.unique(groups))
        n_splits = min(5, n_unique_groups)
        if n_splits < 2:
            logger.warning("Only %d subject group(s) - using simple train/test split instead of CV", n_unique_groups)
            n_splits = 2
            groups_for_cv = np.arange(len(X))  # treat each sample as its own group
        else:
            groups_for_cv = groups
        cv = StratifiedGroupKFold(n_splits=n_splits)

        model = create_model(model_name)
        acc_scores, f1_scores = [], []
        for train_idx, val_idx in cv.split(X, y_l1, groups=groups_for_cv):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_l1[train_idx], y_l1[val_idx]
            fit_kwargs = {}
            if use_weight:
                fit_kwargs["sample_weight"] = _compute_sample_weight(y_tr)
            model.fit(X_tr, y_tr, **fit_kwargs)
            preds = model.predict(X_val)
            acc_scores.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

        logger.info(
            "L1 CV (%s): acc=%.3f±%.3f  f1_macro=%.3f±%.3f",
            model_name,
            np.mean(acc_scores), np.std(acc_scores),
            np.mean(f1_scores), np.std(f1_scores),
        )

        # Final fit on all data
        final_model = create_model(model_name)
        fit_kwargs = {}
        if use_weight:
            fit_kwargs["sample_weight"] = _compute_sample_weight(y_l1)
        final_model.fit(X, y_l1, **fit_kwargs)
        joblib.dump({"model": final_model, "label_encoder": le_l1},
                    MODEL_DIR / f"{model_name}_l1.joblib")
        logger.info("Saved %s_l1.joblib", model_name)

    if level in ("l2", "both"):
        for l1_cat in df["l1_category"].unique():
            mask = df["l1_category"] == l1_cat
            sub_df = df[mask]
            if sub_df["l2_subtype"].nunique() < 2:
                logger.warning("Skipping L2 for '%s' - only 1 subtype", l1_cat)
                continue

            X_sub = X[mask]
            grp_sub = groups[mask]
            le_l2 = LabelEncoder()
            y_l2 = le_l2.fit_transform(sub_df["l2_subtype"].values)

            n_unique_grp = len(np.unique(grp_sub))
            n_splits_l2 = min(5, n_unique_grp)
            if n_splits_l2 < 2:
                n_splits_l2 = 2
                grp_sub_cv = np.arange(len(X_sub))
            else:
                grp_sub_cv = grp_sub
            cv = StratifiedGroupKFold(n_splits=n_splits_l2)

            acc_scores, f1_scores = [], []
            model = create_model(model_name)
            for train_idx, val_idx in cv.split(X_sub, y_l2, groups=grp_sub_cv):
                X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
                y_tr, y_val = y_l2[train_idx], y_l2[val_idx]
                fit_kwargs = {}
                if use_weight:
                    fit_kwargs["sample_weight"] = _compute_sample_weight(y_tr)
                model.fit(X_tr, y_tr, **fit_kwargs)
                preds = model.predict(X_val)
                acc_scores.append(accuracy_score(y_val, preds))
                f1_scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

            logger.info(
                "L2[%s] CV (%s): acc=%.3f±%.3f  f1_macro=%.3f±%.3f",
                l1_cat, model_name,
                np.mean(acc_scores), np.std(acc_scores),
                np.mean(f1_scores), np.std(f1_scores),
            )

            final_model = create_model(model_name)
            fit_kwargs = {}
            if use_weight:
                fit_kwargs["sample_weight"] = _compute_sample_weight(y_l2)
            final_model.fit(X_sub, y_l2, **fit_kwargs)
            joblib.dump({"model": final_model, "label_encoder": le_l2},
                        MODEL_DIR / f"{model_name}_l2_{l1_cat}.joblib")
            logger.info("Saved %s_l2_%s.joblib", model_name, l1_cat)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, classes: list[str], title: str, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes))))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        logger.info("Saved confusion matrix: %s", save_path)
    except ImportError:
        logger.warning("matplotlib/seaborn not available - skipping plot")


def evaluate_model(model_name: str, level: str = "l1") -> dict:
    """Cross-validate model, print metrics, plot confusion matrix. Returns metrics dict."""
    feat_path = FEATURES_DIR / "windowed_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found - run --extract first")
        return {}

    df = pd.read_parquet(feat_path)
    feature_cols = _get_feature_cols(df)

    scaler_path = MODEL_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        saved_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
        X = np.nan_to_num(scaler.transform(df[saved_cols].values), nan=0.0)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols].values)

    groups = df["subject_id"].values
    results: dict = {}

    if level == "l1":
        le = LabelEncoder()
        y = le.fit_transform(df["l1_category"].values)
        n_unique = len(np.unique(groups))
        n_splits = min(5, n_unique)
        if n_splits < 2:
            n_splits = 2
            groups_cv = np.arange(len(X))
        else:
            groups_cv = groups
        cv = StratifiedGroupKFold(n_splits=n_splits)

        model = create_model(model_name)
        all_true, all_pred = [], []
        for train_idx, val_idx in cv.split(X, y, groups=groups_cv):
            fit_kwargs = {}
            if model_name in ("xgboost", "lightgbm"):
                fit_kwargs["sample_weight"] = _compute_sample_weight(y[train_idx])
            model.fit(X[train_idx], y[train_idx], **fit_kwargs)
            all_true.extend(y[val_idx])
            all_pred.extend(model.predict(X[val_idx]))

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        acc = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        print(f"\n=== {model_name} L1 Evaluation ===")
        print(f"Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
        print(classification_report(all_true, all_pred,
                                    target_names=le.classes_, zero_division=0))

        cm = confusion_matrix(all_true, all_pred)
        _plot_confusion_matrix(
            cm, list(le.classes_),
            f"{model_name} L1 Confusion Matrix",
            PLOTS_DIR / f"{model_name}_l1_confusion.png",
        )
        results["l1"] = {"accuracy": acc, "f1_macro": f1}

    return results


def run_evaluation(model_name: str, level: str = "l1") -> None:
    if model_name == "all":
        all_results = {}
        for name in ("xgboost", "random_forest", "lightgbm", "svm", "ensemble"):
            logger.info("Evaluating %s ...", name)
            res = evaluate_model(name, level)
            if res:
                all_results[name] = res.get(level, {})

        if all_results:
            print("\n=== Model Comparison ===")
            print(f"{'Model':<20} {'Accuracy':>10} {'F1-macro':>10}")
            print("-" * 42)
            for name, metrics in all_results.items():
                print(f"{name:<20} {metrics.get('accuracy', 0):.4f}     {metrics.get('f1_macro', 0):.4f}")

            _save_comparison_plot(all_results)
    else:
        evaluate_model(model_name, level)


def _save_comparison_plot(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(results.keys())
        accs = [results[m].get("accuracy", 0) for m in models]
        f1s = [results[m].get("f1_macro", 0) for m in models]

        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, accs, width, label="Accuracy")
        ax.bar(x + width/2, f1s, width, label="F1-macro")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("Model Comparison")
        plt.tight_layout()
        out = PLOTS_DIR / "model_comparison.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        logger.info("Saved model comparison: %s", out)
    except ImportError:
        logger.warning("matplotlib not available - skipping comparison plot")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_video(video_path: str, model_name: str, detailed: bool = False) -> None:
    scaler_path   = MODEL_DIR / "scaler.joblib"
    feat_col_path = MODEL_DIR / "feature_columns.joblib"
    l1_path       = MODEL_DIR / f"{model_name}_l1.joblib"

    for p in (scaler_path, feat_col_path, l1_path):
        if not p.exists():
            logger.error("Required file not found: %s - run --train first", p)
            return

    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(feat_col_path)
    l1_bundle    = joblib.load(l1_path)
    l1_model     = l1_bundle["model"]
    l1_le        = l1_bundle["label_encoder"]

    # Look for IMU file in same directory
    vid_dir = Path(video_path).parent
    txt_files = list(vid_dir.glob("*.txt"))
    imu_df = parse_imu_file(str(txt_files[0])) if txt_files else pd.DataFrame()

    extractor = PoseFeatureExtractor()
    pose_df   = extractor.process_video(video_path)
    extractor.close()

    if pose_df.empty:
        logger.error("Could not extract pose from video: %s", video_path)
        return

    fps      = pose_df.attrs.get("fps", 20.0)
    n_frames = len(pose_df)
    imu_feat = extract_imu_features(imu_df, n_frames, fps)
    for col in imu_feat.columns:
        pose_df[col] = imu_feat[col].values

    ext_cfg     = CONFIG.get("extraction", {})
    window_size = int(ext_cfg.get("window_size", 50))
    overlap     = float(ext_cfg.get("overlap", 0.5))
    windowed    = apply_sliding_window(pose_df, window_size, overlap)

    if windowed.empty:
        logger.error("Video too short for windowing (%d frames)", n_frames)
        return

    # Align to saved feature columns, fill any missing with 0
    for col in feature_cols:
        if col not in windowed.columns:
            windowed[col] = 0.0
    X = np.nan_to_num(scaler.transform(windowed[feature_cols].values), nan=0.0)

    # --- Raw per-window L1 predictions ---
    l1_preds_raw = l1_model.predict(X)
    l1_labels_raw = l1_le.inverse_transform(l1_preds_raw)

    # --- Temporal smoothing: sliding majority vote (window=5) ---
    smooth_k = 5
    l1_labels = l1_labels_raw.copy()
    if len(l1_labels) >= smooth_k:
        smoothed = []
        half_k = smooth_k // 2
        for i in range(len(l1_labels)):
            start = max(0, i - half_k)
            end = min(len(l1_labels), i + half_k + 1)
            counts = Counter(l1_labels_raw[start:end])
            smoothed.append(counts.most_common(1)[0][0])
        l1_labels = np.array(smoothed)

    # --- L2 predictions with smoothed L1 ---
    rows = []
    l2_cache = {}
    for i, l1_label in enumerate(l1_labels):
        l2_label = "n/a"
        l2_path = MODEL_DIR / f"{model_name}_l2_{l1_label}.joblib"
        if l2_path.exists():
            if l1_label not in l2_cache:
                l2_cache[l1_label] = joblib.load(l2_path)
            l2_bundle = l2_cache[l1_label]
            l2_pred = l2_bundle["model"].predict(X[i : i + 1])
            l2_label = l2_bundle["label_encoder"].inverse_transform(l2_pred)[0]
        rows.append({"window": i, "l1": l1_label, "l2": l2_label,
                     "l1_raw": l1_labels_raw[i]})

    result_df = pd.DataFrame(rows)

    # --- L2 smoothing: sliding majority within dominant L1 ---
    l2_vals = result_df["l2"].values.copy()
    if len(l2_vals) >= smooth_k:
        smoothed_l2 = []
        for i in range(len(l2_vals)):
            start = max(0, i - half_k)
            end = min(len(l2_vals), i + half_k + 1)
            neighborhood = [l2_vals[j] for j in range(start, end) if l2_vals[j] != "n/a"]
            if neighborhood:
                counts = Counter(neighborhood)
                smoothed_l2.append(counts.most_common(1)[0][0])
            else:
                smoothed_l2.append(l2_vals[i])
        result_df["l2"] = smoothed_l2

    l1_counts = result_df["l1"].value_counts()
    top_l1 = l1_counts.index[0]
    top_l2 = result_df[result_df["l1"] == top_l1]["l2"].value_counts().index[0]

    print(f"\nPrediction for: {Path(video_path).name}")
    print(f"  L1 (majority): {top_l1} ({l1_counts[top_l1]}/{len(result_df)} windows)")
    print(f"  L2 (majority): {top_l2}")

    if detailed:
        print("\nPer-window predictions (smoothed):")
        print(result_df[["window", "l1", "l2"]].to_string(index=False))
        raw_l1_counts = Counter(l1_labels_raw)
        smoothed_l1_counts = Counter(l1_labels)
        if raw_l1_counts != smoothed_l1_counts:
            print(f"\n  Smoothing effect on L1: {dict(raw_l1_counts)} -> {dict(smoothed_l1_counts)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Activity Classifier - FigShare dual-modality pipeline")
    parser.add_argument("--extract",          action="store_true", help="Extract features from videos")
    parser.add_argument("--train",            action="store_true", help="Train classifiers")
    parser.add_argument("--evaluate",         action="store_true", help="Evaluate trained models")
    parser.add_argument("--predict",          type=str,            help="Predict on a video file")
    parser.add_argument("--predict-detailed", type=str,            help="Detailed per-window prediction")
    parser.add_argument("--model",            type=str,            default="xgboost",
                        choices=["xgboost", "random_forest", "lightgbm", "svm", "ensemble", "all"],
                        help="Model to use for training/prediction")
    parser.add_argument("--max-videos",       type=int,            default=None,
                        help="Limit number of videos for extraction")
    parser.add_argument("--level",            type=str,            default="l1",
                        choices=["l1", "l2", "both"],
                        help="Classification level (l1=4-class, l2=fine-grained, both=hierarchical)")
    parser.add_argument("--config",           type=str,            default=None,
                        help="Path to config.json")
    parser.add_argument("--data-root",        type=str,            default=None,
                        help="Dataset root directory")
    args = parser.parse_args()

    if args.config:
        global CONFIG
        CONFIG = load_config(Path(args.config))

    data_root = Path(args.data_root) if args.data_root else Path(CONFIG["data_root"])

    if not any([args.extract, args.train, args.evaluate, args.predict, args.predict_detailed]):
        parser.print_help()
        sys.exit(0)

    if args.extract:
        run_extraction(data_root, args.max_videos)

    if args.train:
        if args.model == "all":
            for m in ("xgboost", "random_forest", "lightgbm", "svm", "ensemble"):
                logger.info("Training %s ...", m)
                train_model(m, level=args.level)
        else:
            logger.info("Training %s ...", args.model)
            train_model(args.model, level=args.level)

    if args.evaluate:
        run_evaluation(args.model, level=args.level)

    if args.predict:
        predict_video(args.predict, args.model, detailed=False)

    if args.predict_detailed:
        predict_video(args.predict_detailed, args.model, detailed=True)


if __name__ == "__main__":
    main()