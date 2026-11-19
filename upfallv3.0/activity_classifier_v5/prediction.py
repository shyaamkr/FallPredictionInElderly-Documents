"""Prediction pipeline - tri-modal (video + IMU + wearable), fall confirmation."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity_classifier import (
    parse_imu_file,
    PoseFeatureExtractor,
    extract_imu_features,
    apply_sliding_window,
)

from .config import MODEL_DIR, CONFIG, SMOOTH_K, logger
from .features import _compute_derived_features
from .smoothing import temporal_smooth, confidence_filter
from .l2 import get_l2_handler
from .visualization import generate_annotated_output

_COLLAPSE_TO_OTHER = {"stationary", "transitional"}


def _collapse_l1(labels: np.ndarray) -> np.ndarray:
    """Collapse stationary/transitional -> 'other' post-prediction."""
    return np.array(["other" if l in _COLLAPSE_TO_OTHER else l for l in labels])


def _confirm_fall(windowed_df, has_imu, has_wearable,
                  wearable_df=None) -> tuple[bool, dict]:
    """Kinematic + wearable confirmation that a fall actually occurred.

    Only fires when IMU is absent. Real falls must show physical evidence
    of collapse. Wearable signals provide additional supporting evidence.
    Returns (confirmed, details).
    """
    if has_imu:
        return True, {"skipped": True, "reason": "has_imu"}

    details = {}
    score = 0.0
    n_windows = len(windowed_df)

    torso_vals = windowed_df.get("torso_inclination_mean")
    bh_change_vals = windowed_df.get("body_height_change_rate_mean")
    aspect_vals = windowed_df.get("aspect_ratio_mean")
    head_drop_vals = windowed_df.get("head_drop_rate_mean")
    stillness_vals = windowed_df.get("post_window_stillness_mean")

    # Signal 1 (weight=2): Peak torso inclination > 20 degrees
    peak_torso = float(torso_vals.max()) if torso_vals is not None else 0.0
    details["peak_torso_incl"] = round(peak_torso, 1)
    if peak_torso > 20.0:
        score += 2.0

    # Signal 2 (weight=2): Body height drop < -0.15
    min_bh_change = float(bh_change_vals.min()) if bh_change_vals is not None else 0.0
    details["min_body_height_change"] = round(min_bh_change, 4)
    if min_bh_change < -0.15:
        score += 2.0

    # Signal 3 (weight=1): Aspect ratio > 1.0 (person horizontal)
    peak_aspect = float(aspect_vals.max()) if aspect_vals is not None else 0.0
    details["peak_aspect_ratio"] = round(peak_aspect, 3)
    if peak_aspect > 1.0:
        score += 1.0

    # Signal 4 (weight=1): Head drop velocity > 0.05
    peak_head_drop = float(head_drop_vals.max()) if head_drop_vals is not None else 0.0
    details["peak_head_drop_rate"] = round(peak_head_drop, 4)
    if peak_head_drop > 0.05:
        score += 1.0

    # Signal 5 (weight=1): Post-fall stillness increase (3x from start to end)
    if stillness_vals is not None and n_windows >= 3:
        start_still = float(stillness_vals.iloc[:2].mean())
        end_still = float(stillness_vals.iloc[-2:].mean())
        if start_still > 0 and end_still / max(start_still, 1e-8) > 3.0:
            score += 1.0
        details["stillness_ratio"] = round(
            end_still / max(start_still, 1e-8), 2)

    # Negative: All windows have torso < 10 degrees
    if torso_vals is not None and float(torso_vals.max()) < 10.0:
        score -= 3.0
        details["upright_penalty"] = True

    # Negative: No window has body height change < -0.05
    if bh_change_vals is not None and float(bh_change_vals.min()) >= -0.05:
        score -= 2.0
        details["no_drop_penalty"] = True

    # -- Wearable signals (v5) --
    if has_wearable and wearable_df is not None and not wearable_df.empty:
        n_rows = len(wearable_df)
        tail_start = max(0, int(n_rows * 0.7))
        tail_df = wearable_df.iloc[tail_start:]

        # Signal 6 (weight=1.5): HR anomaly in final 30%
        if "heart_rate_bpm" in wearable_df.columns and len(tail_df) > 0:
            tail_hr = tail_df["heart_rate_bpm"].values
            hr_anomaly = np.any((tail_hr < 55) | (tail_hr > 120))
            details["tail_hr_anomaly"] = bool(hr_anomaly)
            if hr_anomaly:
                score += 1.5

        # Signal 7 (weight=1.5): SpO2 dip anywhere in video
        if "spo2_pct" in wearable_df.columns:
            min_spo2 = float(wearable_df["spo2_pct"].min())
            details["min_spo2"] = round(min_spo2, 1)
            if min_spo2 < 93.0:
                score += 1.5

    details["score"] = round(score, 1)
    details["has_wearable"] = has_wearable
    threshold = 3.0
    confirmed = score >= threshold

    logger.info(
        "Fall confirmation: score=%.1f (threshold=%.1f) | "
        "torso=%.1f, bh_change=%.4f, aspect=%.3f, head_drop=%.4f, "
        "has_wearable=%s, confirmed=%s",
        score, threshold,
        details.get("peak_torso_incl", 0),
        details.get("min_body_height_change", 0),
        details.get("peak_aspect_ratio", 0),
        details.get("peak_head_drop_rate", 0),
        has_wearable,
        confirmed,
    )

    return confirmed, details


def _compute_distress_score(windowed_df, pose_df) -> float:
    """Detect visible distress postures from pose data (0-100).

    Looks for: bent-over posture (hands on knees), hands on chest,
    gasping/labored breathing rhythm, head drooping (fatigue/dizziness),
    and sudden movement cessation after activity.

    Thresholds calibrated against real MediaPipe pose output from
    evaluation videos (normalized 0-1 coordinate space).
    """
    score = 0.0
    n_win = len(windowed_df)
    if n_win == 0:
        return 0.0

    def _col_val(name, stat="mean", func="max"):
        col = windowed_df.get(f"{name}_{stat}")
        if col is None:
            return 0.0
        v = float(getattr(col, func)())
        return 0.0 if np.isnan(v) else v

    def _safe_float(series):
        v = float(series)
        return 0.0 if np.isnan(v) else v

    dbg = {}
    late_slice = max(2, n_win // 3)
    early_slice = max(2, n_win // 3)

    # --- Signal 1: Torso forward lean ---
    # MediaPipe produces 8-15 deg for subtle panting lean, 20+ for hands-on-knees
    peak_torso = _col_val("torso_inclination", "mean", "max")
    late_torso = 0.0
    torso_col = windowed_df.get("torso_inclination_mean")
    if torso_col is not None and n_win >= 3:
        late_torso = _safe_float(torso_col.iloc[-late_slice:].max())
    dbg["peak_torso"] = round(peak_torso, 2)
    dbg["late_torso"] = round(late_torso, 2)

    if late_torso > 20:
        score += 30
    elif late_torso > 12:
        score += 20
    elif peak_torso > 10:
        score += 10

    # --- Signal 2: Body height reduction ---
    bh_ratio_col = windowed_df.get("body_height_ratio_mean")
    min_bh_ratio = 1.0
    bh_nan = False
    if bh_ratio_col is not None:
        raw_min = float(bh_ratio_col.min())
        if np.isnan(raw_min):
            bh_nan = True
        else:
            min_bh_ratio = raw_min
            if min_bh_ratio < 0.70:
                score += 25
            elif min_bh_ratio < 0.85:
                score += 15
            elif min_bh_ratio < 0.92:
                score += 8
    dbg["min_bh_ratio"] = "NaN(+0)" if bh_nan else round(min_bh_ratio, 3)

    # --- Signal 3: Head drooping (fatigue, dizziness) ---
    late_head_drop = 0.0
    head_drop_col = windowed_df.get("head_drop_rate_mean")
    if head_drop_col is not None and n_win >= 3:
        late_head_drop = _safe_float(head_drop_col.iloc[-late_slice:].max())
        if late_head_drop > 0.02:
            score += 25
        elif late_head_drop > 0.008:
            score += 15
        elif late_head_drop > 0.003:
            score += 8
    dbg["late_head_drop"] = round(late_head_drop, 5)

    # --- Signal 4: Stillness / movement cessation ---
    # Check both ratio-based (early vs late) AND absolute late stillness
    disp_col = windowed_df.get("cog_displacement_rate_mean")
    still_col = windowed_df.get("post_window_stillness_mean")
    motion_drop = 0.0
    stillness_rise = 0.0
    abs_late_still = 0.0
    if still_col is not None and n_win >= 3:
        abs_late_still = _safe_float(still_col.iloc[-late_slice:].mean())
    if disp_col is not None and still_col is not None and n_win >= 4:
        half = n_win // 2
        early_disp = _safe_float(disp_col.iloc[:half].mean())
        late_disp = _safe_float(disp_col.iloc[half:].mean())
        late_still = _safe_float(still_col.iloc[half:].mean())
        early_still = _safe_float(still_col.iloc[:half].mean())
        motion_drop = early_disp / max(late_disp, 1e-8)
        stillness_rise = late_still / max(early_still, 1e-8)

    s4 = 0
    if motion_drop > 3.0 and stillness_rise > 2.0:
        s4 = 25
    elif motion_drop > 2.0 or stillness_rise > 1.5:
        s4 = 15
    if abs_late_still > 500:
        s4 = max(s4, 20)
    elif abs_late_still > 100:
        s4 = max(s4, 10)
    score += s4
    dbg["motion_drop"] = round(motion_drop, 2)
    dbg["stillness_rise"] = round(stillness_rise, 2)
    dbg["abs_late_still"] = round(abs_late_still, 1)

    # --- Signal 5: Labored breathing rhythm ---
    # Rhythmic vertical oscillation while relatively still
    late_osc = 0.0
    late_still_val = 0.0
    osc_col = windowed_df.get("vertical_oscillation_count_mean")
    if osc_col is not None and n_win >= 3:
        late_osc = _safe_float(osc_col.iloc[-late_slice:].mean())
        if still_col is not None:
            late_still_val = _safe_float(still_col.iloc[-late_slice:].mean())
        if late_osc > 2.0 and late_still_val > 50:
            score += 20
        elif late_osc > 1.5 and late_still_val > 10:
            score += 10
        elif late_osc > 2.0:
            score += 8
    dbg["late_osc"] = round(late_osc, 2)
    dbg["late_still_val"] = round(late_still_val, 2)

    # --- Signal 6: Limb behaviour change ---
    # Arms stop swinging OR change dramatically (repositioning to chest/knees)
    early_lam = 0.0
    late_lam = 0.0
    lam_col = windowed_df.get("limb_angular_momentum_mean")
    if lam_col is not None and n_win >= 3:
        late_lam = _safe_float(lam_col.iloc[-late_slice:].mean())
        early_lam = _safe_float(lam_col.iloc[:early_slice].mean())
        if early_lam > 1e-4:
            lam_ratio = late_lam / max(early_lam, 1e-8)
            if lam_ratio < 0.3 or lam_ratio > 4.0:
                score += 15
            elif lam_ratio < 0.5 or lam_ratio > 2.5:
                score += 8
    dbg["early_lam"] = round(early_lam, 5)
    dbg["late_lam"] = round(late_lam, 5)
    dbg["lam_ratio"] = round(late_lam / max(early_lam, 1e-8), 2) if early_lam > 1e-4 else "N/A"

    # --- Signal 7: Aspect ratio widening ---
    late_aspect = 0.0
    aspect_col = windowed_df.get("aspect_ratio_mean")
    if aspect_col is not None and n_win >= 3:
        late_aspect = _safe_float(aspect_col.iloc[-late_slice:].max())
        if late_aspect > 0.5:
            score += 12
        elif late_aspect > 0.3:
            score += 6
    dbg["late_aspect"] = round(late_aspect, 3)

    # --- Corroboration bonus: multiple distress signals reinforce each other ---
    distress_signals_fired = 0
    if late_torso > 12 or peak_torso > 10:
        distress_signals_fired += 1
    if late_head_drop > 0.008:
        distress_signals_fired += 1
    if abs_late_still > 100 or motion_drop > 2.0:
        distress_signals_fired += 1
    if late_osc > 1.5:
        distress_signals_fired += 1
    if early_lam > 1e-4 and (late_lam / max(early_lam, 1e-8) < 0.5
                              or late_lam / max(early_lam, 1e-8) > 2.5):
        distress_signals_fired += 1

    if distress_signals_fired >= 4:
        score += 20
    elif distress_signals_fired >= 3:
        score += 10
    dbg["signals_fired"] = distress_signals_fired

    score = min(score, 100.0)

    logger.info("Distress score: %.1f | signals: %s", score, dbg)
    print(f"  [DEBUG distress] score={score:.1f} | {dbg}")

    return score


def _compute_risk_score(l1_proba, l1_le, windowed_df, has_imu, has_wearable,
                        wearable_raw_df, fall_confirmation_info,
                        pose_df) -> tuple:
    """Compute a 0-100% fall risk score using max-of-components + modality multiplier.

    Returns (score_pct, level_label, n_modalities).
    """
    # --- Component 1: ML fall probability (peak across windows) ---
    ml_risk = 0.0
    if l1_proba is not None:
        classes = list(l1_le.classes_)
        fall_idx = classes.index("fall") if "fall" in classes else None
        if fall_idx is not None:
            ml_risk = float(np.max(l1_proba[:, fall_idx])) * 100

    # --- Component 2: Wearable vital risk (warning-severity driven) ---
    vital_risk = 0.0
    if has_wearable and wearable_raw_df is not None and not wearable_raw_df.empty:
        n_rows = len(wearable_raw_df)
        if "warning_level" in wearable_raw_df.columns:
            n_watch = int((wearable_raw_df["warning_level"] == "WATCH").sum())
            n_alert = int((wearable_raw_df["warning_level"] == "ALERT").sum())
            n_danger = int((wearable_raw_df["warning_level"] == "DANGER").sum())

            if n_danger > 0:
                vital_risk = 90 + min(n_danger * 3, 10)
            elif n_alert > 0:
                alert_pct = n_alert / max(n_rows, 1)
                vital_risk = 60 + alert_pct * 35
            elif n_watch > 0:
                watch_pct = n_watch / max(n_rows, 1)
                vital_risk = 30 + watch_pct * 25

        if "spo2_pct" in wearable_raw_df.columns:
            min_spo2 = float(wearable_raw_df["spo2_pct"].min())
            if min_spo2 < 90:
                vital_risk = max(vital_risk, 95)
            elif min_spo2 < 93:
                vital_risk = max(vital_risk, 75)

        if "heart_rate_bpm" in wearable_raw_df.columns:
            hr_vals = wearable_raw_df["heart_rate_bpm"]
            if float(hr_vals.min()) < 50 or float(hr_vals.max()) > 150:
                vital_risk = max(vital_risk, 80)

        vital_risk = min(vital_risk, 100)

    # --- Component 3: Kinematic risk from fall confirmation signals ---
    kinematic_risk = 0.0
    if fall_confirmation_info:
        if fall_confirmation_info.get("skipped"):
            kinematic_risk = ml_risk
        else:
            raw_score = fall_confirmation_info.get("score", 0)
            kinematic_risk = max(0, raw_score) / 7.0 * 100

    # --- Component 4: Distress posture risk (video-derived) ---
    distress_risk = _compute_distress_score(windowed_df, pose_df)

    # --- Base risk = max of all available components ---
    base_risk = max(ml_risk, vital_risk, kinematic_risk, distress_risk)
    print(f"  [DEBUG risk] ml={ml_risk:.1f} vital={vital_risk:.1f} "
          f"kinematic={kinematic_risk:.1f} distress={distress_risk:.1f} "
          f"-> base={base_risk:.1f}")

    # --- Modality confidence multiplier ---
    n_modalities = 1 + int(has_imu) + int(has_wearable)
    if has_imu and has_wearable:
        multiplier = 1.0
    elif has_wearable:
        multiplier = 0.95
    elif has_imu:
        multiplier = 0.90
    else:
        multiplier = 0.85

    risk = round(min(max(base_risk * multiplier, 0), 100), 1)

    if risk >= 90:
        level = "HIGH RISK"
    elif risk >= 70:
        level = "MODERATE RISK"
    elif risk >= 50:
        level = "LOW RISK"
    else:
        level = "STABLE"

    return risk, level, n_modalities


def predict_video(video_path: str, model_name: str, detailed: bool = False,
                  visualize: bool = False) -> dict:
    scaler_path = MODEL_DIR / "scaler.joblib"
    feat_col_path = MODEL_DIR / "feature_columns.joblib"
    l1_path = MODEL_DIR / f"{model_name}_l1.joblib"

    for p in (scaler_path, feat_col_path, l1_path):
        if not p.exists():
            logger.error("Required file not found: %s - run --train first", p)
            return {}

    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feat_col_path)
    l1_bundle = joblib.load(l1_path)
    l1_model = l1_bundle["model"]
    l1_le = l1_bundle["label_encoder"]

    # -- Feature extraction --
    vid_dir = Path(video_path).parent
    txt_files = list(vid_dir.glob("*.txt"))
    imu_df = parse_imu_file(str(txt_files[0])) if txt_files else pd.DataFrame()

    extractor = PoseFeatureExtractor()
    pose_df = extractor.process_video(video_path)
    extractor.close()

    if pose_df.empty:
        logger.error("Could not extract pose from video: %s", video_path)
        return {}

    fps = pose_df.attrs.get("fps", 20.0)
    n_frames = len(pose_df)
    imu_feat = extract_imu_features(imu_df, n_frames, fps)
    for col in imu_feat.columns:
        pose_df[col] = imu_feat[col].values

    # -- Data-quality signals (reporting only, not fed to the model) --
    valid_torso = pose_df["torso_inclination"].dropna()
    pose_coverage = float(len(valid_torso)) / n_frames if n_frames else 0.0
    sustained_inversion = bool(
        len(valid_torso) > 0 and float(valid_torso.min()) > 120.0)

    # -- Wearable raw CSV (feeds vital_risk directly, independent of windowing) --
    wearable_csvs = list(vid_dir.glob("*_wearable_sensors.csv"))
    wearable_raw_df = pd.DataFrame()
    has_wearable = False

    if wearable_csvs:
        try:
            wearable_raw_df = pd.read_csv(str(wearable_csvs[0]))
            has_wearable = not wearable_raw_df.empty
        except Exception:
            pass

    # -- Sliding window + derived features --
    ext_cfg = CONFIG.get("extraction", {})
    window_size = int(ext_cfg.get("window_size", 50))
    overlap = float(ext_cfg.get("overlap", 0.5))
    windowed = apply_sliding_window(pose_df, window_size, overlap)

    if windowed.empty:
        logger.error("Video too short for windowing (%d frames)", n_frames)
        return {}

    # -- Wearable window-level stats (v5) - must match merge_wearable_features()
    # exactly, since that's what the model was trained on. --
    from .features import window_wearable_stats
    if wearable_csvs:
        wearable_stats = window_wearable_stats(str(wearable_csvs[0]), len(windowed))
        if wearable_stats.abs().sum().sum() > 0:
            for col in wearable_stats.columns:
                windowed[col] = wearable_stats[col].values
        else:
            has_wearable = False

    windowed = _compute_derived_features(windowed)

    derived_has_wearable = bool(windowed.get(
        "has_wearable", pd.Series([0.0])).iloc[0])
    if derived_has_wearable != has_wearable:
        logger.error(
            "has_wearable mismatch: predict_video=%s vs derived feature=%s "
            "for %s", has_wearable, derived_has_wearable, video_path)

    missing_cols = [col for col in feature_cols if col not in windowed.columns]
    if missing_cols:
        logger.warning("Missing feature columns, filling with 0.0: %s",
                       missing_cols)
    for col in missing_cols:
        windowed[col] = 0.0
    X = np.nan_to_num(scaler.transform(windowed[feature_cols].values), nan=0.0)

    # -- L1 prediction (4-way internal) + confidence filtering + smoothing --
    l1_preds_raw = l1_model.predict(X)
    l1_labels_raw_4way = l1_le.inverse_transform(l1_preds_raw)

    l1_proba = None
    if hasattr(l1_model, "predict_proba"):
        l1_proba = l1_model.predict_proba(X)
        l1_labels_conf = confidence_filter(l1_proba, l1_labels_raw_4way,
                                           threshold=0.4)
    else:
        l1_labels_conf = l1_labels_raw_4way.copy()

    l1_labels_smooth_4way = temporal_smooth(l1_labels_conf, k=SMOOTH_K)

    # -- Collapse stationary/transitional -> "other" --
    l1_labels_raw = _collapse_l1(l1_labels_raw_4way)
    l1_labels_smooth = _collapse_l1(l1_labels_smooth_4way)

    # -- Fall confirmation - always compute for risk score --
    has_imu = not imu_df.empty
    l1_counts_pre = pd.Series(l1_labels_smooth).value_counts()
    top_l1_pre = l1_counts_pre.index[0]

    confirmed, fall_confirmation_info = _confirm_fall(
        windowed, has_imu, has_wearable, wearable_raw_df)

    if top_l1_pre == "fall" and not confirmed:
        logger.info("Fall NOT confirmed - downgrading to 'other'")
        l1_labels_smooth = np.array(["other"] * len(l1_labels_smooth))

    # -- L2 prediction --
    handler_cache: dict = {}
    rows = []
    l2_raw_labels = []
    l2_confidences = []

    for i, l1_label in enumerate(l1_labels_smooth):
        if l1_label not in handler_cache:
            cls = get_l2_handler(l1_label)
            handler_cache[l1_label] = cls(model_name, MODEL_DIR)

        handler = handler_cache[l1_label]
        l2_label, l2_conf = handler.predict_window(X[i])

        l2_raw_labels.append(l2_label)
        l2_confidences.append(l2_conf)
        rows.append({
            "window": i,
            "l1_raw": l1_labels_raw[i],
            "l1_4way": l1_labels_raw_4way[i],
            "l1": l1_label,
            "l2": l2_label,
        })

    # Temporal smoothing on L2
    l2_array = np.array(l2_raw_labels)
    l2_smoothed = temporal_smooth(l2_array, k=SMOOTH_K)
    for i in range(len(rows)):
        rows[i]["l2"] = l2_smoothed[i]

    result_df = pd.DataFrame(rows)
    result_df["l2_confidence"] = l2_confidences

    # -- Final majority votes --
    l1_counts = result_df["l1"].value_counts()
    top_l1 = l1_counts.index[0]

    l2_in_top_l1 = result_df[result_df["l1"] == top_l1]
    if not l2_in_top_l1.empty:
        l2_weight_sums: dict[str, float] = {}
        for _, row in l2_in_top_l1.iterrows():
            label = row["l2"]
            conf = row["l2_confidence"]
            l2_weight_sums[label] = l2_weight_sums.get(label, 0.0) + conf
        top_l2 = max(l2_weight_sums, key=l2_weight_sums.get)
    else:
        top_l2 = "n/a"

    # -- Finalize --
    handler = handler_cache.get(top_l1)
    if handler is not None:
        top_l2, disambiguation_info = handler.finalize(
            top_l2, pose_df, result_df)
    else:
        disambiguation_info = {"fired": False, "drift": 0.0,
                               "regularity": 0.0, "decision": "", "score": 0}

    # -- Output --
    modalities = []
    modalities.append("video")
    if has_imu:
        modalities.append("IMU")
    if has_wearable:
        modalities.append("wearable")
    modality_str = " + ".join(modalities)

    risk_score, risk_level, n_modalities = _compute_risk_score(
        l1_proba, l1_le, windowed, has_imu, has_wearable,
        wearable_raw_df, fall_confirmation_info, pose_df)

    print(f"  [DEBUG quality] pose_coverage={pose_coverage:.3f} "
          f"sustained_inversion={sustained_inversion}")

    print(f"\n[v5] Prediction for: {Path(video_path).name}")
    print(f"  Modalities: {modality_str}")
    print(f"  L1 (majority): {top_l1} "
          f"({l1_counts[top_l1]}/{len(result_df)} windows)")
    print(f"  L2 (majority): {top_l2}")
    print(f"  Fall risk:     {risk_score}% -- {risk_level}"
          f"  [{n_modalities} modalities: {modality_str}]")
    if fall_confirmation_info and not fall_confirmation_info.get("skipped"):
        print(f"  Fall confirmed: {fall_confirmation_info.get('score', 0):.1f} "
              f"(threshold=3.0)")
    print(f"  ----------------------------------------")
    print(f"  Risk legend:   >=90% HIGH | 70-89% MODERATE | 50-69% LOW | <50% STABLE")

    if detailed:
        print(f"\n  Smoothing: k={SMOOTH_K}, confidence_threshold=0.4")
        print(f"\nPer-window predictions (smoothed):")
        print(result_df[["window", "l1_4way", "l1", "l2"]].to_string(index=False))

        raw_l1_counts = Counter(l1_labels_raw_4way)
        smooth_l1_counts = Counter(l1_labels_smooth_4way)
        if raw_l1_counts != smooth_l1_counts:
            print(f"\n  L1 smoothing effect (4-way internal):")
            print(f"    Before: {dict(raw_l1_counts)}")
            print(f"    After:  {dict(smooth_l1_counts)}")

    if visualize:
        print(f"\n  Generating annotated video...")
        out_dir = generate_annotated_output(
            video_path, result_df, top_l1, top_l2,
            disambiguation_info, window_size, overlap,
        )
        if out_dir:
            print(f"  Output: {out_dir}")

    return {"l1": top_l1, "l2": top_l2, "windows": len(result_df),
            "modalities": modalities, "risk_score": risk_score,
            "risk_level": risk_level, "df": result_df,
            "pose_coverage": pose_coverage,
            "sustained_inversion": sustained_inversion}


def compare_predictions(video_path: str, model_name: str) -> None:
    """Run same video through v1, v2, v3, v4, and v5 side by side."""
    print("=" * 60)
    print(f"COMPARISON: {Path(video_path).name}")
    print(f"Model: {model_name}")
    print("=" * 60)

    # v1
    try:
        from activity_classifier import (
            predict_video as predict_v1,
            MODEL_DIR as V1_MODEL_DIR,
        )
        v1_l1_path = V1_MODEL_DIR / f"{model_name}_l1.joblib"
        if v1_l1_path.exists():
            print("\n--- v1 (baseline, no smoothing) ---")
            predict_v1(video_path, model_name, detailed=False)
        else:
            print("\n--- v1: models not found, skipping ---")
    except Exception:
        print("\n--- v1: not available ---")

    # v2
    try:
        from activity_classifier_v2 import (
            predict_video as predict_v2,
            MODEL_DIR as V2_MODEL_DIR,
        )
        v2_l1_path = V2_MODEL_DIR / f"{model_name}_l1.joblib"
        if v2_l1_path.exists():
            print("\n--- v2 (temporal smoothing + confidence filtering) ---")
            predict_v2(video_path, model_name, detailed=False)
        else:
            print("\n--- v2: models not found ---")
    except Exception:
        print("\n--- v2: not available ---")

    # v3
    try:
        from activity_classifier_v3.prediction import predict_video as predict_v3
        from activity_classifier_v3.config import MODEL_DIR as V3_MODEL_DIR
        v3_l1_path = V3_MODEL_DIR / f"{model_name}_l1.joblib"
        if v3_l1_path.exists():
            print("\n--- v3 (modular + disambiguation) ---")
            predict_v3(video_path, model_name, detailed=False)
        else:
            print("\n--- v3: models not found ---")
    except Exception:
        print("\n--- v3: not available ---")

    # v4
    try:
        from activity_classifier_v4.config import MODEL_DIR as V4_MODEL_DIR
        from activity_classifier_v4.prediction import predict_video as predict_v4
        v4_l1_path = V4_MODEL_DIR / f"{model_name}_l1.joblib"
        if v4_l1_path.exists():
            print("\n--- v4 (3-way L1 + fall confirmation) ---")
            predict_v4(video_path, model_name, detailed=False)
        else:
            print("\n--- v4: models not found ---")
    except Exception:
        print("\n--- v4: not available ---")

    # v5
    v5_l1_path = MODEL_DIR / f"{model_name}_l1.joblib"
    if v5_l1_path.exists():
        print("\n--- v5 (tri-modal: video + IMU + wearable) ---")
        predict_video(video_path, model_name, detailed=False)
    else:
        print("\n--- v5: models not found, run --train first ---")

    print("\n" + "=" * 60)