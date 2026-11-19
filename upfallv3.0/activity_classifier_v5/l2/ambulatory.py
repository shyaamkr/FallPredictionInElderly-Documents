"""L2 handler for the ambulatory category.

Actions: walking, slow_walking, jogging, stairs_up, stairs_down.
Includes jogging-vs-stairs disambiguation logic.
"""

from pathlib import Path

import numpy as np

from ..config import logger


# -- Per-action handlers ---------------------------------------------------

def _handle_walking(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_slow_walking(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_jogging(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_stairs_up(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_stairs_down(X_row, proba, default_confidence: float) -> float:
    return default_confidence


# -- Disambiguation helpers ------------------------------------------------

def _check_vertical_drift(pose_df) -> float:
    if 'nose_y' not in pose_df.columns or len(pose_df) < 20:
        return 0.0

    nose_y = pose_df['nose_y'].dropna().values
    if len(nose_y) < 20:
        return 0.0

    q_len = len(nose_y) // 4
    q1_mean = nose_y[:q_len].mean()
    q4_mean = nose_y[-q_len:].mean()

    oscillation = nose_y.std()
    if oscillation < 1e-6:
        return 0.0

    drift = (q4_mean - q1_mean) / oscillation
    return float(drift)


def _check_oscillation_regularity(pose_df) -> float:
    if 'nose_y' not in pose_df.columns or len(pose_df) < 15:
        return 0.0

    nose_y = pose_df['nose_y'].dropna().values
    if len(nose_y) < 15:
        return 0.0

    x = np.arange(len(nose_y))
    slope = np.polyfit(x, nose_y, 1)[0]
    detrended = nose_y - (slope * x + nose_y[0])

    n = len(detrended)
    detrended = detrended - detrended.mean()
    variance = np.var(detrended)
    if variance < 1e-10:
        return 0.0

    max_autocorr = 0.0
    for lag in range(3, min(40, n // 3)):
        seg1 = detrended[:n - lag]
        seg2 = detrended[lag:]
        if len(seg1) < 2:
            continue
        autocorr = np.correlate(seg1, seg2)[0] / (variance * (n - lag))
        max_autocorr = max(max_autocorr, autocorr)

    return float(max_autocorr)


def _compute_jogging_score(pose_df, result_df=None) -> tuple[float, dict]:
    details = {}
    score = 0.0
    nose_y = (pose_df['nose_y'].dropna().values
              if 'nose_y' in pose_df.columns else np.array([]))
    n_windows = len(result_df) if result_df is not None else 0
    is_short = len(nose_y) < 50 or n_windows <= 5

    regularity = _check_oscillation_regularity(pose_df)
    details['regularity'] = regularity
    if regularity > 0.65:
        score += 1.0

    leg_stds = []
    for col in ('left_leg_angle', 'right_leg_angle'):
        if col in pose_df.columns:
            vals = pose_df[col].dropna().values
            if len(vals) >= 10:
                details[f'{col}_std'] = round(float(vals.std()), 3)
                leg_stds.append(float(vals.std()))
    avg_leg_std = np.mean(leg_stds) if leg_stds else 0.0
    details['avg_leg_std'] = round(avg_leg_std, 3)
    if avg_leg_std > 20.0:
        score += 2.0
    elif avg_leg_std > 2.5:
        score += 1.0

    if 'ankle_distance' in pose_df.columns:
        ad = pose_df['ankle_distance'].dropna().values
        ad_std = float(ad.std()) if len(ad) >= 10 else 0.0
        details['ankle_distance_std'] = round(ad_std, 4)
        if ad_std > 0.05:
            score += 1.0
    else:
        details['ankle_distance_std'] = 0.0

    if 'cog_velocity_y' in pose_df.columns:
        vy = pose_df['cog_velocity_y'].dropna().values
        vy_std = float(vy.std()) if len(vy) >= 10 else 0.0
        details['cog_vy_std'] = round(vy_std, 4)
        if vy_std > 0.22:
            score += 2.0
    else:
        details['cog_vy_std'] = 0.0

    jogging_frac = 0.0
    avg_stairs_conf = 1.0
    if result_df is not None and len(result_df) > 0:
        l2_vals = result_df['l2'].values
        jogging_count = int(np.sum(np.isin(l2_vals, ['jogging'])))
        jogging_frac = jogging_count / len(result_df)
        stairs_mask = np.isin(l2_vals, ['stairs_down', 'stairs_up'])
        if stairs_mask.any() and 'l2_confidence' in result_df.columns:
            avg_stairs_conf = float(
                result_df.loc[stairs_mask, 'l2_confidence'].mean())
        details['jogging_window_frac'] = round(jogging_frac, 3)
        details['avg_stairs_confidence'] = round(avg_stairs_conf, 3)
        if jogging_frac >= 0.25 or avg_stairs_conf < 0.55:
            score += 2.0
    else:
        details['jogging_window_frac'] = 0.0
        details['avg_stairs_confidence'] = 1.0

    drift = _check_vertical_drift(pose_df)
    details['drift'] = round(drift, 3)
    nose_y_range = float(nose_y.max() - nose_y.min()) if len(nose_y) > 5 else 0.0
    details['nose_y_range'] = round(nose_y_range, 4)

    has_reliable_drift = len(nose_y) >= 50
    if has_reliable_drift and abs(drift) > 1.5 and regularity < 0.65:
        score -= 1.0
    if has_reliable_drift and abs(drift) > 2.0 and nose_y_range > 0.3:
        score -= 1.0

    vy_std = details.get('cog_vy_std', 0.0)
    if has_reliable_drift and vy_std < 0.10:
        score -= 2.0

    details['score'] = round(score, 1)
    details['is_short'] = is_short
    return score, details


def _compute_slow_walking_score(pose_df, result_df=None) -> tuple[float, dict]:
    details = {}
    score = 0.0
    nose_y = (pose_df['nose_y'].dropna().values
              if 'nose_y' in pose_df.columns else np.array([]))
    n_windows = len(result_df) if result_df is not None else 0
    is_short = len(nose_y) < 50 or n_windows <= 5

    drift = _check_vertical_drift(pose_df)
    details['drift'] = round(drift, 3)
    if abs(drift) < 0.5:
        score += 1.0

    regularity = _check_oscillation_regularity(pose_df)
    details['regularity'] = regularity
    if regularity > 0.6:
        score += 1.0

    step_sym = 0.0
    if 'step_symmetry' in pose_df.columns:
        ss = pose_df['step_symmetry'].dropna().values
        step_sym = float(ss.mean()) if len(ss) >= 10 else 0.0
        details['step_symmetry_mean'] = round(step_sym, 3)
        if step_sym > 0.75:
            score += 2.0

    cog_vy_std = 0.0
    if 'cog_velocity_y' in pose_df.columns:
        vy = pose_df['cog_velocity_y'].dropna().values
        cog_vy_std = float(vy.std()) if len(vy) >= 10 else 0.0
        details['cog_vy_std'] = round(cog_vy_std, 4)
        if cog_vy_std < 0.20:
            score += 1.0

    slow_walk_frac = 0.0
    avg_stairs_conf = 1.0
    if result_df is not None and len(result_df) > 0:
        l2_vals = result_df['l2'].values
        sw_count = int(np.sum(np.isin(l2_vals, ['slow_walking', 'walking'])))
        slow_walk_frac = sw_count / len(result_df)
        stairs_mask = np.isin(l2_vals, ['stairs_down', 'stairs_up'])
        if stairs_mask.any() and 'l2_confidence' in result_df.columns:
            avg_stairs_conf = float(
                result_df.loc[stairs_mask, 'l2_confidence'].mean())
        details['slow_walk_window_frac'] = round(slow_walk_frac, 3)
        details['avg_stairs_confidence'] = round(avg_stairs_conf, 3)
        if avg_stairs_conf < 0.75 or slow_walk_frac > 0:
            score += 2.0
    else:
        details['slow_walk_window_frac'] = 0.0
        details['avg_stairs_confidence'] = 1.0

    has_reliable = len(nose_y) >= 50

    torso_std = 0.0
    if 'torso_inclination' in pose_df.columns:
        ti = pose_df['torso_inclination'].dropna().values
        torso_std = float(ti.std()) if len(ti) >= 10 else 0.0
        details['torso_incl_std'] = round(torso_std, 3)
        if has_reliable and torso_std > 15.0:
            score -= 1.0

    disp_mean = 0.0
    if 'cog_displacement_rate' in pose_df.columns:
        dr = pose_df['cog_displacement_rate'].dropna().values
        disp_mean = float(dr.mean()) if len(dr) >= 10 else 0.0
        details['disp_rate_mean'] = round(disp_mean, 4)
        if has_reliable and disp_mean < 0.02:
            score -= 1.0

    details['score'] = round(score, 1)
    details['is_short'] = is_short
    return score, details


def _disambiguate_ambulatory(video_path, top_l2, pose_df,
                             result_df=None) -> tuple[str, dict]:
    if top_l2 not in ('stairs_down', 'stairs_up'):
        return top_l2, {"fired": False, "drift": 0.0, "regularity": 0.0,
                        "decision": "", "score": 0}

    original_l2 = top_l2

    jog_score, jog_details = _compute_jogging_score(pose_df, result_df)
    jog_threshold = 2.5 if jog_details.get('is_short') else 3.0

    logger.info(
        'Jogging check: score=%.1f (threshold=%.1f) | regularity=%.3f, '
        'avg_leg_std=%.3f, ankle_std=%.4f, cog_vy_std=%.4f, '
        'jogging_frac=%.3f, drift=%.3f',
        jog_score, jog_threshold,
        jog_details.get('regularity', 0),
        jog_details.get('avg_leg_std', 0),
        jog_details.get('ankle_distance_std', 0),
        jog_details.get('cog_vy_std', 0),
        jog_details.get('jogging_window_frac', 0),
        jog_details.get('drift', 0),
    )

    if jog_score >= jog_threshold:
        logger.info('Jogging score %.1f >= %.1f: overriding %s to jogging',
                    jog_score, jog_threshold, top_l2)
        return 'jogging', {
            "fired": True,
            "drift": jog_details.get('drift', 0.0),
            "regularity": jog_details.get('regularity', 0.0),
            "score": jog_details.get('score', 0),
            "decision": f"{original_l2} -> jogging",
        }

    sw_score, sw_details = _compute_slow_walking_score(pose_df, result_df)
    sw_threshold = 2.5 if sw_details.get('is_short') else 3.0

    logger.info(
        'Slow walking check: score=%.1f (threshold=%.1f) | drift=%.3f, '
        'regularity=%.3f, step_sym=%.3f, cog_vy_std=%.4f, '
        'stairs_conf=%.3f, torso_std=%.3f',
        sw_score, sw_threshold,
        sw_details.get('drift', 0),
        sw_details.get('regularity', 0),
        sw_details.get('step_symmetry_mean', 0),
        sw_details.get('cog_vy_std', 0),
        sw_details.get('avg_stairs_confidence', 0),
        sw_details.get('torso_incl_std', 0),
    )

    if sw_score >= sw_threshold:
        logger.info('Slow walking score %.1f >= %.1f: overriding %s to slow_walking',
                    sw_score, sw_threshold, top_l2)
        return 'slow_walking', {
            "fired": True,
            "drift": sw_details.get('drift', 0.0),
            "regularity": sw_details.get('regularity', 0.0),
            "score": sw_details.get('score', 0),
            "decision": f"{original_l2} -> slow_walking",
        }

    drift = jog_details.get('drift', 0.0)
    regularity = jog_details.get('regularity', 0.0)
    return top_l2, {
        "fired": False,
        "drift": drift,
        "regularity": regularity,
        "score": jog_details.get("score", 0),
        "decision": f"keep {original_l2}",
    }


# -- Handler class ---------------------------------------------------------

class AmbulatoryHandler:
    """L2 handler for ambulatory activities."""

    _category = "ambulatory"

    def __init__(self, model_name: str, model_dir):
        from . import BaseL2Handler
        self._base = BaseL2Handler(model_name, model_dir)
        self._base._category = self._category
        self._base._actions = {
            "walking": _handle_walking,
            "slow_walking": _handle_slow_walking,
            "jogging": _handle_jogging,
            "stairs_up": _handle_stairs_up,
            "stairs_down": _handle_stairs_down,
        }
        self.model_name = model_name
        self.model_dir = model_dir

    def predict_window(self, X_row: np.ndarray) -> tuple[str, float]:
        return self._base.predict_window(X_row)

    def finalize(self, top_l2: str, pose_df, result_df) -> tuple[str, dict]:
        return _disambiguate_ambulatory(
            None, top_l2, pose_df, result_df)