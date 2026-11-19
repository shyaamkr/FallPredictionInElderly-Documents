"""Model training - 4-way internal L1, per-category L2, 4-tier modality dropout."""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from .config import MODEL_DIR, FEATURES_DIR, logger
from .models import create_model
from .features import (
    _get_feature_cols, _compute_sample_weight, _compute_derived_features,
    WEARABLE_FEATURE_NAMES,
)


def _remap_l1(action_ids: np.ndarray) -> np.ndarray:
    """Map action_id to 3-way L1: ambulatory / fall / other."""
    from .config import L1_MAPPING
    act_to_l1 = {}
    for l1_cat, act_list in L1_MAPPING.items():
        for act_id in act_list:
            act_to_l1[act_id] = l1_cat
    return np.array([act_to_l1.get(int(a), "other") for a in action_ids])


def _identify_modality_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Identify IMU and wearable columns within the feature set."""
    imu_prefixes = ("imu_", "peak_impact_g", "jerk_magnitude",
                    "orientation_change_rate", "vertical_acc_asymmetry")
    imu_derived = ("impact_regularity", "gait_energy_ratio", "has_imu")

    wearable_prefixes = tuple(f"{wf}_" for wf in WEARABLE_FEATURE_NAMES)
    wearable_derived = ("has_wearable", "hr_spo2_coupling",
                        "vital_instability", "gsr_hr_product",
                        "vital_anomaly_score")

    imu_cols = [c for c in feature_cols
                if any(c.startswith(p) for p in imu_prefixes)
                or c in imu_derived]

    wearable_cols = [c for c in feature_cols
                     if any(c.startswith(p) for p in wearable_prefixes)
                     or c in wearable_derived]

    return imu_cols, wearable_cols


def train_model(model_name: str, level: str = "both") -> None:
    feat_path = FEATURES_DIR / "windowed_features_v5.parquet"
    if not feat_path.exists():
        feat_path = FEATURES_DIR / "windowed_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found at %s - run --extract first",
                     feat_path)
        return

    df = pd.read_parquet(feat_path)
    df = _compute_derived_features(df)
    feature_cols = _get_feature_cols(df)

    logger.info("L1 (original 4-way) distribution: %s",
                df["l1_category"].value_counts().to_dict())
    logger.info("Feature columns: %d total", len(feature_cols))

    imu_cols, wearable_cols = _identify_modality_columns(feature_cols)
    logger.info("IMU columns: %d, Wearable columns: %d",
                len(imu_cols), len(wearable_cols))

    # 4-tier modality dropout augmentation
    rng = np.random.default_rng(42)
    n_total = len(df)

    # Tier 2: No wearable (15%)
    n_no_wearable = int(n_total * 0.15)
    idx_no_wearable = rng.choice(n_total, size=n_no_wearable, replace=False)
    df_no_wearable = df.iloc[idx_no_wearable].copy()
    for col in wearable_cols:
        if col in df_no_wearable.columns:
            df_no_wearable[col] = 0.0

    # Tier 3: No IMU (15%)
    remaining = np.setdiff1d(np.arange(n_total), idx_no_wearable)
    n_no_imu = int(n_total * 0.15)
    idx_no_imu = rng.choice(remaining, size=min(n_no_imu, len(remaining)),
                            replace=False)
    df_no_imu = df.iloc[idx_no_imu].copy()
    for col in imu_cols:
        if col in df_no_imu.columns:
            df_no_imu[col] = 0.0

    # Tier 4: Video only - no IMU and no wearable (20%)
    used = np.union1d(idx_no_wearable, idx_no_imu)
    remaining2 = np.setdiff1d(np.arange(n_total), used)
    n_video_only = int(n_total * 0.20)
    idx_video_only = rng.choice(remaining2,
                                size=min(n_video_only, len(remaining2)),
                                replace=False)
    df_video_only = df.iloc[idx_video_only].copy()
    for col in imu_cols + wearable_cols:
        if col in df_video_only.columns:
            df_video_only[col] = 0.0

    df = pd.concat([df, df_no_wearable, df_no_imu, df_video_only],
                   ignore_index=True)
    logger.info(
        "4-tier augmentation: +%d no-wearable, +%d no-imu, "
        "+%d video-only (total: %d)",
        n_no_wearable, len(idx_no_imu), len(idx_video_only), len(df),
    )

    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X = np.nan_to_num(X, nan=0.0)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")
    logger.info("Saved scaler and feature columns (%d cols)", len(feature_cols))

    groups = df["subject_id"].values
    use_weight = model_name in ("xgboost", "lightgbm")

    # -- L1 training (4-way internal) --
    if level in ("l1", "both"):
        le_11 = LabelEncoder()
        y_11 = le_11.fit_transform(df["l1_category"].values)
        logger.info("L1 classes: %s", list(le_11.classes_))

        n_unique_groups = len(np.unique(groups))
        n_splits = min(5, n_unique_groups)
        if n_splits < 2:
            logger.warning("Only %d subject group(s) - using simple split",
                           n_unique_groups)
            n_splits = 2
            groups_for_cv = np.arange(len(X))
        else:
            groups_for_cv = groups
        cv = StratifiedGroupKFold(n_splits=n_splits)

        model = create_model(model_name)
        acc_scores, f1_scores = [], []
        for train_idx, val_idx in cv.split(X, y_11, groups=groups_for_cv):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_11[train_idx], y_11[val_idx]
            fit_kwargs = {}
            if use_weight:
                fit_kwargs["sample_weight"] = _compute_sample_weight(y_tr)
            model.fit(X_tr, y_tr, **fit_kwargs)
            preds = model.predict(X_val)
            acc_scores.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds, average="macro",
                                      zero_division=0))

        logger.info(
            "L1 CV (%s): acc=%.3f+/-%.3f  f1_macro=%.3f+/-%.3f",
            model_name, np.mean(acc_scores), np.std(acc_scores),
            np.mean(f1_scores), np.std(f1_scores),
        )

        final_model = create_model(model_name)
        fit_kwargs = {}
        if use_weight:
            fit_kwargs["sample_weight"] = _compute_sample_weight(y_11)
        final_model.fit(X, y_11, **fit_kwargs)
        joblib.dump({"model": final_model, "label_encoder": le_11},
                    MODEL_DIR / f"{model_name}_11.joblib")
        logger.info("Saved %s_11.joblib", model_name)

    # -- L2 training (ambulatory and fall only) --
    if level in ("l2", "both"):
        for l1_cat in ("ambulatory", "fall", "transitional", "stationary"):
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
            for train_idx, val_idx in cv.split(X_sub, y_l2,
                                               groups=grp_sub_cv):
                X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
                y_tr, y_val = y_l2[train_idx], y_l2[val_idx]
                fit_kwargs = {}
                if use_weight:
                    fit_kwargs["sample_weight"] = _compute_sample_weight(y_tr)
                model.fit(X_tr, y_tr, **fit_kwargs)
                preds = model.predict(X_val)
                acc_scores.append(accuracy_score(y_val, preds))
                f1_scores.append(f1_score(y_val, preds, average="macro",
                                          zero_division=0))

            logger.info(
                "L2[%s] CV (%s): acc=%.3f+/-%.3f  f1_macro=%.3f+/-%.3f",
                l1_cat, model_name, np.mean(acc_scores), np.std(acc_scores),
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