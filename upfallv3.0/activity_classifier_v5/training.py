"""Model training - 4-way internal L1, per-category L2, 4-tier modality dropout."""

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from .config import MODEL_DIR, FEATURES_DIR, logger
from .models import create_model
from .features import (
    _get_feature_cols, _compute_sample_weight, _compute_derived_features,
    WEARABLE_FEATURE_NAMES,
)

def _make_model(model_name: str, num_class=None):
    """Create model with correct num_class. Works with old/new models.py."""
    if num_class is not None:
        num_class = int(num_class)
        if num_class < 2:
            raise ValueError(f"Cannot train with num_class={num_class}. Only 1 class in y!")
    try:
        # New models.py accepts num_class
        return create_model(model_name, num_class=num_class)
    except TypeError:
        # Old models.py - fallback, then patch params
        m = create_model(model_name)
        if num_class is not None and hasattr(m, "set_params"):
            try:
                if num_class > 2:
                    m.set_params(objective="multi:softprob", num_class=num_class)
                elif num_class == 2:
                    m.set_params(objective="binary:logistic")
            except Exception:
                pass
        return m

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

def _has_all_classes(y_sub: np.ndarray, y_all: np.ndarray) -> bool:
    """Return True if y_sub contains every class present in y_all."""
    return set(np.unique(y_all)).issubset(set(np.unique(y_sub)))

def _compute_adaptive_splits(
    y: np.ndarray,
    groups: np.ndarray,
    max_splits: int = 5,
) -> tuple[int, np.ndarray, int]:
    """Compute safe CV split count from minimum class count and group count."""
    class_counts = np.bincount(y)
    positive_counts = class_counts[class_counts > 0]
    min_class_count = int(np.min(positive_counts)) if len(positive_counts) > 0 else 0

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 2:
        groups_for_cv = np.arange(len(y))
        effective_groups = len(groups_for_cv)
    else:
        groups_for_cv = groups
        effective_groups = n_unique_groups

    if min_class_count < 2 or effective_groups < 2:
        return 0, groups_for_cv, min_class_count

    n_splits = min(max_splits, effective_groups, min_class_count)
    return max(2, n_splits), groups_for_cv, min_class_count

def _run_cv_folds(
    X_data: np.ndarray,
    y_data: np.ndarray,
    groups_data: np.ndarray,
    model_name: str,
    use_weight: bool,
    max_splits: int = 5,
    context_label: str = "L1",
    num_class=None,
) -> tuple[list[float], list[float]]:
    """Run StratifiedGroupKFold with adaptive n_splits and clean single-sample handling."""
    n_splits, groups_for_cv, min_class_count = _compute_adaptive_splits(
        y_data, groups_data, max_splits=max_splits
    )
    if n_splits < 2:
        logger.warning(
            "Skipping CV for %s - least populated class has %d sample(s) "
            "(groups=%d); fitting full subset directly",
            context_label, min_class_count, len(np.unique(groups_for_cv)),
        )
        return [], []

    if len(np.unique(groups_data)) < 2:
        logger.warning(
            "Only %d subject group(s) in %s - falling back to index groups with n_splits=%d",
            len(np.unique(groups_data)), context_label, n_splits,
        )

    cv = StratifiedGroupKFold(n_splits=n_splits)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        splits = list(cv.split(X_data, y_data, groups=groups_for_cv))

    acc_scores, f1_scores = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        if not _has_all_classes(y_tr, y_data):
            logger.warning(
                "Skipping %s fold %d: missing class(es) in y_tr (%s vs %s)",
                context_label, fold_idx, np.unique(y_tr), np.unique(y_data),
            )
            continue

        model = _make_model(model_name, num_class=num_class)
        fit_kwargs = {}
        if use_weight:
            fit_kwargs["sample_weight"] = _compute_sample_weight(y_tr)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            model.fit(X_tr, y_tr, **fit_kwargs)
            preds = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds, average="macro",
                                  zero_division=0))

    return acc_scores, f1_scores

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
        n_classes_l1 = len(le_11.classes_)
        if n_classes_l1 < 2:
            raise ValueError(f"L1 has only {n_classes_l1} class: {list(le_11.classes_)} - check l1_category in parquet")

        acc_scores, f1_scores = _run_cv_folds(
            X, y_11, groups,
            model_name=model_name,
            use_weight=use_weight,
            max_splits=5,
            context_label="L1",
            num_class=n_classes_l1,
        )
        if acc_scores:
            logger.info(
                "L1 CV (%s, folds=%d): acc=%.3f+/-%.3f  f1_macro=%.3f+/-%.3f",
                model_name, len(acc_scores), np.mean(acc_scores),
                np.std(acc_scores), np.mean(f1_scores), np.std(f1_scores),
            )
        else:
            logger.warning("No valid L1 CV folds completed")

        final_model = _make_model(model_name, num_class=n_classes_l1)
        fit_kwargs = {}
        if use_weight:
            fit_kwargs["sample_weight"] = _compute_sample_weight(y_11)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            final_model.fit(X, y_11, **fit_kwargs)
        joblib.dump({"model": final_model, "label_encoder": le_11},
                    MODEL_DIR / f"{model_name}_11.joblib")
        logger.info("Saved %s_11.joblib", model_name)

    # -- L2 training (per L1 category) --
    if level in ("l2", "both"):
        for l1_cat in ("ambulatory", "fall", "transitional", "stationary"):
            mask = (df["l1_category"] == l1_cat).values
            sub_df = df[mask]
            if sub_df["l2_subtype"].nunique() < 2:
                logger.warning("Skipping L2 for '%s' - only 1 subtype", l1_cat)
                continue

            X_sub = X[mask]
            grp_sub = groups[mask]
            le_l2 = LabelEncoder()
            y_l2 = le_l2.fit_transform(sub_df["l2_subtype"].values)
            n_classes_l2 = len(le_l2.classes_)

            acc_scores, f1_scores = _run_cv_folds(
                X_sub, y_l2, grp_sub,
                model_name=model_name,
                use_weight=use_weight,
                max_splits=5,
                context_label=f"L2[{l1_cat}]",
                num_class=n_classes_l2,
            )
            if acc_scores:
                logger.info(
                    "L2[%s] CV (%s, folds=%d): acc=%.3f+/-%.3f  f1_macro=%.3f+/-%.3f",
                    l1_cat, model_name, len(acc_scores), np.mean(acc_scores),
                    np.std(acc_scores), np.mean(f1_scores), np.std(f1_scores),
                )
            else:
                logger.warning(
                    "No valid L2[%s] CV folds completed; saving model fitted on full '%s' subset",
                    l1_cat, l1_cat,
                )

            final_model = _make_model(model_name, num_class=n_classes_l2)
            fit_kwargs = {}
            if use_weight:
                fit_kwargs["sample_weight"] = _compute_sample_weight(y_l2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                final_model.fit(X_sub, y_l2, **fit_kwargs)
            joblib.dump({"model": final_model, "label_encoder": le_l2},
                        MODEL_DIR / f"{model_name}_l2_{l1_cat}.joblib")
            logger.info("Saved %s_l2_%s.joblib", model_name, l1_cat)