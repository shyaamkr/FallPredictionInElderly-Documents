"""Model evaluation and confusion matrix reporting."""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from .config import MODEL_DIR, FEATURES_DIR, PLOTS_DIR, logger
from .features import _compute_derived_features

def _resolve_model_bundle_path(model_name: str, suffix: str) -> Path:
    """Resolve model bundle path in MODEL_DIR (e.g., suffix='l1').

    FIX: training.py saves as {model}_11.joblib, evaluation used to
    look for {model}_l1.joblib only. Try both.
    """
    # Try l1 first, then 11 (what training.py actually saves)
    for cand_suffix in (suffix, "11" if suffix == "l1" else suffix, "l2" if suffix == "l2" else suffix):
        p = Path(MODEL_DIR) / f"{model_name}_{cand_suffix}.joblib"
        if p.exists():
            return p
    # Default to l1 name if none exists (so warning shows expected path)
    return Path(MODEL_DIR) / f"{model_name}_{suffix}.joblib"

def evaluate_model(model_name: str, level: str = "l1") -> dict:
    feat_path = FEATURES_DIR / "windowed_features_v5.parquet"
    if not feat_path.exists():
        feat_path = FEATURES_DIR / "windowed_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found - run --extract first")
        return {}

    df = pd.read_parquet(feat_path)
    df = _compute_derived_features(df)

    scaler_path = Path(MODEL_DIR) / "scaler.joblib"
    feat_cols_path = Path(MODEL_DIR) / "feature_columns.joblib"
    if not scaler_path.exists() or not feat_cols_path.exists():
        logger.error("Scaler or feature_columns not found in %s - run --train first", MODEL_DIR)
        return {}

    scaler = joblib.load(scaler_path)
    saved_cols = joblib.load(feat_cols_path)
    for col in saved_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = np.nan_to_num(scaler.transform(df[saved_cols].values), nan=0.0)

    results = {}

    if level in ("l1", "both"):
        l1_path = _resolve_model_bundle_path(model_name, "l1")
        if not l1_path.exists():
            logger.warning("Skipping %s - missing %s", model_name, l1_path)
            return results

        l1_bundle = joblib.load(l1_path)
        l1_model = l1_bundle["model"]
        l1_le = l1_bundle["label_encoder"]
        y_true = l1_le.transform(df["l1_category"].values)
        y_pred = l1_model.predict(X)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results["l1"] = {"accuracy": acc, "f1_macro": f1}

        print(f"\n[v5] {model_name} L1 Evaluation (4-way internal):")
        # FIX: pass labels= explicitly, otherwise crashes when a class
        # is missing in y_true/y_pred (2 classes vs 4 target_names)
        all_labels = list(range(len(l1_le.classes_)))
        print(classification_report(
            y_true, y_pred,
            labels=all_labels,
            target_names=list(l1_le.classes_),
            zero_division=0))

        # FIX: same for confusion_matrix - otherwise shape is 2x2 not 4x4
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        _save_confusion_matrix(
            cm, l1_le.classes_.tolist(),
            f"{model_name} L1 (v5 - tri-modal)",
            PLOTS_DIR / f"{model_name}_l1_confusion.png",
        )

    return results

def _save_confusion_matrix(cm, classes, title, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        logger.info("Saved confusion matrix: %s", save_path)
    except Exception as e:
        logger.warning("Could not save confusion matrix: %s", e)

def run_evaluation(model_name: str, level: str = "l1") -> None:
    if model_name == "all":
        all_results = {}
        for name in ("xgboost", "random_forest", "lightgbm", "svm",
                     "ensemble"):
            model_path = _resolve_model_bundle_path(name, "l1")
            if not model_path.exists():
                logger.warning("Skipping %s - not trained", name)
                continue
            logger.info("Evaluating %s ...", name)
            res = evaluate_model(name, level)
            if res:
                all_results[name] = res.get("l1", {})

        if all_results:
            print("\n=== v5 Model Comparison (4-way internal L1) ===")
            print(f"{'Model':<20} {'Accuracy':>10} {'F1-macro':>10}")
            print("-" * 42)
            for name, metrics in all_results.items():
                print(f"{name:<20} {metrics.get('accuracy', 0):.4f}"
                      f"     {metrics.get('f1_macro', 0):.4f}")
    else:
        evaluate_model(model_name, level)