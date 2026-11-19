"""Model evaluation and confusion matrix reporting."""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from .config import MODEL_DIR, FEATURES_DIR, PLOTS_DIR, logger
from .features import _compute_derived_features


def evaluate_model(model_name: str, level: str = "l1") -> dict:
    feat_path = FEATURES_DIR / "windowed_features_v5.parquet"
    if not feat_path.exists():
        feat_path = FEATURES_DIR / "windowed_features.parquet"
    if not feat_path.exists():
        logger.error("Features not found - run --extract first")
        return {}

    df = pd.read_parquet(feat_path)
    df = _compute_derived_features(df)
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    saved_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    for col in saved_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = np.nan_to_num(scaler.transform(df[saved_cols].values), nan=0.0)

    results = {}

    if level in ("l1", "both"):
        l1_bundle = joblib.load(MODEL_DIR / f"{model_name}_l1.joblib")
        l1_model = l1_bundle["model"]
        l1_le = l1_bundle["label_encoder"]
        y_true = l1_le.transform(df["l1_category"].values)
        y_pred = l1_model.predict(X)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results["l1"] = {"accuracy": acc, "f1_macro": f1}

        print(f"\n[v5] {model_name} L1 Evaluation (4-way internal):")
        print(classification_report(y_true, y_pred,
                                    target_names=l1_le.classes_,
                                    zero_division=0))

        cm = confusion_matrix(y_true, y_pred)
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
            model_path = MODEL_DIR / f"{name}_l1.joblib"
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