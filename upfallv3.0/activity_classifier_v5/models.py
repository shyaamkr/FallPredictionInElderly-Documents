"""Model factory - tuned v2 hyperparameters."""

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb


def create_model(model_name: str):
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", eval_metric="mlogloss",
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, random_state=42,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=25, class_weight="balanced",
            min_samples_leaf=3, random_state=42,
        )
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            objective="multiclass", subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5, random_state=42, verbose=-1,
        )
    if model_name == "svm":
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
            ensemble=False,
        )
    if model_name == "ensemble":
        return _create_ensemble()
    raise ValueError(f"Unknown model: {model_name}")


def _create_ensemble() -> StackingClassifier:
    estimators = [
        ("xgboost", create_model("xgboost")),
        ("random_forest", create_model("random_forest")),
        ("lightgbm", create_model("lightgbm")),
        ("svm", create_model("svm")),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1,
    )