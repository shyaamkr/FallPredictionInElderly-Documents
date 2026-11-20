"""Model factory - supports all models used by cli.py."""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def create_model(model_name, num_class=None):
    name = model_name.lower().strip().replace("-", "_").replace(" ", "_")

    # --- XGBoost ---
    if name in ("xgboost", "xgb", "xgbclassifier"):
        from xgboost import XGBClassifier
        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        if num_class is not None:
            num_class = int(num_class)
            if num_class < 2:
                raise ValueError(f"num_class must be >=2, got {num_class}")
            if num_class > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = num_class
            else:
                params["objective"] = "binary:logistic"
        return XGBClassifier(**params)

    # --- LightGBM ---
    if name in ("lightgbm", "lgbm", "lgb"):
        from lightgbm import LGBMClassifier
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
        )
        if num_class is not None and int(num_class) > 2:
            params["objective"] = "multiclass"
            params["num_class"] = int(num_class)
        return LGBMClassifier(**params)

    # --- Random Forest ---
    if name in ("random_forest", "rf", "randomforest"):
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
        )

    # --- SVM ---
    if name in ("svm", "svc", "rbf_svm", "linear_svm"):
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=42,
        )

    # --- KNN ---
    if name in ("knn", "kneighbors", "k_neighbors"):
        return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)

    # --- Logistic ---
    if name in ("logistic", "logreg", "logistic_regression", "lr"):
        return LogisticRegression(
            max_iter=1000, multi_class="multinomial", class_weight="balanced", n_jobs=-1
        )

    # --- MLP ---
    if name in ("mlp", "mlpclassifier", "neural"):
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

    # --- ENSEMBLE - THIS WAS MISSING ---
    if name in ("ensemble", "voting", "voting_classifier", "combined"):
        # Soft voting: needs predict_proba from all members
        # We build members by calling create_model recursively
        # so num_class is passed correctly to xgboost/lightgbm
        xgb = create_model("xgboost", num_class=num_class)
        lgbm = create_model("lightgbm", num_class=num_class)
        rf = create_model("random_forest", num_class=num_class)
        svm = create_model("svm", num_class=num_class)

        return VotingClassifier(
            estimators=[
                ("xgb", xgb),
                ("lgbm", lgbm),
                ("rf", rf),
                ("svm", svm),
            ],
            voting="soft",  # average probabilities
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported: xgboost, lightgbm, random_forest, svm, knn, logistic, mlp, ensemble"
    )