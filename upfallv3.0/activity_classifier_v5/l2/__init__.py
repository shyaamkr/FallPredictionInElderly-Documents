"""L2 handler registry - ambulatory and fall only; 'other' is a no-op."""

from pathlib import Path

import numpy as np
import joblib

from ..config import logger


class BaseL2Handler:
    """Base class for per-category L2 prediction."""

    _category: str = ""
    _actions: dict = {}

    def __init__(self, model_name: str, model_dir: Path):
        self.model_name = model_name
        self.model_dir = model_dir
        self._bundle = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        path = self.model_dir / f"{self.model_name}_l2_{self._category}.joblib"
        if path.exists():
            self._bundle = joblib.load(path)
        else:
            logger.warning("L2 model not found: %s", path)

    def predict_window(self, X_row: np.ndarray) -> tuple[str, float]:
        self._load()
        if self._bundle is None:
            return ("n/a", 0.0)

        model = self._bundle["model"]
        le = self._bundle["label_encoder"]
        pred = model.predict(X_row.reshape(1, -1))
        label = le.inverse_transform(pred)[0]

        confidence = 0.0
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_row.reshape(1, -1))[0]
            confidence = float(proba.max())

        handler_fn = self._actions.get(label)
        if handler_fn is not None:
            confidence = handler_fn(X_row, proba, confidence)

        return (label, confidence)

    def finalize(self, top_l2: str, pose_df, result_df) -> tuple[str, dict]:
        return top_l2, {"fired": False, "drift": 0.0, "regularity": 0.0,
                        "decision": "", "score": 0}


class OtherHandler:
    """No-op handler for 'other' (stationary + transitional). No L2 model."""

    def __init__(self, model_name: str, model_dir: Path):
        pass

    def predict_window(self, X_row: np.ndarray) -> tuple[str, float]:
        return ("other", 0.0)

    def finalize(self, top_l2: str, pose_df, result_df) -> tuple[str, dict]:
        return "other", {"fired": False, "drift": 0.0, "regularity": 0.0,
                         "decision": "", "score": 0}


from .ambulatory import AmbulatoryHandler
from .fall import FallHandler

L2_REGISTRY: dict[str, type] = {
    "ambulatory": AmbulatoryHandler,
    "fall": FallHandler,
    "other": OtherHandler,
}


def get_l2_handler(l1_category: str) -> type:
    """Return the handler class for the given L1 category."""
    cls = L2_REGISTRY.get(l1_category)
    if cls is None:
        raise ValueError(f"No L2 handler for L1 category: {l1_category!r}")
    return cls