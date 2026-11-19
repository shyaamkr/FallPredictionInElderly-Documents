"""L2 handler for the fall category.

Actions: forward_fall, backward_fall, lateral_fall, tripping, syncope, slip.
"""

import numpy as np


# -- Per-action handlers ---------------------------------------------------

def _handle_forward_fall(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_backward_fall(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_lateral_fall(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_tripping(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_syncope(X_row, proba, default_confidence: float) -> float:
    return default_confidence


def _handle_slip(X_row, proba, default_confidence: float) -> float:
    return default_confidence


# -- Handler class ---------------------------------------------------------

class FallHandler:
    """L2 handler for fall activities."""

    _category = "fall"

    def __init__(self, model_name: str, model_dir):
        from . import BaseL2Handler
        self._base = BaseL2Handler(model_name, model_dir)
        self._base._category = self._category
        self._base._actions = {
            "forward_fall": _handle_forward_fall,
            "backward_fall": _handle_backward_fall,
            "lateral_fall": _handle_lateral_fall,
            "tripping": _handle_tripping,
            "syncope": _handle_syncope,
            "slip": _handle_slip,
        }
        self.model_name = model_name
        self.model_dir = model_dir

    def predict_window(self, X_row: np.ndarray) -> tuple[str, float]:
        return self._base.predict_window(X_row)

    def finalize(self, top_l2: str, pose_df, result_df) -> tuple[str, dict]:
        return top_l2, {"fired": False, "drift": 0.0, "regularity": 0.0,
                        "decision": "", "score": 0}