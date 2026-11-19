"""Global configuration, paths, and logging for v5."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity_classifier import load_config

VERSION = "5.0"
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
CONFIG = load_config(CONFIG_PATH)

_OUTPUT_DIR = Path(CONFIG["output_dir"])
MODEL_DIR = _OUTPUT_DIR / "models_v5"
PLOTS_DIR = _OUTPUT_DIR / "plots_v5"
FEATURES_DIR = _OUTPUT_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_K = 5

L1_MAPPING = {
    "fall": list(range(1, 13)),
    "ambulatory": list(range(13, 18)),
    "other": list(range(18, 26)),
}

_ACT_L1_MAP = CONFIG["activity_mapping"]["11"]
_ACT_L2_MAP = CONFIG["activity_mapping"]["12"]

ACT_TO_L1: dict = {}
for _cat, _ids in _ACT_L1_MAP.items():
    for _id in _ids:
        ACT_TO_L1[_id] = _cat

ACT_TO_L2: dict = {}
for _subtypes in _ACT_L2_MAP.values():
    for _subtype, _ids in _subtypes.items():
        for _id in _ids:
            ACT_TO_L2[_id] = _subtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] v5: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("v5")