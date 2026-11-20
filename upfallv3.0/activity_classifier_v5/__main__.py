"""Allow running as: python -m activity_classifier_v5"""

# MUST BE FIRST - before ANY other import (cli -> mediapipe -> LiteRT)
# If this is after `from .cli import main`, LiteRT already printed.
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["LITERT_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

import warnings
import logging
warnings.filterwarnings("ignore")

# Hide absl / mediapipe / LiteRT INFO + WARNING, keep ERROR only
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold(absl_logging.ERROR)
except Exception:
    pass

# Don't use logging.disable(logging.INFO) - it also kills your own
# [INFO] activity_classifier logger. Set levels instead:
for _name in ("mediapipe", "absl", "tensorflow", "liteRT", "LiteRT"):
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass

# Only NOW import cli - cli imports mediapipe / sklearn / etc.
from .cli import main

if __name__ == "__main__":
    main()