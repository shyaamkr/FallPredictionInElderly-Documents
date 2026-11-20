"""Run activity_classifier_v5 with C++ LiteRT logs hidden. Use this instead of python -m ..."""
import os
import subprocess
import sys

# Must be set BEFORE Python starts mediapipe - so we pass them to the child process
env = os.environ.copy()
env["GLOG_minloglevel"] = "3"
env["TF_CPP_MIN_LOG_LEVEL"] = "3"
env["ABSL_MIN_LOG_LEVEL"] = "3"
env["LITERT_LOG_LEVEL"] = "3"
env["MEDIAPIPE_DISABLE_GPU"] = "1"
env["PYTHONWARNINGS"] = "ignore"

# Pass through all your args: --predict-detailed "..." --model xgboost --visualize
cmd = [sys.executable, "-m", "activity_classifier_v5"] + sys.argv[1:]

# stderr=DEVNULL hides ALL C++ LiteRT / Sparsity warnings
# stdout stays visible, so you still see your own [INFO] + results
result = subprocess.run(cmd, env=env, stderr=subprocess.DEVNULL)
sys.exit(result.returncode)