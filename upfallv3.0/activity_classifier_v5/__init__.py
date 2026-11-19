"""Activity Classifier v5 - Tri-Modal (Video + IMU + Wearable) Fall Prediction."""

from .config import VERSION, MODEL_DIR
from .prediction import predict_video, compare_predictions
from .training import train_model
from .evaluation import run_evaluation