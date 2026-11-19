"""CLI entry point for v5 - tri-modal fall prediction (video + IMU + wearable)."""

import argparse
import sys
from pathlib import Path

from .config import CONFIG, MODEL_DIR, logger
from .prediction import predict_video, compare_predictions
from .training import train_model
from .evaluation import run_evaluation
from .features import merge_wearable_features

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity_classifier import load_config, run_extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activity Classifier v5 - Tri-Modal "
                    "(Video + IMU + Wearable) Fall Prediction")
    parser.add_argument("--extract", action="store_true",
                        help="Extract features from videos")
    parser.add_argument("--merge-wearable", action="store_true",
                        help="Merge wearable sensor CSVs into feature parquet")
    parser.add_argument("--train", action="store_true",
                        help="Train classifiers (4-way L1 + L2)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model(s)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Classify activity in a video")
    parser.add_argument("--predict-detailed", type=str, default=None,
                        help="Per-window breakdown")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare v1/v2/v3/v4/v5 on a video")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "random_forest", "lightgbm",
                                 "svm", "ensemble", "all"],
                        help="Model to use")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit videos for extraction")
    parser.add_argument("--level", type=str, default="both",
                        choices=["l1", "l2", "both"],
                        help="Classification level")
    parser.add_argument("--config", type=str, default=None,
                        help="Alternate config file")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root directory")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate annotated video with bbox, skeleton, "
                             "and labels")
    args = parser.parse_args()

    if args.config:
        import activity_classifier_v5.config as cfg
        cfg.CONFIG = load_config(Path(args.config))

    data_root = (Path(args.data_root) if args.data_root
                 else Path(CONFIG["data_root"]))

    if not any([args.extract, args.merge_wearable, args.train, args.evaluate,
                args.predict, args.predict_detailed, args.compare]):
        parser.print_help()
        sys.exit(0)

    if args.extract:
        run_extraction(data_root, args.max_videos)

    if args.merge_wearable:
        logger.info("Merging wearable sensor features into parquet...")
        merge_wearable_features()

    if args.train:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if args.model == "all":
            for m in ("xgboost", "random_forest", "lightgbm", "svm",
                      "ensemble"):
                logger.info("Training %s ...", m)
                train_model(m, level=args.level)
        else:
            logger.info("Training %s ...", args.model)
            train_model(args.model, level=args.level)

    if args.evaluate:
        run_evaluation(args.model, level=args.level)

    if args.predict:
        predict_video(args.predict, args.model, detailed=False,
                      visualize=args.visualize)

    if args.predict_detailed:
        predict_video(args.predict_detailed, args.model, detailed=True,
                      visualize=args.visualize)

    if args.compare:
        compare_predictions(args.compare, args.model)