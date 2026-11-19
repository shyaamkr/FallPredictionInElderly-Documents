[project_readme.md](https://github.com/user-attachments/files/35106412/project_readme.md)
# upfallv3.0 — Multi-Sensor Fall-Risk Activity Classifier

A tri-modal (video + IMU + wearable vitals) machine learning pipeline for human activity recognition, fall detection, and fall-risk assessment. Built around the [FigShare Sensor-Based Fall Detection Dataset](https://figshare.com/articles/dataset/28596332/2).

## Repository structure

```
activity_classifier_v5/             Current pipeline (package)
  config.py                         Paths, label mappings, SMOOTH_K
  features.py                       Feature engineering, wearable windowing
  models.py                         create_model() factory (5 model types)
  training.py                       Modality-dropout augmentation + training
  prediction.py                     predict_video(), risk score, fall confirmation
  evaluation.py                     Cross-validated evaluation + comparison tables
  smoothing.py                      Confidence filter + temporal smoothing
  visualization.py                  Annotated video / keyframe generation
  l2/                               Per-L1-category fine-classification handlers
  cli.py                            python -m activity_classifier_v5 entry point

activity_classifier.py              Shared extraction code (pose/IMU, windowing) + v1
activity_classifier_v2/v3/v4.py     Earlier iterations, kept for comparison
compare_v5_models.py                Run all 5 models against one video, side by side
dataset_analysis.py                 Plotly HTML dataset dashboard
generate_synthetic_imu.py           Synthetic IMU generator (for IMU-absent eval clips)
generate_wearable_fall_data.py      Synthetic wearable-vitals generator

config.json                         All paths, hyperparameters, activity mappings
requirements.txt
pose_landmark_full.task             MediaPipe pose model

docs/
  thesis_progress_report.md         Full thesis chapter (abstract -> methodology -> results)
  viva_prep_notes.md                Feature/logic reference + probable examiner Q&A
  RISK_SCORE_CHANGELOG.md           Dated log of every risk-score logic change and why
  GCP_PORTING_INSTRUCTIONS.txt      Screenshot-based code porting workflow (restricted env)
  COLAB_PORTING_INSTRUCTIONS.txt    Same, adapted for Google Colab

tests/                              pytest suite
output/                             Generated — features, models, plots, logs (git-ignored
                                    except small reference files; see .gitignore)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Requires the [FigShare dataset](https://figshare.com/articles/dataset/28596332/2) downloaded locally and `config.json`'s `data_root` pointed at it, plus `pose_landmark_full.task` in the project root (downloadable from [MediaPipe's model repo](https://storage.googleapis.com/mediapipe-models/pose_landmark/pose_landmark_full/float16/latest/pose_landmark_full.task)).

## Usage

```bash
# Extract features from the dataset (one-time, ~2h+ for the full corpus)
python -m activity_classifier_v5 --extract
python -m activity_classifier_v5 --merge-wearable

# Train (per model, or all 5)
python -m activity_classifier_v5 --train --model xgboost --level both
python -m activity_classifier_v5 --train --model all

# Evaluate — prints a 4-way L1 comparison table + confusion matrices
python -m activity_classifier_v5 --evaluate --model all --level l1

# Predict on a single video
python -m activity_classifier_v5 --predict "path/to/video.mp4" --model ensemble
python -m activity_classifier_v5 --predict-detailed "path/to/video.mp4" --model ensemble --visualize

# Compare all 5 models on the same video
python compare_v5_models.py --video "path/to/video.mp4" --model all --detailed --visualize
```
