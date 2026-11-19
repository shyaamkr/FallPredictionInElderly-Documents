"""Annotated video generation - pose skeleton, bounding box, labels."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from .config import BASE_DIR, logger


L1_COLORS = {
    "ambulatory": (0, 200, 83),
    "fall": (0, 0, 213),
    "other": (180, 180, 180),
}

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


def _draw_pose_skeleton(frame, landmarks, color=(0, 255, 200)):
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
        if vis > 0.5:
            cv2.circle(frame, (px, py), 3, color, -1)

    for (i, j) in POSE_CONNECTIONS:
        if i < len(points) and j < len(points):
            v1 = landmarks[i].visibility if hasattr(landmarks[i], 'visibility') else 1.0
            v2 = landmarks[j].visibility if hasattr(landmarks[j], 'visibility') else 1.0
            if v1 > 0.5 and v2 > 0.5:
                cv2.line(frame, points[i], points[j], color, 2)
    return frame


def _draw_bounding_box(frame, landmarks, label: str, confidence: float,
                       color):
    h, w = frame.shape[:2]
    xs, ys = [], []
    for lm in landmarks:
        vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
        if vis > 0.5:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if not xs:
        return frame

    margin_x = int((max(xs) - min(xs)) * 0.1) + 10
    margin_y = int((max(ys) - min(ys)) * 0.1) + 10
    x1 = max(0, min(xs) - margin_x)
    y1 = max(0, min(ys) - margin_y)
    x2 = min(w, max(xs) + margin_x)
    y2 = min(h, max(ys) + margin_y)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), font, font_scale,
                (255, 255, 255), thickness)
    return frame


def _draw_disambiguation_overlay(frame, drift: float, regularity: float,
                                 decision: str, score: int = 0):
    h, w = frame.shape[:2]
    panel_w, panel_h = 320, 120
    x1 = w - panel_w - 10
    y1 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "DISAMBIGUATION", (x1 + 10, y1 + 22),
                font, 0.5, (0, 200, 255), 1)
    cv2.putText(frame, f"Score: {score}/5", (x1 + 10, y1 + 48),
                font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Drift: {drift:.3f}", (x1 + 10, y1 + 68),
                font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Regularity: {regularity:.3f}", (x1 + 10, y1 + 88),
                font, 0.5, (200, 200, 200), 1)

    decision_color = (0, 255, 100) if "jogging" in decision else (100, 100, 255)
    cv2.putText(frame, f"Decision: {decision}", (x1 + 10, y1 + 110),
                font, 0.5, decision_color, 1)
    return frame


def generate_annotated_output(video_path: str, result_df: pd.DataFrame,
                              top_l1: str, top_l2: str,
                              disambiguation_info: dict,
                              window_size: int, overlap: float) -> str:
    """Second pass over video - draws skeleton, bbox, labels."""
    vid_path = Path(video_path)
    out_base = BASE_DIR / "output" / "visualizations" / vid_path.stem
    out_base.mkdir(parents=True, exist_ok=True)
    keyframes_dir = out_base / "keyframes"
    keyframes_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        logger.error("Cannot open video for visualization: %s", video_path)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = int(window_size * (1 - overlap))
    n_windows = len(result_df)

    out_video_path = str(out_base / "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(BASE_DIR / "pose_landmark_full.task")
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    frame_idx = 0
    prev_window_idx = -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    disambiguation_evidence_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        window_idx = min(frame_idx // step, n_windows - 1)
        if window_idx < 0:
            window_idx = 0

        row = result_df.iloc[window_idx]
        l1_label = row["l1"]
        l2_label = row["l2"]
        l2_conf = row.get("l2_confidence", 0.0)
        color = L1_COLORS.get(l1_label, (200, 200, 200))

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            frame = _draw_pose_skeleton(frame, landmarks, color)
            display_label = f"{l1_label} > {l2_label}"
            frame = _draw_bounding_box(frame, landmarks, display_label,
                                       l2_conf, color)

        if disambiguation_info.get("fired"):
            frame = _draw_disambiguation_overlay(
                frame,
                disambiguation_info["drift"],
                disambiguation_info["regularity"],
                disambiguation_info["decision"],
                disambiguation_info.get("score", 0),
            )

        info_text = f"Frame {frame_idx} | Window {window_idx}/{n_windows-1}"
        cv2.putText(frame, info_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(frame)

        if window_idx != prev_window_idx:
            conf_pct = int(l2_conf * 100)
            fname = (f"window_{window_idx:02d}_{l1_label}_{l2_label}"
                     f"_{conf_pct}pct.png")
            cv2.imwrite(str(keyframes_dir / fname), frame)
            prev_window_idx = window_idx

        if (disambiguation_info.get("fired") and frame_idx == mid_frame_idx
                and not disambiguation_evidence_saved):
            cv2.imwrite(str(keyframes_dir / "disambiguation_evidence.png"),
                        frame)
            disambiguation_evidence_saved = True

        frame_idx += 1

    writer.release()
    cap.release()
    landmarker.close()

    logger.info("Annotated video saved: %s", out_video_path)
    logger.info("Key frames saved: %s", keyframes_dir)
    return str(out_base)