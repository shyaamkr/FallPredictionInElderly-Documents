"""Temporal smoothing and confidence filtering for predictions."""

from collections import Counter

import numpy as np

from .config import SMOOTH_K


def temporal_smooth(labels: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    """Apply sliding majority vote of window size k."""
    if len(labels) < k:
        return labels.copy()
    smoothed = []
    half_k = k // 2
    for i in range(len(labels)):
        start = max(0, i - half_k)
        end = min(len(labels), i + half_k + 1)
        counts = Counter(labels[start:end])
        smoothed.append(counts.most_common(1)[0][0])
    return np.array(smoothed)


def confidence_filter(proba: np.ndarray, labels: np.ndarray,
                      threshold: float = 0.4) -> np.ndarray:
    """Replace low-confidence predictions with neighbor consensus."""
    result = labels.copy()
    max_conf = proba.max(axis=1)
    for i in range(len(result)):
        if max_conf[i] < threshold:
            start = max(0, i - 2)
            end = min(len(result), i + 3)
            neighbors = [labels[j] for j in range(start, end)
                         if j != i and max_conf[j] >= threshold]
            if neighbors:
                result[i] = Counter(neighbors).most_common(1)[0][0]
    return result