"""
Stage 6: Temporal Consistency Modeling

Smooth predictions across adjacent time segments to reduce isolated
false positives. Bird vocalizations exhibit temporal continuity, while
noise bursts are typically isolated.
"""

import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def sliding_window_smooth(
    predictions: np.ndarray,
    window_size: int = None,
) -> np.ndarray:
    """
    Apply a sliding window average to binary predictions.

    Args:
        predictions: Array of probabilities or binary predictions.
        window_size: Number of adjacent segments to average over.

    Returns:
        Smoothed predictions (same length as input).
    """
    window_size = window_size or config.TEMPORAL_WINDOW_SIZE
    if len(predictions) <= window_size:
        return predictions.copy()

    kernel = np.ones(window_size) / window_size
    # Pad edges to maintain length
    pad_width = window_size // 2
    padded = np.pad(predictions, pad_width, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    # Ensure output has same length as input
    return smoothed[: len(predictions)]


def majority_vote(
    predictions: np.ndarray,
    window_size: int = None,
) -> np.ndarray:
    """
    Apply majority voting across adjacent segments.

    For each segment, the prediction becomes the majority class
    within the surrounding window.

    Args:
        predictions: Array of binary labels (0 or 1).
        window_size: Window size for voting.

    Returns:
        Array of voted binary labels.
    """
    window_size = window_size or config.TEMPORAL_WINDOW_SIZE
    half = window_size // 2
    n = len(predictions)
    voted = np.zeros(n, dtype=int)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = predictions[start:end]
        voted[i] = 1 if np.sum(window) > len(window) / 2 else 0

    return voted


def confidence_average(
    confidences: np.ndarray,
    window_size: int = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Average confidence scores across adjacent segments, then re-threshold.

    Args:
        confidences: Array of continuous confidence scores [0, 1].
        window_size: Window size for averaging.
        threshold: Threshold to re-binarize after averaging.

    Returns:
        Tuple of (smoothed_confidences, binary_labels).
    """
    smoothed = sliding_window_smooth(confidences, window_size)
    labels = (smoothed >= threshold).astype(int)
    return smoothed, labels


def apply_temporal_smoothing(
    predictions: np.ndarray,
    confidences: np.ndarray = None,
    method: str = None,
    window_size: int = None,
) -> np.ndarray:
    """
    Apply temporal smoothing to segment predictions.

    Dispatcher function that calls the appropriate smoothing method.

    Args:
        predictions: Binary labels or probabilities.
        confidences: Confidence scores (required for confidence_avg method).
        method: Smoothing method name. Defaults to config value.
        window_size: Override window size.

    Returns:
        Smoothed predictions (binary labels).
    """
    method = method or config.TEMPORAL_METHOD

    if method == "sliding":
        smoothed = sliding_window_smooth(predictions.astype(float), window_size)
        return (smoothed >= 0.5).astype(int)

    elif method == "majority":
        return majority_vote(predictions.astype(int), window_size)

    elif method == "confidence_avg":
        if confidences is None:
            confidences = predictions.astype(float)
        _, labels = confidence_average(confidences, window_size)
        return labels

    else:
        raise ValueError(f"Unknown temporal method: {method}")


if __name__ == "__main__":
    print("Stage 6: Use run_pipeline.py to apply temporal smoothing.")
