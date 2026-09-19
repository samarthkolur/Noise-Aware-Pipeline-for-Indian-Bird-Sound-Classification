"""
Stage 7: Post-Processing & Ecological Validation

Apply multi-signal refinement after classification:
    1. Spectral sanity check (HPSS harmonic ratio) — rejects broadband noise
    2. Temporal consistency smoothing — removes isolated false positives
    3. Ecological priors (optional) — rejects implausible species detections

Research shows these post-processing steps "massively improve precision
with a negligible cost to recall."

Output:
    Refined prediction arrays (labels + confidences).
"""

import numpy as np
import librosa

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Spectral / HPSS Check ──────────────────────────────────────────────────

def compute_harmonic_ratio(audio: np.ndarray, sr: int = None) -> float:
    """
    Compute harmonic energy ratio using HPSS.

    Bird calls are predominantly harmonic (ratio > 0.5).
    Noise is predominantly percussive or broadband (ratio < 0.3).

    Args:
        audio: Raw audio array.
        sr: Sample rate (unused by HPSS, kept for API consistency).

    Returns:
        Harmonic energy ratio in [0, 1].
    """
    harmonic, percussive = librosa.effects.hpss(audio)
    h_energy = np.sum(harmonic ** 2)
    p_energy = np.sum(percussive ** 2)
    eps = 1e-10
    return float(h_energy / (h_energy + p_energy + eps))


def apply_spectral_check(
    file_paths: np.ndarray,
    threshold: float = None,
) -> np.ndarray:
    """
    Compute harmonic ratios for segments. Used as a post-processing signal.

    Segments with harmonic ratio below threshold are likely noise.

    Args:
        file_paths: Array of audio file paths.
        threshold: Minimum harmonic ratio. Defaults to config value.

    Returns:
        (N,) array of harmonic ratios.
    """
    threshold = threshold or config.HARMONIC_RATIO_THRESHOLD

    ratios = np.zeros(len(file_paths), dtype=np.float32)
    for i, fpath in enumerate(file_paths):
        try:
            y, sr = librosa.load(str(fpath), sr=config.TARGET_SR)
            ratios[i] = compute_harmonic_ratio(y, sr)
        except Exception:
            ratios[i] = 0.0

        if (i + 1) % 500 == 0:
            print(f"    [Stage 7] Spectral check: {i+1}/{len(file_paths)}...")

    above = int((ratios >= threshold).sum())
    print(f"  [Stage 7] Spectral check: {above}/{len(ratios)} above "
          f"harmonic threshold ({threshold})")

    return ratios


# ─── Temporal Smoothing ──────────────────────────────────────────────────────

def sliding_window_smooth(
    predictions: np.ndarray,
    window_size: int = None,
) -> np.ndarray:
    """
    Apply a sliding window average to predictions.

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
    pad_width = window_size // 2
    padded = np.pad(predictions, pad_width, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(predictions)]


def majority_vote(
    predictions: np.ndarray,
    window_size: int = None,
) -> np.ndarray:
    """
    Apply majority voting across adjacent segments.

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
) -> tuple:
    """
    Average confidence scores across adjacent segments, then re-threshold.

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


# ─── Ecological Priors (Optional) ───────────────────────────────────────────

def apply_ecological_priors(
    predictions: np.ndarray,
    species_labels: np.ndarray = None,
    priors_path: str = None,
) -> np.ndarray:
    """
    Optionally reject ecologically implausible detections.

    If a species-location prior matrix is provided, segments predicted as
    bird for species that don't exist in the recording location are rejected.

    Note: This is a stub — requires external data (species occurrence matrices)
    for full implementation. Currently returns predictions unchanged.

    Args:
        predictions: Binary labels.
        species_labels: Species labels for each segment.
        priors_path: Path to species-location priors JSON.

    Returns:
        Filtered predictions.
    """
    if not config.ENABLE_ECOLOGICAL_PRIORS:
        return predictions

    priors_path = priors_path or config.SPECIES_LOCATION_PRIORS
    if priors_path is None or not os.path.exists(priors_path):
        print("  [Stage 7] No ecological priors file — skipping")
        return predictions

    # Future: load priors and filter predictions
    print("  [Stage 7] Ecological priors: not yet implemented (pass-through)")
    return predictions


# ─── Combined Post-Processing ────────────────────────────────────────────────

def postprocess_predictions(
    predictions: np.ndarray,
    confidences: np.ndarray,
    file_paths: np.ndarray = None,
    harmonic_ratios: np.ndarray = None,
) -> tuple:
    """
    Apply all post-processing steps in sequence.

    1. Temporal smoothing (on predictions)
    2. Spectral check override (bird with low harmonic ratio → noise)

    Args:
        predictions: Binary labels from classifier.
        confidences: Classifier probabilities.
        file_paths: Audio file paths (for spectral check if ratios not cached).
        harmonic_ratios: Pre-computed harmonic ratios (optional).

    Returns:
        Tuple of (refined_predictions, harmonic_ratios).
    """
    print("\n  [Stage 7] Post-processing predictions...")

    # Step 1: Temporal smoothing
    if config.ENABLE_TEMPORAL_SMOOTHING:
        print("  Applying temporal smoothing...")
        refined = apply_temporal_smoothing(predictions, confidences)
    else:
        refined = predictions.copy()

    # Step 2: Spectral check — override bird predictions with low harmonic ratio
    if harmonic_ratios is None and file_paths is not None:
        print("  Computing harmonic ratios...")
        harmonic_ratios = apply_spectral_check(file_paths)

    if harmonic_ratios is not None:
        n_overrides = 0
        for i in range(len(refined)):
            if (refined[i] == config.BIRD_LABEL and
                    harmonic_ratios[i] < config.HARMONIC_RATIO_THRESHOLD):
                refined[i] = config.NOISE_LABEL
                n_overrides += 1
        print(f"  [Stage 7] Spectral override: {n_overrides} bird→noise")

    n_bird = int((refined == config.BIRD_LABEL).sum())
    n_noise = int((refined == config.NOISE_LABEL).sum())
    print(f"  [Stage 7] Post-processed: {n_bird} bird, {n_noise} noise")

    return refined, harmonic_ratios
