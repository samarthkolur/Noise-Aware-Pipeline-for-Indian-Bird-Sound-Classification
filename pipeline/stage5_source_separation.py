"""
Stage 5: Source Separation Refinement

Apply Harmonic-Percussive Source Separation (HPSS) to decompose audio
segments. Bird calls are predominantly harmonic — segments dominated by
percussive energy are likely noise.
"""

import numpy as np
import librosa

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def apply_hpss(audio: np.ndarray, sr: int = None) -> tuple:
    """
    Decompose audio into harmonic and percussive components using librosa HPSS.

    Args:
        audio: Audio time series.
        sr: Sample rate (unused by HPSS, kept for API consistency).

    Returns:
        Tuple of (harmonic, percussive) audio arrays.
    """
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive


def compute_harmonic_ratio(harmonic: np.ndarray, percussive: np.ndarray) -> float:
    """
    Compute harmonic energy ratio.

    Ratio = harmonic_energy / (harmonic_energy + percussive_energy + eps)

    A ratio close to 1.0 indicates primarily harmonic content (likely bird).
    A ratio close to 0.0 indicates primarily percussive/broadband content (likely noise).

    Returns:
        Float in [0, 1].
    """
    h_energy = np.sum(harmonic ** 2)
    p_energy = np.sum(percussive ** 2)
    eps = 1e-10
    return float(h_energy / (h_energy + p_energy + eps))


def compute_harmonic_ratio_from_audio(audio: np.ndarray, sr: int = None) -> float:
    """
    One-shot: decompose audio and return harmonic ratio.

    Args:
        audio: Raw audio array.
        sr: Sample rate.

    Returns:
        Harmonic energy ratio in [0, 1].
    """
    harmonic, percussive = apply_hpss(audio, sr)
    return compute_harmonic_ratio(harmonic, percussive)


def filter_by_harmonicity(
    segments: list,
    threshold: float = None,
) -> tuple:
    """
    Filter a list of audio segments by harmonic energy ratio.

    Args:
        segments: List of audio arrays.
        threshold: Minimum harmonic ratio to keep. Defaults to config value.

    Returns:
        Tuple of (passed_segments, passed_indices, ratios).
    """
    threshold = threshold or config.HARMONIC_RATIO_THRESHOLD

    passed = []
    passed_indices = []
    ratios = []

    for i, seg in enumerate(segments):
        ratio = compute_harmonic_ratio_from_audio(seg)
        ratios.append(ratio)
        if ratio >= threshold:
            passed.append(seg)
            passed_indices.append(i)

    return passed, passed_indices, ratios


def batch_harmonic_ratios(
    audio_segments: list,
) -> np.ndarray:
    """
    Compute harmonic ratios for a batch of audio segments.

    Args:
        audio_segments: List of numpy audio arrays.

    Returns:
        1-D numpy array of harmonic ratios.
    """
    ratios = []
    for seg in audio_segments:
        ratios.append(compute_harmonic_ratio_from_audio(seg))
    return np.array(ratios, dtype=np.float32)


if __name__ == "__main__":
    print("Stage 5: Use run_pipeline.py to run source separation.")
