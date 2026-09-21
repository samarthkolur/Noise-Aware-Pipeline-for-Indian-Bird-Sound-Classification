"""Bird Guard: harmonic-structure override for Noise Segregation V2 (design.md §6.2, §20).

Accepts a segment that Noise Segregation V2 routed to the noise class and
checks for strong harmonic content or a prominent spectral peak — both
signatures of a tonal bird call that the broadband/impulsive noise features
can misfire on. If either check passes, the segment is retained as bird-like
regardless of V2's vote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from librosa.effects import hpss
from scipy.signal import welch

from pipeline.config import BirdGuardConfig


@dataclass
class BirdGuardResult:
    harmonic_ratio: float
    peak_median_ratio: float
    triggered: bool


def harmonic_ratio(audio: np.ndarray) -> float:
    """Fraction of signal energy in the harmonic component (via HPSS)."""
    if np.allclose(audio, 0.0):
        return 0.0
    harmonic, percussive = hpss(audio.astype(np.float32))
    harmonic_energy = float(np.sum(harmonic**2))
    total_energy = float(np.sum(audio**2))
    return harmonic_energy / total_energy if total_energy > 0 else 0.0


def spectral_peak_to_median_ratio(audio: np.ndarray, sr: int = 48000) -> float:
    """Ratio of the dominant PSD peak to the median PSD, via Welch's method.

    A single raw FFT over a whole 3 s clip has tens of thousands of bins;
    extreme-value statistics alone push even white noise's max/median ratio
    well above design.md's threshold (3.0). Welch's method averages multiple
    overlapping windows, smoothing the noise floor while preserving a
    genuine narrowband tone's peak — matching the qualitative "prominent
    spectral peak" check the paper intends.
    """
    nperseg = min(2048, len(audio))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg)
    median = np.median(psd)
    if median <= 0:
        return 0.0
    return float(np.max(psd) / median)


def apply_bird_guard(
    audio: np.ndarray, config: BirdGuardConfig, sr: int = 48000
) -> BirdGuardResult:
    """Compute Bird Guard's harmonic/peak checks and whether they override to bird-like."""
    h_ratio = harmonic_ratio(audio)
    peak_ratio = spectral_peak_to_median_ratio(audio, sr)
    triggered = (
        h_ratio > config.harmonic_ratio_threshold or peak_ratio > config.peak_median_threshold
    )
    return BirdGuardResult(
        harmonic_ratio=h_ratio, peak_median_ratio=peak_ratio, triggered=triggered
    )
