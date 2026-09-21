"""Noise Segregation V2: six-subframe, five-feature weighted-voting acoustic
scorer (design.md §6.2).

Each 3 s segment is split into 6 non-overlapping 0.5 s subframes. Five
hand-crafted features are computed per subframe, min-max normalised to [0, 1]
within the segment, and combined via a weighted sum (config-driven, DD-007).
A subframe is "noise-like" if its score exceeds 0.5; the segment's class is
decided by majority vote across the 6 subframes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, correlate, sosfilt

from pipeline.config import NoiseSegregationConfig

SUBFRAME_VOTE_THRESHOLD = 0.5


@dataclass
class NoiseSegregationResult:
    noise_score: float  # mean weighted subframe score, in [0, 1]
    noise_class: str  # "bird" or "noise"
    subframe_scores: list[float] = field(default_factory=list)
    subframe_votes: list[bool] = field(default_factory=list)  # True = noise-like


def zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    signs = np.sign(x)
    signs[signs == 0] = 1
    return float(np.mean(signs[:-1] != signs[1:]))


def spectral_flatness(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x)) + 1e-12
    power = spectrum**2
    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)
    return float(geometric_mean / arithmetic_mean) if arithmetic_mean > 0 else 0.0


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = np.sum(spectrum)
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def insect_periodicity_score(x: np.ndarray, sr: int, band_hz: tuple[int, int]) -> float:
    """Autocorrelation peak strength within the insect-chirp band (3-8 kHz)."""
    low, high = band_hz
    nyquist = sr / 2.0
    high_f = float(high)
    if len(x) < 16 or high_f >= nyquist:
        high_f = min(high_f, nyquist - 1)
    if low <= 0 or high_f <= low:
        return 0.0
    sos = butter(4, [low / nyquist, high_f / nyquist], btype="band", output="sos")
    banded = sosfilt(sos, x)

    autocorr = correlate(banded, banded, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    if autocorr[0] <= 0:
        return 0.0
    autocorr = autocorr / autocorr[0]

    # Ignore the zero-lag peak; look for the strongest periodic side-peak.
    if len(autocorr) < 3:
        return 0.0
    side_peak = float(np.max(autocorr[1:]))
    return max(0.0, side_peak)


def _clip01(values: list[float]) -> list[float]:
    return [float(np.clip(v, 0.0, 1.0)) for v in values]


def _split_subframes(
    audio: np.ndarray, sr: int, n_subframes: int, subframe_duration_s: float
) -> list[np.ndarray]:
    subframe_len = int(round(subframe_duration_s * sr))
    subframes = []
    for i in range(n_subframes):
        start = i * subframe_len
        end = start + subframe_len
        chunk = audio[start:end]
        if len(chunk) < subframe_len:
            chunk = np.pad(chunk, (0, subframe_len - len(chunk)))
        subframes.append(chunk)
    return subframes


def classify_segment(
    audio: np.ndarray, sr: int, config: NoiseSegregationConfig
) -> NoiseSegregationResult:
    """Run Noise Segregation V2 on a single 3 s segment."""
    subframes = _split_subframes(audio, sr, config.n_subframes, config.subframe_duration_s)

    zcrs = [zero_crossing_rate(sf_) for sf_ in subframes]
    flatnesses = [spectral_flatness(sf_) for sf_ in subframes]
    centroids = [spectral_centroid(sf_, sr) for sf_ in subframes]
    insect_scores = [insect_periodicity_score(sf_, sr, config.insect_band_hz) for sf_ in subframes]

    centroid_flags = [1.0 if c > config.centroid_threshold_hz else 0.0 for c in centroids]
    centroid_std = float(np.std(centroids))
    centroid_std_feature = [centroid_std] * len(subframes)  # segment-level, broadcast per subframe

    # Fixed-scale normalization (DD-016): spectral flatness and the insect
    # autocorrelation score are already bounded in [0, 1] by construction;
    # ZCR and centroid std are scaled against absolute reference constants
    # from config.yaml rather than the segment's own min/max.
    zcr_n = _clip01([z / config.zcr_reference for z in zcrs])
    flatness_n = _clip01(flatnesses)
    cflag_n = centroid_flags
    cstd_n = _clip01([c / config.centroid_std_reference_hz for c in centroid_std_feature])
    insect_n = _clip01(insect_scores)

    w = config.weights
    subframe_scores = [
        w.zcr * zcr_n[i]
        + w.spectral_flatness * flatness_n[i]
        + w.centroid_flag * cflag_n[i]
        + w.centroid_std * cstd_n[i]
        + w.insect_periodicity * insect_n[i]
        for i in range(len(subframes))
    ]
    subframe_votes = [score > SUBFRAME_VOTE_THRESHOLD for score in subframe_scores]

    n_noise_votes = sum(subframe_votes)
    noise_class = "noise" if n_noise_votes > len(subframe_votes) / 2 else "bird"

    return NoiseSegregationResult(
        noise_score=float(np.mean(subframe_scores)),
        noise_class=noise_class,
        subframe_scores=subframe_scores,
        subframe_votes=subframe_votes,
    )
