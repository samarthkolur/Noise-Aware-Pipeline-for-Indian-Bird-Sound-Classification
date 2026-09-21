"""Synthetic noise generation for the noise class (DD-010).

Opt-in only: gated behind config.synthetic.allow_synthetic_noise. Never used
by default. A real iBC53 noise-class corpus (real-world traffic/insect/urban
recordings) was not available in this environment — see design.md §25 Known
Issues — so this module exists to make the pipeline exercisable end-to-end
during development. Any metric derived from synthetic noise must not be
reported as a paper reproduction result.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

NOISE_GENERATORS = ("white", "pink", "brown", "band_limited")


def _white_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, n_samples).astype(np.float32)


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    # Voss-McCartney approximation via 1/f spectral shaping in the FFT domain.
    white = rng.normal(0.0, 1.0, n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples)
    return pink.astype(np.float32)


def _brown_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, n_samples)
    brown = np.cumsum(white)
    return brown.astype(np.float32)


def _band_limited_noise(
    n_samples: int, sr: int, rng: np.random.Generator, low_hz: float = 3000, high_hz: float = 8000
) -> np.ndarray:
    white = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
    nyquist = sr / 2.0
    sos = butter(4, [low_hz / nyquist, min(high_hz / nyquist, 0.99)], btype="band", output="sos")
    return sosfilt(sos, white).astype(np.float32)


def generate_noise_segment(
    noise_type: str, n_samples: int, sr: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate one synthetic noise segment of the given type, peak-normalised."""
    rng = rng or np.random.default_rng()
    if noise_type == "white":
        signal = _white_noise(n_samples, rng)
    elif noise_type == "pink":
        signal = _pink_noise(n_samples, rng)
    elif noise_type == "brown":
        signal = _brown_noise(n_samples, rng)
    elif noise_type == "band_limited":
        signal = _band_limited_noise(n_samples, sr, rng)
    else:
        raise ValueError(f"Unknown synthetic noise type: {noise_type!r}")

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.9 * signal / peak
    return signal.astype(np.float32)


def generate_synthetic_noise_corpus(
    n_segments: int,
    sr: int,
    segment_length_s: float,
    noise_types: list[str],
    allow_synthetic_noise: bool,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate a corpus of synthetic noise segments, gated by allow_synthetic_noise.

    Raises if called without the opt-in flag; always logs a warning when active,
    per CLAUDE.md's ML-specific standard ("Synthetic data is opt-in, never the
    default... must log a clear warning when active").
    """
    if not allow_synthetic_noise:
        raise RuntimeError(
            "Synthetic noise generation requires synthetic.allow_synthetic_noise: true "
            "in config.yaml. It is opt-in and must never run by default (DD-010)."
        )

    warnings.warn(
        "Generating SYNTHETIC noise segments (not real-world recordings). "
        "Metrics computed against this data must not be reported as a paper "
        "reproduction result. See design.md Known Issues.",
        stacklevel=2,
    )
    logger.warning(
        "SYNTHETIC NOISE ACTIVE: generating %d synthetic segments (%s) — "
        "real iBC53 noise corpus was unavailable.",
        n_segments,
        ", ".join(noise_types),
    )

    rng = np.random.default_rng(seed)
    n_samples = int(round(segment_length_s * sr))
    segments = []
    for i in range(n_segments):
        noise_type = noise_types[i % len(noise_types)]
        segments.append(generate_noise_segment(noise_type, n_samples, sr, rng))
    return segments
