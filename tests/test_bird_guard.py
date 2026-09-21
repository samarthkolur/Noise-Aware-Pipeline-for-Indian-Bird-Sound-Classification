import numpy as np
import pytest

from pipeline.bird_guard import apply_bird_guard, harmonic_ratio, spectral_peak_to_median_ratio
from pipeline.config import load_config
from pipeline.noise_segregation import classify_segment

SR = 48000
DURATION_S = 3.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _time_axis():
    return np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)


def _noisy_harmonic_signal(noise_scale: float, seed: int = 1) -> np.ndarray:
    """A harmonic tone buried under noise_scale * white noise, peak-normalised."""
    t = _time_axis()
    carrier = (
        0.5 * np.sin(2 * np.pi * 2000 * t)
        + 0.3 * np.sin(2 * np.pi * 4000 * t)
        + 0.2 * np.sin(2 * np.pi * 6000 * t)
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(t))
    noise = noise / np.max(np.abs(noise))
    signal = carrier + noise_scale * noise
    return (0.9 * signal / np.max(np.abs(signal))).astype(np.float32)


def test_bird_guard_retains_harmonic_signal_that_v2_routes_to_noise(cfg):
    """Phase 3 exit criterion (design.md §10): Bird Guard retains a harmonic
    test signal that Noise Segregation V2 routes to noise."""
    signal = _noisy_harmonic_signal(noise_scale=2.0)

    v2_result = classify_segment(signal, SR, cfg.noise_segregation)
    assert v2_result.noise_class == "noise"

    guard_result = apply_bird_guard(signal, cfg.bird_guard)
    assert guard_result.triggered is True


def test_harmonic_ratio_high_for_pure_tone():
    t = _time_axis()
    sine = (0.8 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    assert harmonic_ratio(sine) > 0.9


def test_harmonic_ratio_low_for_white_noise():
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, int(SR * DURATION_S)).astype(np.float32)
    white = 0.8 * white / np.max(np.abs(white))
    assert harmonic_ratio(white) < 0.5


def test_spectral_peak_to_median_high_for_pure_tone():
    t = _time_axis()
    sine = (0.8 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    assert spectral_peak_to_median_ratio(sine) > 3.0


def test_bird_guard_not_triggered_for_white_noise(cfg):
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, int(SR * DURATION_S)).astype(np.float32)
    white = 0.8 * white / np.max(np.abs(white))
    result = apply_bird_guard(white, cfg.bird_guard)
    assert result.triggered is False
