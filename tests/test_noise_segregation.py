import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.noise_segregation import classify_segment

SR = 48000
DURATION_S = 3.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _time_axis():
    return np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)


def test_pure_sine_is_bird_like(cfg):
    t = _time_axis()
    sine = (0.8 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    result = classify_segment(sine, SR, cfg.noise_segregation)
    assert result.noise_class == "bird"


def test_white_noise_is_noise_like(cfg):
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, int(SR * DURATION_S)).astype(np.float32)
    white = 0.8 * white / np.max(np.abs(white))
    result = classify_segment(white, SR, cfg.noise_segregation)
    assert result.noise_class == "noise"


def test_majority_vote_matches_subframe_votes(cfg):
    t = _time_axis()
    sine = (0.8 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    result = classify_segment(sine, SR, cfg.noise_segregation)
    n_noise_votes = sum(result.subframe_votes)
    expected = "noise" if n_noise_votes > len(result.subframe_votes) / 2 else "bird"
    assert result.noise_class == expected


def test_returns_six_subframe_scores(cfg):
    t = _time_axis()
    sine = (0.8 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    result = classify_segment(sine, SR, cfg.noise_segregation)
    assert len(result.subframe_scores) == cfg.noise_segregation.n_subframes
    assert len(result.subframe_votes) == cfg.noise_segregation.n_subframes
