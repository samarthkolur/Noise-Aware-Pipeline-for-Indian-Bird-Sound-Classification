import numpy as np
import pytest

from pipeline.synthetic import (
    NOISE_GENERATORS,
    generate_noise_segment,
    generate_synthetic_noise_corpus,
)


@pytest.mark.parametrize("noise_type", NOISE_GENERATORS)
def test_generate_noise_segment_shape_and_peak(noise_type):
    sr = 48000
    n_samples = sr * 3
    signal = generate_noise_segment(noise_type, n_samples, sr, rng=np.random.default_rng(0))
    assert signal.shape == (n_samples,)
    assert signal.dtype == np.float32
    assert np.max(np.abs(signal)) <= 0.91


def test_generate_noise_segment_unknown_type_raises():
    with pytest.raises(ValueError):
        generate_noise_segment("not_a_real_type", 1000, 48000)


def test_generate_synthetic_noise_corpus_requires_opt_in():
    with pytest.raises(RuntimeError):
        generate_synthetic_noise_corpus(
            n_segments=5,
            sr=48000,
            segment_length_s=3.0,
            noise_types=["white"],
            allow_synthetic_noise=False,
        )


def test_generate_synthetic_noise_corpus_produces_requested_count():
    segments = generate_synthetic_noise_corpus(
        n_segments=7,
        sr=48000,
        segment_length_s=1.0,
        noise_types=["white", "pink"],
        allow_synthetic_noise=True,
        seed=0,
    )
    assert len(segments) == 7
    assert all(s.shape == (48000,) for s in segments)


def test_generate_synthetic_noise_corpus_warns(recwarn):
    generate_synthetic_noise_corpus(
        n_segments=1,
        sr=48000,
        segment_length_s=1.0,
        noise_types=["white"],
        allow_synthetic_noise=True,
        seed=0,
    )
    assert any("SYNTHETIC" in str(w.message) for w in recwarn.list)
