"""
Test: Noise Segregation V2 — Feature extractors and scoring engine.

Validates that:
    1. Feature extractors return values in expected ranges
    2. NoiseSegregatorV2 scoring produces valid noise scores
    3. All ablation configurations are loadable and functional
    4. Factory function works for all named configs
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.noise_segregation_v2 import (
    compute_spectral_flatness,
    compute_zcr_normalized,
    compute_insect_periodicity,
    NoiseSegregatorV2,
    create_segregator,
    get_available_configs,
    ABLATION_CONFIGS,
)


def _synthetic_audio(sr=48000, duration=3.0, tone_freq=1000):
    """Generate synthetic audio for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pure tone (bird-like: tonal, low ZCR, low flatness)
    tone = 0.5 * np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
    # White noise (noise-like: broadband, high ZCR, high flatness)
    noise = np.random.randn(len(t)).astype(np.float32) * 0.3
    return tone, noise, sr


def test_spectral_flatness():
    """Test spectral flatness values for tonal vs noise signals."""
    print("\n── Test: Spectral Flatness ──")
    tone, noise, sr = _synthetic_audio()

    flat_tone = compute_spectral_flatness(tone, sr)
    flat_noise = compute_spectral_flatness(noise, sr)

    assert 0.0 <= flat_tone <= 1.0, f"Flatness out of range: {flat_tone}"
    assert 0.0 <= flat_noise <= 1.0, f"Flatness out of range: {flat_noise}"

    # Tonal signal should have lower flatness than noise
    assert flat_tone < flat_noise, (
        f"Tone flatness ({flat_tone:.4f}) should be < noise flatness ({flat_noise:.4f})"
    )

    # Edge cases
    empty_flat = compute_spectral_flatness(np.array([]), sr)
    assert empty_flat == 0.5, "Empty audio should return 0.5"

    zero_flat = compute_spectral_flatness(np.zeros(1000), sr)
    assert zero_flat == 0.5, "Zero audio should return 0.5"

    print(f"  Tone: {flat_tone:.4f}, Noise: {flat_noise:.4f}")
    print("✅ Spectral flatness: PASSED")


def test_zcr_normalized():
    """Test ZCR normalization for tonal vs noise signals."""
    print("\n── Test: ZCR Normalized ──")
    tone, noise, sr = _synthetic_audio()

    zcr_tone = compute_zcr_normalized(tone, sr)
    zcr_noise = compute_zcr_normalized(noise, sr)

    assert 0.0 <= zcr_tone <= 1.0, f"ZCR out of range: {zcr_tone}"
    assert 0.0 <= zcr_noise <= 1.0, f"ZCR out of range: {zcr_noise}"

    # Noise should have higher ZCR than pure tone
    assert zcr_tone < zcr_noise, (
        f"Tone ZCR ({zcr_tone:.4f}) should be < noise ZCR ({zcr_noise:.4f})"
    )

    # Edge cases
    empty_zcr = compute_zcr_normalized(np.array([]), sr)
    assert empty_zcr == 0.5, "Empty audio should return 0.5"

    print(f"  Tone: {zcr_tone:.4f}, Noise: {zcr_noise:.4f}")
    print("✅ ZCR normalized: PASSED")


def test_insect_periodicity():
    """Test insect periodicity for periodic vs non-periodic signals."""
    print("\n── Test: Insect Periodicity ──")
    sr = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create a highly periodic signal (insect-like: AM modulation at 15 Hz)
    carrier = np.sin(2 * np.pi * 5000 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 15 * t))  # 15 Hz AM
    insect_like = (carrier * modulator).astype(np.float32)

    # Non-periodic signal (random noise)
    random_signal = np.random.randn(len(t)).astype(np.float32)

    # Pure tone (no periodicity in envelope)
    pure_tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    ip_insect = compute_insect_periodicity(insect_like, sr)
    ip_random = compute_insect_periodicity(random_signal, sr)
    ip_tone = compute_insect_periodicity(pure_tone, sr)

    assert 0.0 <= ip_insect <= 1.0, f"Insect periodicity out of range: {ip_insect}"
    assert 0.0 <= ip_random <= 1.0, f"Random periodicity out of range: {ip_random}"
    assert 0.0 <= ip_tone <= 1.0, f"Tone periodicity out of range: {ip_tone}"

    # Edge cases
    empty_ip = compute_insect_periodicity(np.array([]), sr)
    assert empty_ip == 0.0, "Empty audio should return 0.0"

    print(f"  Insect-like: {ip_insect:.4f}")
    print(f"  Random noise: {ip_random:.4f}")
    print(f"  Pure tone: {ip_tone:.4f}")
    print("✅ Insect periodicity: PASSED")


def test_noise_segregator_scoring():
    """Test the noise scoring engine."""
    print("\n── Test: Noise Segregator Scoring ──")
    tone, noise, sr = _synthetic_audio()

    segregator = create_segregator("original")

    result_tone = segregator.classify_segment(tone, sr)
    result_noise = segregator.classify_segment(noise, sr)

    assert "features" in result_tone
    assert "noise_score" in result_tone
    assert "is_noise" in result_tone
    assert "label" in result_tone

    assert 0.0 <= result_tone["noise_score"] <= 1.0
    assert 0.0 <= result_noise["noise_score"] <= 1.0

    # Tonal signal should have lower noise score
    assert result_tone["noise_score"] < result_noise["noise_score"], (
        f"Tone score ({result_tone['noise_score']:.4f}) should be < "
        f"noise score ({result_noise['noise_score']:.4f})"
    )

    print(f"  Tone noise score: {result_tone['noise_score']:.4f} (is_noise={result_tone['is_noise']})")
    print(f"  Noise noise score: {result_noise['noise_score']:.4f} (is_noise={result_noise['is_noise']})")
    print("✅ Noise segregator scoring: PASSED")


def test_all_configs_loadable():
    """Test that all ablation configurations are loadable."""
    print("\n── Test: All Configs Loadable ──")
    configs = get_available_configs()
    assert len(configs) == 5, f"Expected 5 configs, got {len(configs)}"

    expected = {"original", "equal_weights", "no_flatness", "no_zcr", "no_insect_periodicity"}
    assert set(configs.keys()) == expected, f"Missing configs: {expected - set(configs.keys())}"

    for name in configs:
        seg = create_segregator(name)
        assert isinstance(seg, NoiseSegregatorV2)
        assert seg.config_name == name
        assert sum(seg.weights.values()) <= 1.001  # Allow floating point

        # Quick smoke test
        audio = np.random.randn(int(48000 * 3)).astype(np.float32)
        result = seg.classify_segment(audio, 48000)
        assert result["label"] in (0, 1)

    print(f"  All {len(configs)} configs loaded and functional")
    print("✅ All configs loadable: PASSED")


def test_weight_zeroing():
    """Test that setting a weight to 0 actually ignores that feature."""
    print("\n── Test: Weight Zeroing ──")
    sr = 48000
    audio = np.random.randn(int(sr * 3)).astype(np.float32) * 0.5

    # Config with only spectral flatness
    seg_flatonly = NoiseSegregatorV2(
        weights={"spectral_flatness": 1.0, "zcr": 0.0, "insect_periodicity": 0.0},
        threshold=0.5,
        config_name="test_flat_only",
    )

    features = seg_flatonly.extract_features(audio, sr)
    score = seg_flatonly.compute_noise_score(features)

    # Score should equal the spectral flatness value (weight=1.0, others=0.0)
    expected = features["spectral_flatness"]
    assert abs(score - expected) < 1e-6, (
        f"Score ({score}) != expected flatness-only ({expected})"
    )

    print(f"  Flatness-only score: {score:.4f} (expected: {expected:.4f})")
    print("✅ Weight zeroing: PASSED")


if __name__ == "__main__":
    test_spectral_flatness()
    test_zcr_normalized()
    test_insect_periodicity()
    test_noise_segregator_scoring()
    test_all_configs_loadable()
    test_weight_zeroing()
    print("\n✅ All noise segregation tests passed!")
