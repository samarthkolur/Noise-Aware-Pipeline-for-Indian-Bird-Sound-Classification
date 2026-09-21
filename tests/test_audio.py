import numpy as np

from pipeline.audio import is_silent, resample_audio, rms_db, segment_audio


def test_resample_audio_changes_length():
    sr_orig, sr_target = 44100, 48000
    audio = np.zeros(sr_orig, dtype=np.float32)
    resampled = resample_audio(audio, sr_orig, sr_target)
    assert abs(len(resampled) - sr_target) < 10


def test_resample_audio_noop_when_same_rate():
    audio = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    resampled = resample_audio(audio, 48000, 48000)
    np.testing.assert_array_equal(audio, resampled)


def test_segment_audio_non_overlapping_fixed_length():
    sr = 48000
    audio = np.arange(sr * 7, dtype=np.float32)  # 7 seconds
    segments = segment_audio(audio, sr, segment_length_s=3.0)
    assert len(segments) == 2  # trailing 1s partial segment dropped
    assert all(len(s) == sr * 3 for s in segments)


def test_rms_db_silence_is_very_negative():
    silence = np.zeros(1000, dtype=np.float32)
    assert rms_db(silence) < -60.0


def test_rms_db_full_scale_sine_near_zero_dbfs():
    t = np.linspace(0, 1, 48000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    level = rms_db(sine)
    assert -4.0 < level < -2.0  # RMS of a unit sine is ~-3 dBFS


def test_is_silent_true_below_threshold():
    silence = np.zeros(1000, dtype=np.float32)
    assert is_silent(silence, silence_db_threshold=-40.0) is True


def test_is_silent_false_above_threshold():
    t = np.linspace(0, 1, 48000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert is_silent(sine, silence_db_threshold=-40.0) is False
