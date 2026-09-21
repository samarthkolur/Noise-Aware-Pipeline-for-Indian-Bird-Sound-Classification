"""Audio I/O, resampling, segmentation, and RMS silence gating (design.md §6.1)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from librosa import resample as _librosa_resample

SILENCE_FLOOR_DB = -120.0  # dBFS assigned to a literally-zero signal


def load_audio(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32, resampled to target_sr."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    return _librosa_resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def segment_audio(audio: np.ndarray, sr: int, segment_length_s: float) -> list[np.ndarray]:
    """Split audio into non-overlapping fixed-length segments.

    The final partial segment (shorter than segment_length_s) is dropped, matching
    the paper's fixed-length 3 s segmentation.
    """
    segment_len = int(round(segment_length_s * sr))
    if segment_len <= 0:
        raise ValueError("segment_length_s * sr must be positive")
    n_segments = len(audio) // segment_len
    return [audio[i * segment_len : (i + 1) * segment_len] for i in range(n_segments)]


def rms_db(audio: np.ndarray) -> float:
    """RMS level of a signal in dBFS (full-scale = 1.0)."""
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms <= 0.0:
        return SILENCE_FLOOR_DB
    return 20.0 * float(np.log10(rms))


def is_silent(audio: np.ndarray, silence_db_threshold: float) -> bool:
    """True if the segment's RMS level is below the silence threshold (dBFS)."""
    return rms_db(audio) < silence_db_threshold


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write mono float32 audio to a WAV file."""
    sf.write(path, audio, sr, subtype="PCM_16")
