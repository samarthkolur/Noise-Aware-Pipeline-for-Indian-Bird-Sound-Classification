"""Lightweight audio heuristics for error tagging (insect / wind / faint bird).

Thresholds are collected in ``HEURISTIC_DEFAULTS`` and can be overridden
via ``tag_segment(..., overrides={...})``.  The research suite loads these
from ``config.yaml → research.heuristics`` for fully config-driven behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None


# ── Configurable defaults ──────────────────────────────────
# Each key maps to a threshold used in tag_segment(). The research suite
# reads these from config.yaml → research.heuristics and passes as overrides.
HEURISTIC_DEFAULTS: Dict[str, float] = {
    "zcr_insect_thresh": 0.25,
    "flatness_insect_thresh": 0.15,
    "low_freq_wind_ratio_thresh": 0.55,
    "faint_rms_db_thresh": -35.0,
    "periodicity_lag_min": 200,
    "periodicity_lag_max": 3000,
    "periodicity_peak_ratio_thresh": 0.15,
    "faint_periodic_rms_db_thresh": -32.0,
}


def _load_mono(path: Path, max_sec: float = 3.5) -> Optional[tuple[np.ndarray, int]]:
    path = Path(path)
    if not path.is_file():
        return None
    if librosa is not None:
        y, sr = librosa.load(str(path), sr=None, mono=True, duration=max_sec)
        return y.astype(np.float32), int(sr)
    if sf is not None:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        max_len = int(max_sec * sr)
        if len(y) > max_len:
            y = y[:max_len]
        return y.astype(np.float32), int(sr)
    return None


def tag_segment(
    path: str | Path,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Return heuristic scores; higher flatness/zcr → insect-like; low_freq_ratio for wind.

    Args:
        path: Path to a WAV segment.
        overrides: Optional dict of threshold overrides (keys from HEURISTIC_DEFAULTS).
    """
    # Merge defaults with any overrides
    params = {**HEURISTIC_DEFAULTS}
    if overrides:
        params.update(overrides)

    out: Dict[str, Optional[float]] = {
        "rms_db": None,
        "zcr": None,
        "spectral_flatness_mean": None,
        "low_freq_energy_ratio": None,
        "tags": [],
    }
    loaded = _load_mono(Path(path))
    if loaded is None:
        return out
    y, sr = loaded
    if len(y) < 16:
        return out

    eps = 1e-12
    rms = float(np.sqrt(np.mean(y**2)))
    out["rms_db"] = float(20.0 * np.log10(rms + eps))

    # Zero-crossing rate
    zc = np.mean(np.abs(np.diff(np.sign(y)))) / 2.0
    out["zcr"] = float(zc)

    if librosa is not None:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
        flat = librosa.feature.spectral_flatness(S=S)[0]
        out["spectral_flatness_mean"] = float(np.mean(flat))

        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        total_e = float(np.sum(S) + eps)
        low_mask = freqs < 500.0
        low_e = float(np.sum(S[low_mask, :]))
        out["low_freq_energy_ratio"] = low_e / total_e
    else:
        # FFT-based flatness proxy
        spec = np.abs(np.fft.rfft(y, n=min(1024, len(y)))) ** 2
        flat = float(np.exp(np.mean(np.log(spec + eps))) / (np.mean(spec) + eps))
        out["spectral_flatness_mean"] = flat
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        total_e = float(np.sum(spec) + eps)
        low_e = float(np.sum(spec[freqs < 500.0]))
        out["low_freq_energy_ratio"] = low_e / total_e

    tags = []
    if out["zcr"] is not None and out["spectral_flatness_mean"] is not None:
        if (
            out["zcr"] > params["zcr_insect_thresh"]
            and out["spectral_flatness_mean"] > params["flatness_insect_thresh"]
        ):
            tags.append("insect_like")
    if (
        out["low_freq_energy_ratio"] is not None
        and out["low_freq_energy_ratio"] > params["low_freq_wind_ratio_thresh"]
    ):
        tags.append("wind_like")
    if out["rms_db"] is not None and out["rms_db"] < params["faint_rms_db_thresh"]:
        tags.append("faint_rms")

    # Crude periodicity proxy (faint tonal / pulsed calls vs. flat noise)
    if len(y) > 8000:
        yc = y - y.mean()
        full_ac = np.correlate(yc, yc, mode="full")
        mid = len(yc) - 1
        lag_min_idx = int(params["periodicity_lag_min"])
        lag_max_idx = int(params["periodicity_lag_max"])
        lag_win = full_ac[mid + lag_min_idx : mid + lag_max_idx]
        peak = float(np.max(np.abs(lag_win)) / (np.abs(full_ac[mid]) + eps))
        out["periodicity_peak_ratio"] = peak
        if (
            out["rms_db"] is not None
            and out["rms_db"] < params["faint_periodic_rms_db_thresh"]
            and peak > params["periodicity_peak_ratio_thresh"]
        ):
            tags.append("faint_periodic_candidate")
    out["tags"] = tags
    return out
