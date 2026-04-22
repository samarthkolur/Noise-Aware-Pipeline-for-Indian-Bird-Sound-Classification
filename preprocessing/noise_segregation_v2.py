"""
noise_segregation_v2.py — Rule-based noise vs bird-like scoring (V2).

Each 3 s segment is split into six 0.5 s subframes. Per subframe we compute
features, a noise score S in [0, 1], and vote noise if S >= vote_threshold.
The segment label is decided by majority vote across subframes.

This module does not replace BirdNET; it only routes or labels segments before
embedding when enabled in config (see preprocessing.preprocessing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise ImportError("noise_segregation_v2 requires librosa") from e

Label = Literal["bird", "noise"]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _norm_ratio(x: float, ref: float) -> float:
    """Map x/ref into [0, 1] (plan: norm(feature, reference_scale))."""
    if ref <= 0:
        return 0.0
    return _clip01(x / ref)


@dataclass
class V2SubframeResult:
    """Per 0.5 s subframe: features summary and noise score."""

    rms_db: float
    zcr: float
    flatness: float
    centroid_mean_hz: float
    centroid_std_hz: float
    autocorr_peak: float
    noise_score: float
    noise_vote: bool


@dataclass
class V2SegmentResult:
    """Aggregated V2 decision for one 3 s segment."""

    label: Label
    mean_score: float
    noise_votes: int
    total_subframes: int
    subframes: List[V2SubframeResult]


class NoiseSegregationV2:
    """Noise score S and majority voting over six 0.5 s subframes."""

    def __init__(
        self,
        sample_rate: int = 48_000,
        segment_duration_s: float = 3.0,
        subframe_duration_s: float = 0.5,
        vote_threshold: float = 0.5,
        majority_tie: str = "bird",
        target_rms_db: float = -20.0,
        # Score weights (plan defaults)
        w_zcr: float = 0.25,
        w_flat: float = 0.30,
        w_centroid_flag: float = 0.15,
        w_centroid_std: float = 0.15,
        w_insect: float = 0.15,
        zcr_ref: float = 0.30,
        flatness_ref: float = 1.0,
        centroid_std_ref: float = 2500.0,
        # Heuristic flags (Hz); tunable via config
        centroid_low_hz: float = 400.0,
        centroid_high_hz: float = 7000.0,
        insect_lag_min_hz: float = 200.0,
        insect_lag_max_hz: float = 2000.0,
        insect_peak_thresh: float = 0.35,
    ) -> None:
        self.sample_rate = sample_rate
        self.segment_duration_s = segment_duration_s
        self.subframe_duration_s = subframe_duration_s
        self.vote_threshold = vote_threshold
        self.majority_tie = majority_tie
        self.target_rms_db = target_rms_db
        self.w_zcr = w_zcr
        self.w_flat = w_flat
        self.w_centroid_flag = w_centroid_flag
        self.w_centroid_std = w_centroid_std
        self.w_insect = w_insect
        self.zcr_ref = zcr_ref
        self.flatness_ref = flatness_ref
        self.centroid_std_ref = centroid_std_ref
        self.centroid_low_hz = centroid_low_hz
        self.centroid_high_hz = centroid_high_hz
        self.insect_lag_min_hz = insect_lag_min_hz
        self.insect_lag_max_hz = insect_lag_max_hz
        self.insect_peak_thresh = insect_peak_thresh

        self.subframe_samples = int(sample_rate * subframe_duration_s)
        self.segment_samples = int(sample_rate * segment_duration_s)

    @classmethod
    def from_config(cls, cfg: dict) -> "NoiseSegregationV2":
        audio = cfg.get("audio", {})
        ns = cfg.get("noise_segregation", {})
        v2 = ns.get("v2", {})
        return cls(
            sample_rate=int(audio.get("sample_rate", 48_000)),
            segment_duration_s=float(audio.get("segment_duration_s", 3.0)),
            subframe_duration_s=float(v2.get("subframe_duration_s", 0.5)),
            vote_threshold=float(v2.get("vote_threshold", 0.5)),
            majority_tie=str(v2.get("majority_tie", "bird")),
            target_rms_db=float(v2.get("target_rms_db", -20.0)),
            w_zcr=float(v2.get("w_zcr", 0.25)),
            w_flat=float(v2.get("w_flat", 0.30)),
            w_centroid_flag=float(v2.get("w_centroid_flag", 0.15)),
            w_centroid_std=float(v2.get("w_centroid_std", 0.15)),
            w_insect=float(v2.get("w_insect", 0.15)),
            zcr_ref=float(v2.get("zcr_ref", 0.30)),
            flatness_ref=float(v2.get("flatness_ref", 1.0)),
            centroid_std_ref=float(v2.get("centroid_std_ref", 2500.0)),
            centroid_low_hz=float(v2.get("centroid_low_hz", 400.0)),
            centroid_high_hz=float(v2.get("centroid_high_hz", 7000.0)),
            insect_lag_min_hz=float(v2.get("insect_lag_min_hz", 200.0)),
            insect_lag_max_hz=float(v2.get("insect_lag_max_hz", 2000.0)),
            insect_peak_thresh=float(v2.get("insect_peak_thresh", 0.35)),
        )

    def normalize_rms(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform to a target RMS level in dB (float32 mono)."""
        x = np.asarray(waveform, dtype=np.float64).flatten()
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-20))
        if rms < 1e-10:
            return x.astype(np.float32)
        current_db = 20.0 * np.log10(rms)
        gain_db = self.target_rms_db - current_db
        gain = 10.0 ** (gain_db / 20.0)
        y = np.clip(x * gain, -1.0, 1.0)
        return y.astype(np.float32)

    def _rms_db(self, x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-20))
        if rms < 1e-10:
            return -100.0
        return float(20.0 * np.log10(rms))

    def _zcr(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).flatten()
        if len(x) < 2:
            return 0.0
        s = np.sign(x)
        s[s == 0] = 1
        crossings = np.sum(np.abs(np.diff(s))) / 2.0
        return float(crossings / max(len(x) - 1, 1))

    def _spectral_flatness(self, x: np.ndarray) -> float:
        """Geometric mean / arithmetic mean of power spectrum (librosa)."""
        x = np.asarray(x, dtype=np.float32).flatten()
        if len(x) < 32:
            return 0.0
        # librosa API: pass y= only (sr not accepted in all versions)
        flat = librosa.feature.spectral_flatness(y=x)
        return float(np.mean(flat))

    def _centroid_mean_std_hz(self, x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).flatten()
        if len(x) < 32:
            return 0.0, 0.0
        cent = librosa.feature.spectral_centroid(
            y=x, sr=self.sample_rate, n_fft=512, hop_length=128
        )
        c = cent.flatten()
        return float(np.mean(c)), float(np.std(c))

    def _autocorr_peak(self, x: np.ndarray) -> float:
        """Largest normalized autocorrelation peak in the insect lag band.

        Uses only lags ``[lag_min, lag_max]`` via dot products — **not** full
        ``np.correlate`` — so 0.5 s @ 48 kHz stays fast (full correlate was O(n²)).
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        n = len(x)
        if n < 32:
            return 0.0
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-12:
            return 0.0
        ac0 = float(np.dot(x, x)) + 1e-12
        lag_min = max(1, int(self.sample_rate / self.insect_lag_max_hz))
        lag_max = min(n - 1, int(self.sample_rate / self.insect_lag_min_hz))
        if lag_max <= lag_min:
            return 0.0
        best = 0.0
        for k in range(lag_min, lag_max + 1):
            c = float(np.dot(x[:-k], x[k:])) / ac0
            if c > best:
                best = c
        return best

    def _centroid_flag(self, centroid_mean_hz: float) -> float:
        """1.0 if centroid suggests broadband / non-tonal extremes."""
        if (
            centroid_mean_hz < self.centroid_low_hz
            or centroid_mean_hz > self.centroid_high_hz
        ):
            return 1.0
        return 0.0

    def _insect_flag(self, autocorr_peak: float) -> float:
        """1.0 if periodicity in insect-like band (proxy via autocorr peak)."""
        return 1.0 if autocorr_peak >= self.insect_peak_thresh else 0.0

    def noise_score_subframe(self, sub: np.ndarray) -> V2SubframeResult:
        """Compute S for one subframe waveform (mono, any length)."""
        zcr = self._zcr(sub)
        flat = self._spectral_flatness(sub)
        c_mean, c_std = self._centroid_mean_std_hz(sub)
        ac_pk = self._autocorr_peak(sub)
        rms = self._rms_db(sub)

        s = (
            self.w_zcr * _norm_ratio(zcr, self.zcr_ref)
            + self.w_flat * _norm_ratio(flat, self.flatness_ref)
            + self.w_centroid_flag * self._centroid_flag(c_mean)
            + self.w_centroid_std * _norm_ratio(c_std, self.centroid_std_ref)
            + self.w_insect * self._insect_flag(ac_pk)
        )
        s = _clip01(s)
        vote = s >= self.vote_threshold
        return V2SubframeResult(
            rms_db=rms,
            zcr=zcr,
            flatness=flat,
            centroid_mean_hz=c_mean,
            centroid_std_hz=c_std,
            autocorr_peak=ac_pk,
            noise_score=s,
            noise_vote=vote,
        )

    def classify_segment(self, waveform_mono: np.ndarray) -> V2SegmentResult:
        """Classify one 3 s mono segment (length should match segment_samples)."""
        x = self.normalize_rms(np.asarray(waveform_mono, dtype=np.float32).flatten())
        if x.size != self.segment_samples:
            if x.size < self.segment_samples:
                x = np.pad(x, (0, self.segment_samples - x.size))
            else:
                x = x[: self.segment_samples]

        subframes: List[V2SubframeResult] = []
        pos = 0
        while pos + self.subframe_samples <= x.size:
            chunk = x[pos : pos + self.subframe_samples]
            subframes.append(self.noise_score_subframe(chunk))
            pos += self.subframe_samples

        if not subframes:
            # Degenerate: treat as bird
            return V2SegmentResult(
                label="bird",
                mean_score=0.0,
                noise_votes=0,
                total_subframes=0,
                subframes=[],
            )

        noise_votes = sum(1 for s in subframes if s.noise_vote)
        n = len(subframes)
        mean_score = float(np.mean([s.noise_score for s in subframes]))

        if noise_votes > n // 2:
            label: Label = "noise"
        elif noise_votes < n // 2:
            label = "bird"
        else:
            label = "noise" if self.majority_tie == "noise" else "bird"

        return V2SegmentResult(
            label=label,
            mean_score=mean_score,
            noise_votes=noise_votes,
            total_subframes=n,
            subframes=subframes,
        )
