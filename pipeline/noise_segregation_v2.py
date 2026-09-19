"""
Noise Segregation V2: Multi-Feature Weighted Noise Classification

A refined noise segregation module that scores audio segments using a
weighted combination of hand-crafted acoustic features. Each feature
targets a specific noise signature:

    1. Spectral Flatness     — Broadband noise detector (traffic, wind, rain)
    2. Zero-Crossing Rate    — Percussive/impulsive noise detector (clicks, pops)
    3. Insect Periodicity    — Periodic insect buzz/chirp detector

The final noise score is:
    noise_score = w_flatness * flatness + w_zcr * zcr_norm + w_insect * insect_flag

Segments scoring above the threshold are classified as noise regardless
of BirdNET confidence (i.e., this acts as a false-positive override).

The weights are manually tuned based on acoustic domain knowledge:
    - Spectral flatness is the strongest noise indicator (weight 0.5)
    - ZCR captures transient noise missed by flatness (weight 0.3)
    - Insect periodicity catches a common tropical confusion (weight 0.2)

This module is designed for ablation: each feature can be independently
disabled to quantify its contribution to overall noise rejection.
"""

import numpy as np
import librosa
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Default Weights (Manually Tuned) ───────────────────────────────────────

ORIGINAL_WEIGHTS = {
    "spectral_flatness": 0.5,
    "zcr": 0.3,
    "insect_periodicity": 0.2,
}

NOISE_SCORE_THRESHOLD = 0.45  # Segments above this → noise override

# ZCR normalization ceiling — empirically derived from 48 kHz bird recordings.
# Actual ZCR for bird audio clusters at 0.33-0.60 (raw librosa values),
# so we normalize with a ceiling of 0.5 to center the distribution.
ZCR_NORMALIZATION_CEILING = 0.5

# Insect periodicity detection requires BOTH:
#   (a) strong autocorrelation peak (> 0.6) AND
#   (b) at least 2 consistent peaks to confirm periodic structure.
# Single peaks can arise from any quasi-periodic audio (bird trills, etc.)
INSECT_AUTOCORR_PEAK_THRESHOLD = 0.6
INSECT_MIN_CONSISTENT_PEAKS = 2


# ─── Ablation Weight Configurations ─────────────────────────────────────────
# Note: thresholds are calibrated per-config to ensure viable class balance.
# When a feature is removed, the remaining features' dynamic range changes,
# so the threshold must be adjusted to maintain a comparable operating point.

ABLATION_CONFIGS = {
    "original": {
        "weights": {"spectral_flatness": 0.5, "zcr": 0.3, "insect_periodicity": 0.2},
        "threshold": 0.45,
        "description": "Original manually-tuned weights",
    },
    "equal_weights": {
        "weights": {"spectral_flatness": 1/3, "zcr": 1/3, "insect_periodicity": 1/3},
        "threshold": 0.45,
        "description": "Equal weights across all features",
    },
    "no_flatness": {
        "weights": {"spectral_flatness": 0.0, "zcr": 0.6, "insect_periodicity": 0.4},
        "threshold": 0.50,
        "description": "Without spectral flatness (redistributed to remaining)",
    },
    "no_zcr": {
        "weights": {"spectral_flatness": 0.7, "zcr": 0.0, "insect_periodicity": 0.3},
        "threshold": 0.40,
        "description": "Without ZCR (redistributed to remaining)",
    },
    "no_insect_periodicity": {
        "weights": {"spectral_flatness": 0.625, "zcr": 0.375, "insect_periodicity": 0.0},
        "threshold": 0.45,
        "description": "Without insect periodicity flag (redistributed to remaining)",
    },
}


# ─── Feature Extractors ─────────────────────────────────────────────────────

def compute_spectral_flatness(y: np.ndarray, sr: int) -> float:
    """
    Compute mean spectral flatness of an audio segment.

    Spectral flatness = geometric mean / arithmetic mean of the power spectrum.
    Values near 1.0 indicate broadband noise (flat spectrum).
    Values near 0.0 indicate tonal content (peaked spectrum, e.g., bird call).

    Args:
        y: Audio time-series array.
        sr: Sample rate.

    Returns:
        Mean spectral flatness in [0, 1]. Returns 0.5 on failure.
    """
    try:
        if len(y) == 0 or np.all(y == 0):
            return 0.5
        flatness = librosa.feature.spectral_flatness(y=y)
        return float(np.mean(flatness))
    except Exception:
        return 0.5


def compute_zcr_normalized(y: np.ndarray, sr: int) -> float:
    """
    Compute normalized zero-crossing rate.

    ZCR counts sign changes per frame, normalized to [0, 1].
    High ZCR → noisy/percussive signals (clicks, static, broadband noise).
    Low ZCR → tonal signals (bird calls, whistles).

    Normalization: At 48 kHz, raw ZCR for bird recordings clusters in
    0.10-0.30 (tonal) vs 0.30-0.50 (noisy). We normalize with a ceiling
    of ZCR_NORMALIZATION_CEILING (0.5) to map the full discriminative
    range into [0, 1].

    Args:
        y: Audio time-series array.
        sr: Sample rate.

    Returns:
        Normalized ZCR in [0, 1]. Returns 0.5 on failure.
    """
    try:
        if len(y) == 0 or np.all(y == 0):
            return 0.5
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
        mean_zcr = float(np.mean(zcr))
        # Normalize using the empirically-derived ceiling
        normalized = min(mean_zcr / ZCR_NORMALIZATION_CEILING, 1.0)
        return normalized
    except Exception:
        return 0.5


def compute_insect_periodicity(y: np.ndarray, sr: int) -> float:
    """
    Detect periodic insect-like patterns via autocorrelation analysis.

    Insects (cicadas, crickets) produce highly periodic signals in the
    5-30 Hz modulation range. This manifests as strong, *repeating*
    autocorrelation peaks in the amplitude envelope.

    Method:
        1. Compute amplitude envelope (RMS energy per frame)
        2. Compute normalized autocorrelation of the envelope
        3. Search for peaks in the insect modulation frequency range
        4. Require MULTIPLE consistent peaks (not just one) to confirm
           periodic structure — this avoids false positives from bird
           trills or single repeated call elements

    The flag is continuous in [0, 1]:
        0.0 = no periodic insect-like pattern detected
        1.0 = very strong, consistent periodic pattern (likely insect)

    Args:
        y: Audio time-series array.
        sr: Sample rate.

    Returns:
        Insect periodicity score in [0, 1]. Returns 0.0 on failure.
    """
    try:
        if len(y) == 0 or np.all(y == 0):
            return 0.0

        # Compute amplitude envelope using RMS
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

        if len(rms) < 20:
            return 0.0

        # Remove DC component and normalize variance
        rms = rms - np.mean(rms)
        rms_std = np.std(rms)
        if rms_std < 1e-10:
            return 0.0
        rms = rms / rms_std

        # Autocorrelation of the normalized envelope
        autocorr = np.correlate(rms, rms, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Take positive lags only

        if len(autocorr) < 2 or autocorr[0] == 0:
            return 0.0

        # Normalize by lag-0 (energy)
        autocorr = autocorr / autocorr[0]

        # Focus on insect modulation frequencies: 5-30 Hz
        # (narrower range than before to reduce false positives)
        frame_rate = sr / hop_length
        min_lag = max(2, int(frame_rate / 30))  # 30 Hz modulation
        max_lag = min(len(autocorr) - 1, int(frame_rate / 5))  # 5 Hz modulation

        if min_lag >= max_lag or max_lag >= len(autocorr):
            return 0.0

        search_region = autocorr[min_lag:max_lag + 1]

        # Find peaks above the strict threshold
        peak_threshold = INSECT_AUTOCORR_PEAK_THRESHOLD
        peaks_above = np.where(search_region > peak_threshold)[0]

        if len(peaks_above) == 0:
            return 0.0

        # Count distinct peaks (separated by at least 2 lags)
        n_distinct_peaks = 1
        last_peak_pos = peaks_above[0]
        for pos in peaks_above[1:]:
            if pos - last_peak_pos >= 2:
                n_distinct_peaks += 1
                last_peak_pos = pos

        # Require multiple consistent peaks to confirm insect periodicity
        if n_distinct_peaks < INSECT_MIN_CONSISTENT_PEAKS:
            return 0.0

        # Score based on peak strength and count
        max_peak = float(np.max(search_region))
        peak_count_factor = min(n_distinct_peaks / 4.0, 1.0)  # Saturates at 4 peaks
        periodicity_score = max_peak * peak_count_factor

        return min(periodicity_score, 1.0)

    except Exception:
        return 0.0


# ─── Noise Scoring Engine ───────────────────────────────────────────────────

class NoiseSegregatorV2:
    """
    Multi-feature noise segregation engine with configurable weights.

    Computes a weighted noise score from three acoustic features and
    classifies segments as noise if the score exceeds a threshold.

    Attributes:
        weights: Dict mapping feature names to weights (must sum to 1.0).
        threshold: Score above which a segment is classified as noise.
    """

    def __init__(
        self,
        weights: dict = None,
        threshold: float = None,
        config_name: str = "original",
    ):
        """
        Initialize the noise segregator.

        Args:
            weights: Feature weight dict. If None, uses the named config.
            threshold: Noise score threshold.
            config_name: Name of ablation config to use if weights is None.
        """
        if weights is not None:
            self.weights = weights
            self.threshold = threshold or NOISE_SCORE_THRESHOLD
        else:
            cfg = ABLATION_CONFIGS.get(config_name, ABLATION_CONFIGS["original"])
            self.weights = cfg["weights"]
            self.threshold = cfg.get("threshold", NOISE_SCORE_THRESHOLD)

        self.config_name = config_name

    def extract_features(self, y: np.ndarray, sr: int) -> dict:
        """
        Extract all three acoustic features from an audio segment.

        Args:
            y: Audio time-series array.
            sr: Sample rate.

        Returns:
            Dict with keys: spectral_flatness, zcr, insect_periodicity.
        """
        return {
            "spectral_flatness": compute_spectral_flatness(y, sr),
            "zcr": compute_zcr_normalized(y, sr),
            "insect_periodicity": compute_insect_periodicity(y, sr),
        }

    def compute_noise_score(self, features: dict) -> float:
        """
        Compute weighted noise score from extracted features.

        Args:
            features: Dict with feature values.

        Returns:
            Weighted noise score in [0, 1].
        """
        score = 0.0
        for feat_name, weight in self.weights.items():
            feat_val = features.get(feat_name, 0.0)
            score += weight * feat_val
        return score

    def classify_segment(self, y: np.ndarray, sr: int) -> dict:
        """
        Classify a single audio segment as bird or noise.

        Args:
            y: Audio time-series array.
            sr: Sample rate.

        Returns:
            Dict with keys: features, noise_score, is_noise, label.
        """
        features = self.extract_features(y, sr)
        noise_score = self.compute_noise_score(features)
        is_noise = noise_score >= self.threshold

        return {
            "features": features,
            "noise_score": noise_score,
            "is_noise": is_noise,
            "label": config.NOISE_LABEL if is_noise else config.BIRD_LABEL,
        }

    def classify_batch_from_paths(
        self,
        file_paths: np.ndarray,
        birdnet_confidences: np.ndarray,
    ) -> dict:
        """
        Classify a batch of audio segments using the noise segregation scoring.

        For segments where BirdNET is confident it's a bird (conf >= threshold),
        the noise segregator acts as a false-positive override: if the noise
        score is high, it flips the label to noise.

        For segments where BirdNET is not confident, standard pseudo-labeling
        applies.

        Args:
            file_paths: Array of audio file paths.
            birdnet_confidences: Array of BirdNET confidence scores.

        Returns:
            Dict with keys: labels, noise_scores, features_matrix,
                           override_mask (segments where noise score overrode BirdNET).
        """
        n = len(file_paths)
        labels = np.full(n, -1, dtype=np.int64)  # -1 = uncertain
        noise_scores = np.zeros(n, dtype=np.float32)
        override_mask = np.zeros(n, dtype=bool)

        all_features = {
            "spectral_flatness": np.zeros(n, dtype=np.float32),
            "zcr": np.zeros(n, dtype=np.float32),
            "insect_periodicity": np.zeros(n, dtype=np.float32),
        }

        for i in range(n):
            conf = birdnet_confidences[i]
            fpath = str(file_paths[i])

            try:
                y, sr = librosa.load(fpath, sr=config.TARGET_SR)
            except Exception:
                # Can't load audio — use BirdNET confidence alone
                if conf >= config.BIRD_CONFIDENCE_HIGH:
                    labels[i] = config.BIRD_LABEL
                elif conf < config.BIRD_CONFIDENCE_LOW:
                    labels[i] = config.NOISE_LABEL
                continue

            features = self.extract_features(y, sr)
            noise_score = self.compute_noise_score(features)
            noise_scores[i] = noise_score

            for feat_name in all_features:
                all_features[feat_name][i] = features.get(feat_name, 0.0)

            if conf >= config.BIRD_CONFIDENCE_HIGH:
                if noise_score >= self.threshold:
                    # Override: BirdNET says bird, but noise score says noise
                    labels[i] = config.NOISE_LABEL
                    override_mask[i] = True
                else:
                    labels[i] = config.BIRD_LABEL
            elif conf < config.BIRD_CONFIDENCE_LOW:
                labels[i] = config.NOISE_LABEL
            else:
                # Uncertain — leave as -1
                labels[i] = -1

            if (i + 1) % 200 == 0:
                print(f"    [NoiseSegV2] Processed {i+1}/{n} segments...")

        n_bird = int((labels == config.BIRD_LABEL).sum())
        n_noise = int((labels == config.NOISE_LABEL).sum())
        n_uncertain = int((labels == -1).sum())
        n_overrides = int(override_mask.sum())

        print(f"  [NoiseSegV2] Config: {self.config_name}")
        print(f"  [NoiseSegV2] Labels: {n_bird} bird, {n_noise} noise, "
              f"{n_uncertain} uncertain")
        print(f"  [NoiseSegV2] Overrides (BirdNET→noise): {n_overrides}")

        return {
            "labels": labels,
            "noise_scores": noise_scores,
            "features": all_features,
            "override_mask": override_mask,
        }

    def __repr__(self):
        return (
            f"NoiseSegregatorV2(config='{self.config_name}', "
            f"weights={self.weights}, threshold={self.threshold})"
        )


# ─── Convenience Functions ──────────────────────────────────────────────────

def create_segregator(config_name: str = "original") -> NoiseSegregatorV2:
    """
    Factory function to create a noise segregator with a named configuration.

    Args:
        config_name: One of the keys in ABLATION_CONFIGS.

    Returns:
        NoiseSegregatorV2 instance.
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(ABLATION_CONFIGS.keys())}"
        )
    cfg = ABLATION_CONFIGS[config_name]
    return NoiseSegregatorV2(
        weights=cfg["weights"],
        threshold=cfg["threshold"],
        config_name=config_name,
    )


def get_available_configs() -> dict:
    """Return all available ablation configurations."""
    return ABLATION_CONFIGS.copy()


if __name__ == "__main__":
    print("Noise Segregation V2 Module")
    print(f"Available configs: {list(ABLATION_CONFIGS.keys())}")
    for name, cfg in ABLATION_CONFIGS.items():
        print(f"  {name}: {cfg['description']}")
        print(f"    weights: {cfg['weights']}")
