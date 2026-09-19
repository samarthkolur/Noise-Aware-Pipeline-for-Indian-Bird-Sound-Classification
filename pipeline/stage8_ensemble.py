"""
Stage 8: Ensemble Decision & Final Output

Combine all pipeline signals into a final bird/noise decision using
weighted confidence voting. Then output the clean dataset.

Signals:
    1. Binary classifier probability (RF or MLP)
    2. OOD filter decision (in-distribution = bird)
    3. Post-processing verdict (spectral + temporal)
    4. BirdNET raw confidence (tie-breaker)

Weighted voting produces a calibrated confidence score. Segments above
the ensemble threshold are classified as bird and copied to the clean
output dataset.

Output:
    data/noise_aware_dataset/bird/   — clean bird segments
    data/noise_aware_dataset/noise/  — rejected noise segments
"""

import os
import shutil
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class WeightedEnsembleDecider:
    """
    Confidence-weighted ensemble decision maker.

    Unlike the old threshold-counting approach, this uses weighted voting
    to produce a calibrated confidence score for each segment.
    """

    def __init__(
        self,
        weights: dict = None,
        threshold: float = None,
        birdnet_threshold: float = None,
    ):
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        self.threshold = threshold or config.ENSEMBLE_THRESHOLD
        self.birdnet_threshold = birdnet_threshold or config.ENSEMBLE_BIRDNET_THRESHOLD

    def decide_single(self, signals: dict) -> tuple:
        """
        Make a weighted decision for a single segment.

        Args:
            signals: Dictionary with keys:
                - "classifier_prob": float (0–1)
                - "ood_is_bird": int (0 or 1)
                - "postprocessing_label": int (0 or 1)
                - "birdnet_confidence": float (0–1)

        Returns:
            Tuple of (label: int, confidence: float, detail: dict).
        """
        clf_prob = signals.get("classifier_prob", 0.0)
        ood = float(signals.get("ood_is_bird", 1))
        pp_label = float(signals.get("postprocessing_label", 1))
        bn_conf = signals.get("birdnet_confidence", 0.0)

        # Normalize BirdNET confidence to a 0/1 signal
        bn_signal = 1.0 if bn_conf >= self.birdnet_threshold else 0.0

        # Weighted score
        w = self.weights
        weighted_score = (
            w.get("classifier", 0.35) * clf_prob +
            w.get("ood", 0.25) * ood +
            w.get("postprocessing", 0.25) * pp_label +
            w.get("birdnet_raw", 0.15) * bn_signal
        )

        is_bird = int(weighted_score >= self.threshold)

        detail = {
            "classifier_prob": clf_prob,
            "ood_is_bird": int(ood),
            "postprocessing_label": int(pp_label),
            "birdnet_signal": bn_signal,
            "weighted_score": float(weighted_score),
        }

        return is_bird, float(weighted_score), detail

    def decide_batch(self, signals_batch: dict) -> tuple:
        """
        Make decisions for a batch of segments.

        Args:
            signals_batch: Dictionary where each value is a numpy array:
                - "classifier_prob": (N,) array
                - "ood_is_bird": (N,) array
                - "postprocessing_label": (N,) array
                - "birdnet_confidence": (N,) array

        Returns:
            Tuple of (labels: np.ndarray, confidences: np.ndarray).
        """
        n = len(signals_batch.get("classifier_prob", []))

        labels = np.zeros(n, dtype=int)
        confidences = np.zeros(n, dtype=float)

        for i in range(n):
            single_signals = {
                key: float(arr[i]) if i < len(arr) else 0.0
                for key, arr in signals_batch.items()
            }
            label, conf, _ = self.decide_single(single_signals)
            labels[i] = label
            confidences[i] = conf

        n_bird = int((labels == config.BIRD_LABEL).sum())
        n_noise = int((labels == config.NOISE_LABEL).sum())
        print(f"  [Stage 8] Ensemble: {n_bird} bird, {n_noise} noise")

        return labels, confidences

    def __repr__(self):
        return (
            f"WeightedEnsembleDecider("
            f"weights={self.weights}, "
            f"threshold={self.threshold}, "
            f"birdnet_thresh={self.birdnet_threshold})"
        )


def generate_clean_dataset(
    ensemble_labels: np.ndarray,
    file_paths: np.ndarray,
    idx_val: np.ndarray = None,
) -> dict:
    """
    Create the final noise-aware dataset based on ensemble decisions.

    Copies audio segments into bird/ and noise/ subdirectories.

    Args:
        ensemble_labels: Binary labels from ensemble (0=noise, 1=bird).
        file_paths: All file paths (may be larger than labels if using val split).
        idx_val: Validation indices (if labels correspond to val set only).

    Returns:
        Dict with bird/noise counts.
    """
    print("\n" + "=" * 70)
    print("  Generating Noise-Aware Dataset")
    print("=" * 70)

    bird_dir = os.path.join(config.NOISE_AWARE_OUTPUT_DIR, "bird")
    noise_dir = os.path.join(config.NOISE_AWARE_OUTPUT_DIR, "noise")
    os.makedirs(bird_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)

    n_bird, n_noise = 0, 0

    for i in range(len(ensemble_labels)):
        # Map label index to file path
        if idx_val is not None:
            src = str(file_paths[idx_val[i]])
        else:
            src = str(file_paths[i])

        fname = os.path.basename(src)

        if ensemble_labels[i] == config.BIRD_LABEL:
            dst = os.path.join(bird_dir, fname)
            n_bird += 1
        else:
            dst = os.path.join(noise_dir, fname)
            n_noise += 1

        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"  Output: {n_bird} bird, {n_noise} noise segments")
    print(f"  Saved to {config.NOISE_AWARE_OUTPUT_DIR}")

    return {"n_bird": n_bird, "n_noise": n_noise}
