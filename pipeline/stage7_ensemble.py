"""
Stage 7: Ensemble Decision Mechanism

Combine multiple decision signals (classifier probability, OOD score,
harmonic ratio, BirdNET confidence) into a final bird/noise decision.
A segment is classified as bird only if sufficient signals agree
above their respective thresholds.
"""

import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class EnsembleDecider:
    """
    Multi-signal ensemble decision maker.

    Combines:
        1. Binary classifier probability
        2. OOD detector decision (in-distribution = bird)
        3. Harmonic energy ratio
        4. BirdNET confidence score

    A segment is classified as bird only if at least `min_agreement`
    signals exceed their respective thresholds.
    """

    def __init__(
        self,
        classifier_threshold: float = None,
        ood_required: bool = None,
        harmonic_threshold: float = None,
        birdnet_threshold: float = None,
        min_agreement: int = None,
    ):
        self.classifier_threshold = classifier_threshold or config.ENSEMBLE_CLASSIFIER_THRESHOLD
        self.ood_required = ood_required if ood_required is not None else config.ENSEMBLE_OOD_REQUIRED
        self.harmonic_threshold = harmonic_threshold or config.ENSEMBLE_HARMONIC_THRESHOLD
        self.birdnet_threshold = birdnet_threshold or config.ENSEMBLE_BIRDNET_THRESHOLD
        self.min_agreement = min_agreement or config.ENSEMBLE_MIN_AGREEMENT

    def decide_single(self, signals: dict) -> tuple:
        """
        Make a decision for a single segment.

        Args:
            signals: Dictionary with keys:
                - "classifier_prob": float (0–1)
                - "ood_is_bird": int (0 or 1)
                - "harmonic_ratio": float (0–1)
                - "birdnet_confidence": float (0–1)

        Returns:
            Tuple of (label: int, confidence: float, detail: dict).
        """
        votes = {}

        # Signal 1: Classifier probability
        clf_prob = signals.get("classifier_prob", 0.0)
        votes["classifier"] = 1 if clf_prob >= self.classifier_threshold else 0

        # Signal 2: OOD detector
        ood = signals.get("ood_is_bird", 1)
        votes["ood"] = int(ood)

        # Signal 3: Harmonic ratio
        h_ratio = signals.get("harmonic_ratio", 0.0)
        votes["harmonic"] = 1 if h_ratio >= self.harmonic_threshold else 0

        # Signal 4: BirdNET confidence
        bn_conf = signals.get("birdnet_confidence", 0.0)
        votes["birdnet"] = 1 if bn_conf >= self.birdnet_threshold else 0

        # Count agreements
        total_votes = sum(votes.values())
        is_bird = total_votes >= self.min_agreement

        # Compute ensemble confidence
        confidence = np.mean([clf_prob, float(ood), h_ratio, bn_conf])

        return int(is_bird), float(confidence), votes

    def decide_batch(self, signals_batch: dict) -> tuple:
        """
        Make decisions for a batch of segments.

        Args:
            signals_batch: Dictionary where each value is a numpy array:
                - "classifier_prob": (N,) array
                - "ood_is_bird": (N,) array
                - "harmonic_ratio": (N,) array
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

        return labels, confidences

    def __repr__(self):
        return (
            f"EnsembleDecider("
            f"clf_thresh={self.classifier_threshold}, "
            f"ood_required={self.ood_required}, "
            f"harmonic_thresh={self.harmonic_threshold}, "
            f"birdnet_thresh={self.birdnet_threshold}, "
            f"min_agree={self.min_agreement})"
        )


if __name__ == "__main__":
    print("Stage 7: Use run_pipeline.py to run ensemble decisions.")
