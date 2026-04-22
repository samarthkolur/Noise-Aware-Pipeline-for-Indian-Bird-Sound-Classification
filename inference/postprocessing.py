"""
postprocessing.py — Confidence filtering and temporal smoothing of predictions.
"""

from typing import Dict, List, Tuple

import numpy as np


class PostProcessor:
    """Post-process per-segment predictions for a single recording."""

    def __init__(self, cfg: dict) -> None:
        self.confidence_threshold = cfg["inference"]["confidence_threshold"]
        ts_cfg = cfg["inference"]["temporal_smoothing"]
        self.smoothing_enabled = ts_cfg["enabled"]
        self.window_size = ts_cfg["window_size"]

    def filter_by_confidence(
        self, predictions: List[Dict]
    ) -> List[Dict]:
        """Remove predictions below the confidence threshold.

        Args:
            predictions: List of segment dicts from Predictor.

        Returns:
            Filtered list.
        """
        filtered = []
        for seg in predictions:
            kept = [
                (species, conf)
                for species, conf in seg["predictions"]
                if conf >= self.confidence_threshold
            ]
            if kept:
                filtered.append({**seg, "predictions": kept})
        return filtered

    def temporal_smooth(
        self,
        predictions: List[Dict],
    ) -> List[Dict]:
        """Smooth predictions over adjacent segments using majority voting.

        Args:
            predictions: List of segment dicts, ordered by segment_idx.

        Returns:
            Smoothed predictions.
        """
        if not self.smoothing_enabled or len(predictions) < self.window_size:
            return predictions

        n = len(predictions)
        half_w = self.window_size // 2
        smoothed = []

        for i in range(n):
            window_start = max(0, i - half_w)
            window_end = min(n, i + half_w + 1)

            # Collect all top-1 species in the window
            species_votes: Dict[str, List[float]] = {}
            for j in range(window_start, window_end):
                if predictions[j]["predictions"]:
                    sp, conf = predictions[j]["predictions"][0]
                    species_votes.setdefault(sp, []).append(conf)

            if species_votes:
                # Pick species with most votes, break ties by mean confidence
                best_species = max(
                    species_votes,
                    key=lambda s: (
                        len(species_votes[s]),
                        np.mean(species_votes[s]),
                    ),
                )
                avg_conf = float(np.mean(species_votes[best_species]))
                smoothed.append(
                    {
                        "segment_idx": predictions[i]["segment_idx"],
                        "predictions": [(best_species, avg_conf)],
                    }
                )
            else:
                smoothed.append(predictions[i])

        return smoothed
