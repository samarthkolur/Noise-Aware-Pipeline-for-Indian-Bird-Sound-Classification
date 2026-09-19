"""
Stage 5: Out-of-Distribution (OOD) Detection

Detect and reject noise segments that lie outside the learned distribution
of bird vocalizations. Uses Mahalanobis Distance and Isolation Forest —
two complementary detectors for robust anomaly filtering.

Mahalanobis: statistical distance from bird centroid (parametric).
IForest: isolation-based anomaly score (non-parametric).

Output:
    models/ood_ensemble/  (saved detector models + metadata.json)
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Mahalanobis Distance Detector ──────────────────────────────────────────

class MahalanobisDetector:
    """
    Mahalanobis Distance OOD detector.

    Computes distance of each sample from the bird-class centroid using
    the inverse covariance matrix. High distance → likely noise.
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.MAHALANOBIS_THRESHOLD
        self.mean = None
        self.inv_cov = None

    def fit(self, bird_embeddings: np.ndarray):
        """Fit on bird-only embeddings to learn distribution."""
        self.mean = np.mean(bird_embeddings, axis=0)
        cov = np.cov(bird_embeddings.T)
        # Regularize covariance matrix to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances (lower = more bird-like)."""
        diff = X - self.mean
        left = diff @ self.inv_cov
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (bird) if within threshold, 0 (noise) otherwise."""
        distances = self.score(X)
        return (distances <= self.threshold).astype(int)

    def save(self, path: str):
        np.savez(path, mean=self.mean, inv_cov=self.inv_cov, threshold=self.threshold)

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        instance = cls(threshold=float(data["threshold"]))
        instance.mean = data["mean"]
        instance.inv_cov = data["inv_cov"]
        return instance


# ─── Isolation Forest Detector ───────────────────────────────────────────────

class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.

    Points that are easy to isolate (few splits) are anomalies.
    Trained on bird embeddings — noise segments will be outliers.
    """

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=config.IFOREST_N_ESTIMATORS,
            contamination=config.IFOREST_CONTAMINATION,
            random_state=config.RANDOM_SEED,
        )

    def fit(self, bird_embeddings: np.ndarray):
        self.model.fit(bird_embeddings)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score (higher = more normal)."""
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (bird) for inliers, 0 (noise) for outliers."""
        raw = self.model.predict(X)
        return (raw == 1).astype(int)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        instance = cls()
        instance.model = joblib.load(path)
        return instance


# ─── Ensemble OOD ────────────────────────────────────────────────────────────

class EnsembleOOD:
    """
    Ensemble of OOD detectors. A sample is classified as bird (in-distribution)
    only if ALL detectors agree it's an inlier.

    With 2 detectors (Mahalanobis + IForest), this is effectively an AND gate:
    both must say "bird" for it to pass.
    """

    def __init__(self, detectors: list = None):
        self.detectors = detectors or []

    def fit(self, bird_embeddings: np.ndarray):
        """Train all detectors on bird-only embeddings."""
        for det in self.detectors:
            det.fit(bird_embeddings)
        print(f"  [Stage 5] Trained {len(self.detectors)} OOD detectors")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using all detectors. A sample is bird only if all agree.

        Returns:
            (N,) array of 0 (noise/OOD) or 1 (bird/in-distribution).
        """
        if not self.detectors:
            return np.ones(len(X), dtype=int)

        predictions = np.stack([det.predict(X) for det in self.detectors], axis=0)
        # AND gate: all must agree
        return np.all(predictions == 1, axis=0).astype(int)

    def save(self, save_dir: str = None):
        """Save all detectors + metadata to a directory."""
        save_dir = save_dir or os.path.join(config.MODELS_DIR, "ood_ensemble")
        os.makedirs(save_dir, exist_ok=True)

        metadata = {"detectors": []}
        for i, det in enumerate(self.detectors):
            class_name = type(det).__name__
            if isinstance(det, MahalanobisDetector):
                fname = f"detector_{i}.npz"
            else:
                fname = f"detector_{i}.pkl"

            det.save(os.path.join(save_dir, fname))
            metadata["detectors"].append({
                "index": i,
                "class": class_name,
                "file": fname,
            })

        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  [Stage 5] OOD ensemble saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str = None):
        """Load ensemble from a saved directory."""
        load_dir = load_dir or os.path.join(config.MODELS_DIR, "ood_ensemble")

        with open(os.path.join(load_dir, "metadata.json")) as f:
            metadata = json.load(f)

        _DETECTOR_MAP = {
            "MahalanobisDetector": MahalanobisDetector,
            "IsolationForestDetector": IsolationForestDetector,
        }

        detectors = []
        for info in metadata["detectors"]:
            det_class = _DETECTOR_MAP[info["class"]]
            det = det_class.load(os.path.join(load_dir, info["file"]))
            detectors.append(det)

        print(f"  [Stage 5] Loaded {len(detectors)} OOD detectors from {load_dir}")
        return cls(detectors)


def create_ood_detectors(input_dim: int = None, methods: list = None) -> EnsembleOOD:
    """
    Factory function to create OOD detector ensemble.

    Args:
        input_dim: Not used (kept for API compatibility).
        methods: List of method names. Defaults to config.OOD_METHODS.

    Returns:
        EnsembleOOD instance.
    """
    methods = methods or config.OOD_METHODS

    detectors = []
    for method in methods:
        if method == "mahalanobis":
            detectors.append(MahalanobisDetector())
        elif method == "iforest":
            detectors.append(IsolationForestDetector())
        else:
            print(f"  [WARN] Unknown OOD method: {method}, skipping")

    return EnsembleOOD(detectors)
