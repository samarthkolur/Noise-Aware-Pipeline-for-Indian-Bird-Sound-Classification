"""
Stage 4: Out-of-Distribution (OOD) Detection

Detect and reject noise segments that lie outside the learned distribution
of bird vocalizations. Implements Mahalanobis Distance, One-Class SVM,
Isolation Forest, and Autoencoder Reconstruction Error detectors.
"""

import os
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MahalanobisDetector:
    """
    Mahalanobis Distance OOD detector.

    Computes distance of each sample from the bird-class centroid using
    the inverse covariance matrix. High distance → likely noise.
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.MAHALANOBIS_THRESHOLD
        self.mean = None
        self.cov_inv = None

    def fit(self, bird_embeddings: np.ndarray):
        """Fit on bird-only embeddings to learn distribution."""
        self.mean = np.mean(bird_embeddings, axis=0)
        cov = np.cov(bird_embeddings.T)
        # Regularize to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances (lower = more bird-like)."""
        diff = X - self.mean
        left = diff @ self.cov_inv
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (bird) if within threshold, 0 (noise) otherwise."""
        distances = self.score(X)
        return (distances < self.threshold).astype(int)

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "mahalanobis_ood.npz")
        np.savez(path, mean=self.mean, cov_inv=self.cov_inv, threshold=self.threshold)

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "mahalanobis_ood.npz")
        data = np.load(path)
        det = cls(threshold=float(data["threshold"]))
        det.mean = data["mean"]
        det.cov_inv = data["cov_inv"]
        return det


class OneClassSVMDetector:
    """One-Class SVM trained on bird embeddings only."""

    def __init__(self):
        self.model = OneClassSVM(
            kernel=config.OCSVM_KERNEL,
            nu=config.OCSVM_NU,
        )

    def fit(self, bird_embeddings: np.ndarray):
        self.model.fit(bird_embeddings)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Decision function scores (positive = inlier)."""
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (bird) for inliers, 0 (noise) for outliers."""
        preds = self.model.predict(X)  # +1 or -1
        return (preds == 1).astype(int)

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "ocsvm_ood.pkl")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "ocsvm_ood.pkl")
        det = cls.__new__(cls)
        det.model = joblib.load(path)
        return det


class IsolationForestDetector:
    """Isolation Forest anomaly detector."""

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=config.IFOREST_N_ESTIMATORS,
            contamination=config.IFOREST_CONTAMINATION,
            random_state=config.RANDOM_SEED,
        )

    def fit(self, bird_embeddings: np.ndarray):
        self.model.fit(bird_embeddings)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score (higher = more normal)."""
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (bird) for inliers, 0 (noise) for outliers."""
        preds = self.model.predict(X)  # +1 or -1
        return (preds == 1).astype(int)

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "iforest_ood.pkl")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "iforest_ood.pkl")
        det = cls.__new__(cls)
        det.model = joblib.load(path)
        return det


class AutoencoderDetector:
    """
    Autoencoder-based OOD detector.

    Trained to reconstruct bird embeddings. High reconstruction error
    indicates the input is out-of-distribution (noise).
    """

    def __init__(self, input_dim: int):
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = config.AE_RECONSTRUCTION_THRESHOLD

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.AE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_DIM, config.AE_LATENT_DIM),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.AE_LATENT_DIM, config.AE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_DIM, input_dim),
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def fit(self, bird_embeddings: np.ndarray):
        """Train autoencoder on bird-only embeddings."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X = torch.FloatTensor(bird_embeddings).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=config.AE_LEARNING_RATE)
        criterion = torch.nn.MSELoss()

        self.encoder.train()
        self.decoder.train()

        for epoch in range(config.AE_EPOCHS):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                encoded = self.encoder(batch)
                decoded = self.decoder(encoded)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  [AE Epoch {epoch+1}] loss={total_loss/len(loader):.6f}")

        # Set threshold from training data reconstruction errors
        errors = self.score(bird_embeddings)
        if self.threshold is None:
            # Use mean + 2*std as threshold
            self.threshold = float(np.mean(errors) + 2 * np.std(errors))
            print(f"  [AE] Auto threshold: {self.threshold:.6f}")

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE)."""
        import torch

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            encoded = self.encoder(X_t)
            decoded = self.decoder(encoded)
            errors = torch.mean((X_t - decoded) ** 2, dim=1).cpu().numpy()

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (bird) if reconstruction error < threshold, else 0."""
        errors = self.score(X)
        return (errors < self.threshold).astype(int)

    def save(self, path: str = None):
        import torch
        path = path or os.path.join(config.MODELS_DIR, "autoencoder_ood.pt")
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "input_dim": self.input_dim,
            "threshold": self.threshold,
        }, path)

    @classmethod
    def load(cls, path: str = None):
        import torch
        path = path or os.path.join(config.MODELS_DIR, "autoencoder_ood.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        det = cls(checkpoint["input_dim"])
        det.encoder.load_state_dict(checkpoint["encoder"])
        det.decoder.load_state_dict(checkpoint["decoder"])
        det.threshold = checkpoint["threshold"]
        return det


class EnsembleOOD:
    """
    Combines multiple OOD detectors via majority voting.

    A sample is classified as bird only if at least `min_agree` detectors
    predict it as in-distribution.
    """

    def __init__(self, detectors: list = None, min_agree: int = None):
        self.detectors = detectors or []
        self.min_agree = min_agree or max(1, len(self.detectors) // 2 + 1)

    def fit(self, bird_embeddings: np.ndarray):
        for det in self.detectors:
            det.fit(bird_embeddings)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote across detectors. 1=bird, 0=noise."""
        votes = np.zeros(len(X))
        for det in self.detectors:
            votes += det.predict(X)
        return (votes >= self.min_agree).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Average normalized score across detectors."""
        scores = []
        for det in self.detectors:
            s = det.score(X)
            # Normalize to [0, 1]
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                s = (s - s_min) / (s_max - s_min)
            scores.append(s)
        return np.mean(scores, axis=0)

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "ood_ensemble")
        os.makedirs(path, exist_ok=True)
        for i, det in enumerate(self.detectors):
            det.save(os.path.join(path, f"detector_{i}"))


def create_ood_detectors(
    input_dim: int,
    methods: list = None,
) -> EnsembleOOD:
    """
    Factory function to create an ensemble of OOD detectors.

    Args:
        input_dim: Embedding dimension (needed for autoencoder).
        methods: List of OOD method names. Defaults to config.OOD_METHODS.

    Returns:
        EnsembleOOD instance.
    """
    methods = methods or config.OOD_METHODS
    detectors = []

    for method in methods:
        if method == "mahalanobis":
            detectors.append(MahalanobisDetector())
        elif method == "ocsvm":
            detectors.append(OneClassSVMDetector())
        elif method == "iforest":
            detectors.append(IsolationForestDetector())
        elif method == "autoencoder":
            detectors.append(AutoencoderDetector(input_dim))
        else:
            print(f"  [WARN] Unknown OOD method: {method}")

    return EnsembleOOD(detectors)


if __name__ == "__main__":
    print("Stage 4: Use run_pipeline.py to train OOD detectors.")
