"""
Stage 3: Supervised Binary Classifier (Bird vs Noise)

Trains a bird/noise classifier on embedding vectors. Supports MLP (PyTorch),
Logistic Regression, SVM, and Random Forest. Includes Focal Loss for
class-imbalanced training.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Focal Loss (PyTorch) ───────────────────────────────────────────────────

def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class FocalLoss:
    """Binary Focal Loss for class-imbalanced bird/noise classification."""

    def __init__(self, gamma=None, alpha=None):
        torch, nn = _get_torch()
        self.gamma = gamma or config.FOCAL_LOSS_GAMMA
        self.alpha = alpha or config.FOCAL_LOSS_ALPHA
        self.bce = nn.BCELoss(reduction="none")

    def __call__(self, pred, target):
        torch, _ = _get_torch()
        bce_loss = self.bce(pred, target)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


# ─── MLP Classifier (PyTorch) ──────────────────────────────────────────────

class BirdNoiseMLP:
    """
    Multi-Layer Perceptron for binary bird/noise classification.

    Uses PyTorch with configurable hidden layers, dropout, and
    optional focal loss.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None):
        torch, nn = _get_torch()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or config.MLP_HIDDEN_DIMS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self):
        torch, nn = _get_torch()
        layers = []
        prev_dim = self.input_dim

        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(config.MLP_DROPOUT),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Train the MLP with early stopping.

        Returns:
            Dictionary with training history (train_loss, val_loss, val_acc per epoch).
        """
        torch, nn = _get_torch()
        from torch.utils.data import TensorDataset, DataLoader

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.MLP_LEARNING_RATE)

        if config.USE_FOCAL_LOSS:
            criterion = FocalLoss()
        else:
            criterion = nn.BCELoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.MLP_EPOCHS):
            # ── Training ──
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_train_loss)

            # ── Validation ──
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  [Epoch {epoch+1:3d}] "
                    f"train_loss={avg_train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_acc={val_acc:.4f}"
                )

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.MLP_PATIENCE:
                    print(f"  [Early stopping at epoch {epoch+1}]")
                    break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probability of bird class."""
        torch, _ = _get_torch()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_t).cpu().numpy().flatten()
        return probs

    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary labels (0=noise, 1=bird)."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def save(self, path: str = None):
        torch, _ = _get_torch()
        path = path or os.path.join(config.MODELS_DIR, "mlp_classifier.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
        }, path)
        print(f"  [Stage 3] MLP saved to {path}")

    @classmethod
    def load(cls, path: str = None):
        torch, _ = _get_torch()
        path = path or os.path.join(config.MODELS_DIR, "mlp_classifier.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls(checkpoint["input_dim"], checkpoint["hidden_dims"])
        clf.model.load_state_dict(checkpoint["model_state"])
        print(f"  [Stage 3] MLP loaded from {path}")
        return clf


# ─── Sklearn Baselines ─────────────────────────────────────────────────────

class SklearnClassifier:
    """Wrapper around sklearn classifiers for bird/noise classification."""

    def __init__(self, classifier_type: str = None):
        ctype = classifier_type or config.CLASSIFIER_TYPE

        if ctype == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
        elif ctype == "svm":
            self.model = SVC(probability=True, random_state=config.RANDOM_SEED)
        elif ctype == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200, random_state=config.RANDOM_SEED
            )
        else:
            raise ValueError(f"Unknown classifier type: {ctype}")

        self.classifier_type = ctype

    def train_model(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        self.model.fit(X_train, y_train)
        history = {}

        if X_val is not None:
            y_pred = self.model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            history["val_acc"] = acc
            print(f"  [Stage 3] {self.classifier_type} val_acc={acc:.4f}")
            print(classification_report(y_val, y_pred, target_names=["noise", "bird"]))

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X).astype(float)

    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def save(self, path: str = None):
        path = path or os.path.join(
            config.MODELS_DIR, f"{self.classifier_type}_classifier.pkl"
        )
        joblib.dump(self.model, path)
        print(f"  [Stage 3] {self.classifier_type} saved to {path}")

    @classmethod
    def load(cls, path: str, classifier_type: str = "logistic"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.classifier_type = classifier_type
        return obj


# ─── Convenience Functions ──────────────────────────────────────────────────

def create_classifier(input_dim: int = None, classifier_type: str = None):
    """
    Factory function to create either an MLP or sklearn classifier.

    Args:
        input_dim: Required for MLP. Embedding dimension.
        classifier_type: One of "mlp", "logistic", "svm", "rf".

    Returns:
        Classifier instance with .train_model(), .predict(), .save() methods.
    """
    ctype = classifier_type or config.CLASSIFIER_TYPE
    if ctype == "mlp":
        if input_dim is None:
            raise ValueError("input_dim required for MLP classifier")
        return BirdNoiseMLP(input_dim)
    else:
        return SklearnClassifier(ctype)


def generate_pseudo_labels(
    segment_dir: str = None,
    threshold: float = None,
) -> dict:
    """
    Generate bird/noise pseudo-labels using BirdNET confidence scores.

    Segments with max BirdNET confidence below threshold → noise.
    All others → bird.

    Args:
        segment_dir: Directory with segmented audio files.
        threshold: BirdNET confidence threshold.

    Returns:
        Dict mapping file paths to labels (0=noise, 1=bird).
    """
    import glob
    from pipeline.stage2_embeddings import get_birdnet_confidence

    segment_dir = segment_dir or config.SEGMENTED_DIR
    threshold = threshold or config.BIRDNET_NOISE_THRESHOLD

    audio_files = glob.glob(os.path.join(segment_dir, "**", "*.wav"), recursive=True)
    print(f"[Stage 3] Generating pseudo-labels for {len(audio_files)} segments")

    labels = {}
    for fpath in tqdm(audio_files, desc="[Stage 3] Pseudo-labeling"):
        conf = get_birdnet_confidence(fpath)
        labels[fpath] = config.BIRD_LABEL if conf >= threshold else config.NOISE_LABEL

    n_bird = sum(1 for v in labels.values() if v == config.BIRD_LABEL)
    n_noise = sum(1 for v in labels.values() if v == config.NOISE_LABEL)
    print(f"[Stage 3] Pseudo-labels: {n_bird} bird, {n_noise} noise")

    return labels


if __name__ == "__main__":
    print("Stage 3: Use run_pipeline.py to train the classifier.")
