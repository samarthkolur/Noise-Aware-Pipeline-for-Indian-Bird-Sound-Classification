"""
Stage 4: Binary Classifier (Bird vs Noise)

Train a lightweight classifier on BirdNET embeddings to discriminate bird
vocalizations from environmental noise. Random Forest is the primary
classifier (fast, interpretable, <1 min training). MLP retained as optional.

The classifier is trained on the curated hard-negative dataset from Stage 3,
ensuring it learns to reject the "tricky" noise that fools BirdNET.

Output:
    models/rf_classifier.pkl   (or mlp_classifier.pt)
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Focal Loss (PyTorch, for MLP only) ─────────────────────────────────────

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
        bce_loss = self.bce(pred, target)
        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


# ─── Random Forest Classifier (Primary) ─────────────────────────────────────

class BirdNoiseRF:
    """
    Random Forest classifier for bird/noise discrimination.

    Trained on BirdNET embeddings (1024-d). Fast training (<1 min),
    interpretable feature importances, robust to overfitting.
    """

    def __init__(self):
        self.model = SklearnRF(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        self._is_trained = False

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> dict:
        """
        Train the Random Forest classifier.

        Args:
            X_train: Training embeddings (N, 1024).
            y_train: Training labels (N,).
            X_val: Validation embeddings (optional).
            y_val: Validation labels (optional).

        Returns:
            Dictionary with training metrics.
        """
        print(f"  Training Random Forest ({config.RF_N_ESTIMATORS} trees)...")
        self.model.fit(X_train, y_train)
        self._is_trained = True

        # Training metrics
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)

        metrics = {
            "train_accuracy": float(train_acc),
            "train_f1": float(train_f1),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)
            metrics["val_accuracy"] = float(val_acc)
            metrics["val_f1"] = float(val_f1)
            print(f"  Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
            print(f"  Train F1:  {train_f1:.4f} | Val F1:  {val_f1:.4f}")
        else:
            print(f"  Train acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probability of bird class."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained yet.")
        proba = self.model.predict_proba(X)
        # Bird class probability (column index depends on class order)
        bird_idx = list(self.model.classes_).index(config.BIRD_LABEL)
        return proba[:, bird_idx].astype(np.float32)

    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary labels (0=noise, 1=bird)."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def feature_importance(self, top_k: int = 20) -> list:
        """
        Return top-K most important embedding dimensions.

        Returns:
            List of (dimension_index, importance_score) tuples.
        """
        if not self._is_trained:
            return []
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_k]
        return [(int(i), float(importances[i])) for i in top_indices]

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "rf_classifier.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [Stage 4] RF classifier saved to {path}")

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "rf_classifier.pkl")
        instance = cls()
        instance.model = joblib.load(path)
        instance._is_trained = True
        print(f"  [Stage 4] RF classifier loaded from {path}")
        return instance


# ─── MLP Classifier (Optional Secondary) ────────────────────────────────────

class BirdNoiseMLP:
    """
    Multi-Layer Perceptron for binary bird/noise classification.

    Uses PyTorch with configurable hidden layers, dropout, and
    optional focal loss. Retained as an optional secondary classifier.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None):
        torch, nn = _get_torch()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or config.MLP_HIDDEN_DIMS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._build_model()

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

        self.model = nn.Sequential(*layers).to(self.device)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> dict:
        """
        Train the MLP with early stopping.

        Returns:
            Dictionary with training history.
        """
        torch, nn = _get_torch()

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device).unsqueeze(1)

        if X_val is not None:
            X_v = torch.FloatTensor(X_val).to(self.device)
            y_v = torch.FloatTensor(y_val).to(self.device).unsqueeze(1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.MLP_LEARNING_RATE)

        if config.USE_FOCAL_LOSS:
            criterion = FocalLoss()
        else:
            criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(config.MLP_EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_v)
                    val_loss = criterion(val_pred, y_v).item()
                    val_acc = ((val_pred >= 0.5).float() == y_v).float().mean().item()

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= config.MLP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)
                    break

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probability of bird class."""
        torch, _ = _get_torch()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_t).cpu().numpy().flatten()
        return probs.astype(np.float32)

    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary labels (0=noise, 1=bird)."""
        return (self.predict(X) >= threshold).astype(int)

    def save(self, path: str = None):
        torch, _ = _get_torch()
        path = path or os.path.join(config.MODELS_DIR, "mlp_classifier.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
        }, path)
        print(f"  [Stage 4] MLP classifier saved to {path}")

    @classmethod
    def load(cls, path: str = None):
        torch, _ = _get_torch()
        path = path or os.path.join(config.MODELS_DIR, "mlp_classifier.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
        )
        instance.model.load_state_dict(checkpoint["model_state"])
        print(f"  [Stage 4] MLP classifier loaded from {path}")
        return instance


# ─── Factory Function ───────────────────────────────────────────────────────

def create_classifier(input_dim: int = None, classifier_type: str = None):
    """
    Factory function to create the appropriate classifier.

    Args:
        input_dim: Required for MLP. Embedding dimension.
        classifier_type: One of "rf" or "mlp". Defaults to config value.

    Returns:
        Classifier instance with .train_model(), .predict(), .save() methods.
    """
    ctype = classifier_type or config.CLASSIFIER_TYPE

    if ctype == "rf":
        return BirdNoiseRF()
    elif ctype == "mlp":
        if input_dim is None:
            input_dim = config.EMBEDDING_DIM
        return BirdNoiseMLP(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown classifier type: {ctype}. Use 'rf' or 'mlp'.")
