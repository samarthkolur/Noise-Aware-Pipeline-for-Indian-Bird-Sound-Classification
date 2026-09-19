"""
Test: Binary Classifier (RF and MLP) train/predict round-trip.

Validates that classifiers can be trained on synthetic embeddings,
produce valid predictions, and survive save/load round-trips.
"""

import os
import sys
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.stage4_binary_classifier import BirdNoiseRF, BirdNoiseMLP, create_classifier


def _synthetic_data(dim=1024, n_train=200, n_val=50, seed=42):
    """Generate synthetic bird/noise embeddings."""
    np.random.seed(seed)
    # Bird: cluster around [1, 1, ...]
    X_bird = np.random.randn(n_train // 2, dim).astype(np.float32) + 1.0
    X_noise = np.random.randn(n_train // 2, dim).astype(np.float32) - 1.0
    X_train = np.vstack([X_bird, X_noise])
    y_train = np.array([1] * (n_train // 2) + [0] * (n_train // 2))

    X_val = np.random.randn(n_val, dim).astype(np.float32)
    y_val = np.array([1] * (n_val // 2) + [0] * (n_val // 2))

    return X_train, y_train, X_val, y_val


def test_rf_classifier():
    """Test Random Forest classifier."""
    print("\n── Test: RF Classifier ──")
    X_train, y_train, X_val, y_val = _synthetic_data()

    clf = BirdNoiseRF()
    metrics = clf.train_model(X_train, y_train, X_val, y_val)

    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics
    assert metrics["val_accuracy"] > 0.5, f"Val accuracy too low: {metrics['val_accuracy']}"

    # Predictions
    probs = clf.predict(X_val)
    assert probs.shape == (len(X_val),)
    assert np.all((probs >= 0) & (probs <= 1))

    labels = clf.predict_labels(X_val)
    assert set(np.unique(labels)).issubset({0, 1})

    # Feature importance
    top = clf.feature_importance(top_k=5)
    assert len(top) == 5

    # Save/load round-trip
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "rf_test.pkl")
        clf.save(path)
        loaded = BirdNoiseRF.load(path)
        probs_loaded = loaded.predict(X_val)
        assert np.allclose(probs, probs_loaded), "Predictions differ after save/load"
        print("✅ RF classifier: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_mlp_classifier():
    """Test MLP classifier."""
    print("\n── Test: MLP Classifier ──")
    dim = 64  # Smaller for speed
    X_train, y_train, X_val, y_val = _synthetic_data(dim=dim, n_train=100, n_val=20)

    clf = BirdNoiseMLP(input_dim=dim, hidden_dims=[32, 16])

    # Override epochs for speed
    original_epochs = config.MLP_EPOCHS
    config.MLP_EPOCHS = 10
    metrics = clf.train_model(X_train, y_train, X_val, y_val)
    config.MLP_EPOCHS = original_epochs

    probs = clf.predict(X_val)
    assert probs.shape == (len(X_val),)
    assert np.all((probs >= 0) & (probs <= 1))

    # Save/load
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "mlp_test.pt")
        clf.save(path)
        loaded = BirdNoiseMLP.load(path)
        probs_loaded = loaded.predict(X_val)
        assert np.allclose(probs, probs_loaded, atol=1e-5), (
            "MLP predictions differ after save/load"
        )
        print("✅ MLP classifier: PASSED")
    finally:
        shutil.rmtree(tmp_dir)


def test_factory():
    """Test classifier factory function."""
    print("\n── Test: Factory Function ──")
    clf_rf = create_classifier(classifier_type="rf")
    assert isinstance(clf_rf, BirdNoiseRF)

    clf_mlp = create_classifier(input_dim=64, classifier_type="mlp")
    assert isinstance(clf_mlp, BirdNoiseMLP)
    print("✅ Factory function: PASSED")


if __name__ == "__main__":
    test_rf_classifier()
    test_mlp_classifier()
    test_factory()
    print("\n✅ All classifier tests passed!")
