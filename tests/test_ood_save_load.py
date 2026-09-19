"""
Test: OOD Ensemble save/load round-trip (V3 — 2 detectors).

Validates that EnsembleOOD with Mahalanobis + IForest preserves
detector state and produces identical predictions after save/load.
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.stage5_ood_filter import (
    MahalanobisDetector,
    IsolationForestDetector,
    EnsembleOOD,
    create_ood_detectors,
)


def test_ensemble_ood_save_load():
    """Train, save, load, and compare predictions."""
    print("\n── Test: OOD Ensemble ──")
    np.random.seed(42)
    dim = 32
    n_train = 200
    n_test = 50

    bird_train = np.random.randn(n_train, dim).astype(np.float32) * 0.5
    X_test = np.random.randn(n_test, dim).astype(np.float32) * 2.0

    ensemble = create_ood_detectors(methods=["mahalanobis", "iforest"])
    assert len(ensemble.detectors) == 2

    ensemble.fit(bird_train)
    preds_original = ensemble.predict(X_test)

    tmp_dir = tempfile.mkdtemp(prefix="ood_test_")
    save_path = os.path.join(tmp_dir, "ood_ensemble")

    try:
        ensemble.save(save_path)

        meta_path = os.path.join(save_path, "metadata.json")
        assert os.path.exists(meta_path), f"metadata.json not found"

        with open(meta_path) as f:
            metadata = json.load(f)
        assert len(metadata["detectors"]) == 2

        loaded = EnsembleOOD.load(save_path)
        preds_loaded = loaded.predict(X_test)

        assert np.array_equal(preds_original, preds_loaded), (
            f"Predictions differ!\n  Original: {preds_original[:10]}\n"
            f"  Loaded:   {preds_loaded[:10]}"
        )

        print(f"✅ OOD ensemble save/load: PASSED (2 detectors, {n_test} test samples)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_individual_detectors():
    """Test each detector independently."""
    print("\n── Test: Individual Detectors ──")
    np.random.seed(42)
    dim = 16
    bird_data = np.random.randn(100, dim).astype(np.float32)
    test_data = np.random.randn(20, dim).astype(np.float32) * 3.0

    # Mahalanobis
    mah = MahalanobisDetector(threshold=25.0)
    mah.fit(bird_data)
    scores = mah.score(test_data)
    preds = mah.predict(test_data)
    assert scores.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})
    print("  Mahalanobis: OK")

    # Isolation Forest
    ifo = IsolationForestDetector()
    ifo.fit(bird_data)
    scores = ifo.score(test_data)
    preds = ifo.predict(test_data)
    assert scores.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})
    print("  IsolationForest: OK")

    print("✅ Individual detectors: PASSED")


if __name__ == "__main__":
    test_individual_detectors()
    test_ensemble_ood_save_load()
    print("\n✅ All OOD tests passed!")
