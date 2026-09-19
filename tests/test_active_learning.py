"""
Test: Active Learning — Uncertainty sampling and CSV export/import.

Validates that:
    1. UncertaintySampler correctly identifies samples near the decision boundary
    2. CSV export/import round-trips preserve data
    3. ActiveLearningLoop.retrain_with_feedback() augments training correctly
"""

import os
import sys
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.stage6_active_learning import UncertaintySampler, ActiveLearningLoop


class _MockClassifier:
    """Mock classifier that returns fixed probabilities."""
    def __init__(self, probs):
        self._probs = probs

    def predict(self, X):
        return self._probs[:len(X)]


def test_uncertainty_sampling():
    """Test that uncertain samples are near the 0.5 boundary."""
    print("\n── Test: Uncertainty Sampling ──")
    np.random.seed(42)

    n = 100
    # Create probs with some near 0.5 and some far away
    probs = np.random.uniform(0, 1, n).astype(np.float32)
    # Override some to be exactly near boundary
    probs[0] = 0.50
    probs[1] = 0.49
    probs[2] = 0.51
    probs[3] = 0.05
    probs[4] = 0.95

    clf = _MockClassifier(probs)
    X = np.random.randn(n, 32).astype(np.float32)

    sampler = UncertaintySampler(uncertainty_threshold=0.1, top_k=10)
    indices, scores, pred_probs = sampler.find_uncertain_samples(clf, X)

    # Check that samples near 0.5 are included
    for idx in indices:
        assert abs(probs[idx] - 0.5) < 0.1, (
            f"Sample {idx} too far from boundary: prob={probs[idx]}"
        )

    # 0, 1, 2 should be in there (very close to 0.5)
    assert 0 in indices
    assert 1 in indices
    assert 2 in indices

    # 3 and 4 should NOT be in there (far from 0.5)
    assert 3 not in indices
    assert 4 not in indices

    print(f"  Found {len(indices)} uncertain samples, all near boundary ✓")
    print("✅ Uncertainty sampling: PASSED")


def test_csv_export_import():
    """Test CSV export and import round-trip."""
    print("\n── Test: CSV Export/Import ──")

    tmp_dir = tempfile.mkdtemp()
    original_al_dir = config.ACTIVE_LEARNING_DIR
    config.ACTIVE_LEARNING_DIR = tmp_dir

    try:
        indices = np.array([0, 1, 2])
        scores = np.array([0.02, 0.05, 0.08])
        probs = np.array([0.52, 0.45, 0.58])
        paths = np.array(["/audio/seg_001.wav", "/audio/seg_002.wav", "/audio/seg_003.wav"])

        sampler = UncertaintySampler()
        csv_path = sampler.export_for_review(indices, scores, probs, paths, round_num=1)

        assert os.path.exists(csv_path)

        # Simulate expert labels
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Fill in expert labels
        rows[0]["expert_label"] = "bird"
        rows[1]["expert_label"] = "noise"
        rows[2]["expert_label"] = "bird"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Import
        expert_labels = UncertaintySampler.import_expert_labels(csv_path)
        assert len(expert_labels) == 3
        assert expert_labels["/audio/seg_001.wav"] == config.BIRD_LABEL
        assert expert_labels["/audio/seg_002.wav"] == config.NOISE_LABEL
        assert expert_labels["/audio/seg_003.wav"] == config.BIRD_LABEL

        print("✅ CSV export/import: PASSED")

    finally:
        config.ACTIVE_LEARNING_DIR = original_al_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retrain_with_feedback():
    """Test that feedback augments training set correctly."""
    print("\n── Test: Retrain with Feedback ──")

    X_train = np.random.randn(20, 8).astype(np.float32)
    y_train = np.array([0] * 10 + [1] * 10)
    X_pool = np.random.randn(5, 8).astype(np.float32)
    pool_paths = np.array([f"/pool/{i}.wav" for i in range(5)])

    expert_labels = {
        "/pool/0.wav": 1,
        "/pool/2.wav": 0,
    }

    al = ActiveLearningLoop()
    X_aug, y_aug, X_remaining, paths_remaining = al.retrain_with_feedback(
        expert_labels, X_train, y_train, X_pool, pool_paths
    )

    assert len(X_aug) == 22  # 20 original + 2 expert-labeled
    assert len(y_aug) == 22
    assert len(X_remaining) == 3  # 5 - 2 moved to training
    assert len(paths_remaining) == 3

    print("✅ Retrain with feedback: PASSED")


if __name__ == "__main__":
    test_uncertainty_sampling()
    test_csv_export_import()
    test_retrain_with_feedback()
    print("\n✅ All active learning tests passed!")
