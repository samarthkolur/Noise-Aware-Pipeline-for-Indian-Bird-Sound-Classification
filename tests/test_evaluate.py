import numpy as np

from pipeline.evaluate import compute_metrics, per_segment_correctness
from pipeline.stats import paired_significance


def test_perfect_predictions_give_perfect_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = y_true.copy()
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0


def test_metrics_on_hand_crafted_confusion():
    # 2 TP, 1 FP, 1 FN, 2 TN
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
    assert abs(metrics["accuracy"] - 4 / 6) < 1e-9
    assert abs(metrics["fpr"] - 1 / 3) < 1e-9
    assert abs(metrics["fnr"] - 1 / 3) < 1e-9


def test_roc_auc_present_when_probs_given_and_both_classes_present():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == 1.0


def test_per_segment_correctness():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])
    correctness = per_segment_correctness(y_true, y_pred)
    np.testing.assert_array_equal(correctness, [1.0, 0.0, 1.0, 1.0])


def test_paired_significance_identical_arrays_not_significant():
    correctness = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    result = paired_significance(correctness, correctness)
    assert result["significant_at_0.05"] is False


def test_paired_significance_detects_clear_improvement():
    rng = np.random.default_rng(0)
    baseline = (rng.random(200) > 0.5).astype(float)  # ~50% correct
    improved = (rng.random(200) > 0.05).astype(float)  # ~95% correct
    result = paired_significance(baseline, improved)
    assert result["t_p_value"] < 0.05
    assert result["wilcoxon_p_value"] < 0.05


def test_paired_significance_requires_equal_length():
    import pytest

    with pytest.raises(ValueError):
        paired_significance(np.array([1.0, 0.0]), np.array([1.0]))
