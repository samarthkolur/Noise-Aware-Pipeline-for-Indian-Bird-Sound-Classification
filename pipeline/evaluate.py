"""Metrics computation for the three-way benchmark (design.md §9)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Accuracy, Precision, Recall, F1, FPR, FNR, and (if y_prob given) ROC-AUC/PR-AUC.

    y_true/y_pred are 1 for bird, 0 for noise (design.md BIRD_LABEL/NOISE_LABEL).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None and len(set(y_true.tolist())) > 1:
        y_prob = np.asarray(y_prob)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return metrics


def per_segment_correctness(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """1.0 where prediction matches ground truth, 0.0 otherwise (for stats.py)."""
    return (np.asarray(y_true) == np.asarray(y_pred)).astype(np.float64)
