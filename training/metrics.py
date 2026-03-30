"""
metrics.py — Evaluation metrics for binary and multiclass classification.

Supports:
  • Configurable decision threshold (binary mode)
  • Confusion matrix generation
  • Optimal threshold search (maximize F1 or recall at min precision)
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    binary: bool,
    threshold: float = 0.5,
) -> dict:
    """Compute accuracy, precision, recall, and F1.

    Args:
        logits: (B, C) raw scores if multiclass, or (B,) if binary.
        labels: (B,) ground-truth integers.
        binary: If True, treats this as a binary classification problem.
        threshold: Decision threshold for binary mode (default 0.5).

    Returns:
        Dict with keys: acc, prec, rec, f1
    """
    labels_np = labels.cpu().numpy()

    if binary:
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds_np = (probs > threshold).cpu().numpy().astype(int)
        average = "binary"
    else:
        preds_np = logits.argmax(dim=1).cpu().numpy()
        average = "macro"

    acc = accuracy_score(labels_np, preds_np)
    prec = precision_score(labels_np, preds_np, average=average, zero_division=0)
    rec = recall_score(labels_np, preds_np, average=average, zero_division=0)
    f1 = f1_score(labels_np, preds_np, average=average, zero_division=0)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
    }


def compute_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    binary: bool,
    threshold: float = 0.5,
    label_names: list | None = None,
) -> str:
    """Generate a formatted confusion matrix string.

    For binary mode, returns TP/FP/FN/TN counts.
    """
    labels_np = labels.cpu().numpy()

    if binary:
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds_np = (probs > threshold).cpu().numpy().astype(int)
    else:
        preds_np = logits.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1] if binary else None)

    if binary:
        tn, fp, fn, tp = cm.ravel()
        lines = [
            "Confusion Matrix (threshold={:.2f}):".format(threshold),
            "                 Predicted",
            "               Noise   Bird",
            f"  Actual Noise  {tn:5d}  {fp:5d}",
            f"  Actual Bird   {fn:5d}  {tp:5d}",
            "",
            f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}",
            f"  Precision = {tp / max(tp + fp, 1):.4f}",
            f"  Recall    = {tp / max(tp + fn, 1):.4f}",
            f"  F1        = {2 * tp / max(2 * tp + fp + fn, 1):.4f}",
        ]
    else:
        names = label_names or [str(i) for i in range(cm.shape[0])]
        header = "         " + "  ".join(f"{n:>7s}" for n in names)
        lines = ["Confusion Matrix:", header]
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:7d}" for v in row)
            lines.append(f"  {names[i]:>7s}  {row_str}")

    return "\n".join(lines)


def find_optimal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "f1",
    min_precision: float = 0.5,
    steps: int = 50,
) -> dict:
    """Sweep thresholds to find the optimal decision boundary.

    Args:
        logits: (B,) raw binary logits.
        labels: (B,) ground-truth binary labels.
        metric: Optimization target — 'f1', 'recall', or 'recall_at_precision'.
        min_precision: Minimum precision constraint when metric='recall_at_precision'.
        steps: Number of threshold candidates to evaluate.

    Returns:
        Dict with 'best_threshold', 'best_value', and 'curve' (list of dicts).
    """
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()

    thresholds = np.linspace(0.05, 0.95, steps)
    curve = []
    best_val = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (probs > t).astype(int)
        tp = int(((preds == 1) & (labels_np == 1)).sum())
        fp = int(((preds == 1) & (labels_np == 0)).sum())
        fn = int(((preds == 0) & (labels_np == 1)).sum())
        tn = int(((preds == 0) & (labels_np == 0)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        entry = {"threshold": float(t), "precision": prec, "recall": rec, "f1": f1}
        curve.append(entry)

        if metric == "f1":
            score = f1
        elif metric == "recall":
            score = rec
        elif metric == "recall_at_precision":
            score = rec if prec >= min_precision else -1.0
        else:
            score = f1

        if score > best_val:
            best_val = score
            best_thresh = float(t)

    return {
        "best_threshold": best_thresh,
        "best_value": best_val,
        "metric": metric,
        "curve": curve,
    }
