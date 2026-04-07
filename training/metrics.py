"""
metrics.py — Evaluation metrics for binary and multiclass classification.
"""

from __future__ import annotations

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
    """Compute accuracy, precision, recall, and F1."""
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
    """Return a printable confusion matrix string."""
    labels_np = labels.cpu().numpy()
    if binary:
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds_np = (probs > threshold).cpu().numpy().astype(int)
    else:
        preds_np = logits.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(labels_np, preds_np)
    lines = ["Confusion Matrix (rows=true, cols=pred):"]
    lines.append(str(cm))
    return "\n".join(lines)


def compute_binary_per_class_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Precision/recall for class 0 (noise) and class 1 (bird); FPR on noise."""
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > threshold).astype(int)
    y = labels.cpu().numpy()

    # Class 1 = bird (positive in sklearn "positive" sense for PR on bird)
    prec_bird = precision_score(y, preds, pos_label=1, zero_division=0)
    rec_bird = recall_score(y, preds, pos_label=1, zero_division=0)
    prec_noise = precision_score(y, preds, pos_label=0, zero_division=0)
    rec_noise = recall_score(y, preds, pos_label=0, zero_division=0)

    # FPR on noise: P(pred bird | true noise) = FP / (TN+FP)
    noise_mask = y == 0
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        fpr_noise = float("nan")
    else:
        fp = int((preds[noise_mask] == 1).sum())
        fpr_noise = fp / n_noise

    return {
        "prec_bird": float(prec_bird),
        "rec_bird": float(rec_bird),
        "prec_noise": float(prec_noise),
        "rec_noise": float(rec_noise),
        "fpr_noise": fpr_noise,
    }


def find_optimal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "f1",
    steps: int = 50,
    min_precision: float | None = None,
) -> dict:
    """Grid-search thresholds in (0, 1) for binary logits."""
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, steps)
    curve = []
    best_t = 0.5
    best_val = 0.0

    for t in thresholds:
        preds = (probs > t).astype(int)
        prec = precision_score(y, preds, average="binary", zero_division=0)
        rec = recall_score(y, preds, average="binary", zero_division=0)
        f1 = f1_score(y, preds, average="binary", zero_division=0)

        curve.append(
            {
                "threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )

        if metric == "f1":
            score = f1
            if score > best_val:
                best_val = score
                best_t = float(t)
        elif metric == "recall_at_precision" and min_precision is not None:
            if prec >= min_precision and rec > best_val:
                best_val = rec
                best_t = float(t)
        else:
            if f1 > best_val:
                best_val = f1
                best_t = float(t)

    return {
        "best_threshold": best_t,
        "best_value": float(best_val),
        "curve": curve,
    }
