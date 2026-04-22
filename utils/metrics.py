"""
Shared classification metrics (binary + multiclass) and gated-routing helpers.

Single source of truth for scripts and ``training.metrics`` re-exports.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from inference.prediction_api import gated_three_class_pred_tensor


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

    prec_bird = precision_score(y, preds, pos_label=1, zero_division=0)
    rec_bird = recall_score(y, preds, pos_label=1, zero_division=0)
    prec_noise = precision_score(y, preds, pos_label=0, zero_division=0)
    rec_noise = recall_score(y, preds, pos_label=0, zero_division=0)

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


def binary_preds_uncertain_as_bird(probs: np.ndarray, low_threshold: float) -> np.ndarray:
    """Match inference routing: noise only if prob <= low; uncertain and clean_bird count as bird."""
    return (probs > low_threshold).astype(np.int64)


def compute_metrics_from_preds(labels_np: np.ndarray, preds_np: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1 for binary 0/1 arrays."""
    acc = accuracy_score(labels_np, preds_np)
    prec = precision_score(labels_np, preds_np, average="binary", zero_division=0)
    rec = recall_score(labels_np, preds_np, average="binary", zero_division=0)
    f1 = f1_score(labels_np, preds_np, average="binary", zero_division=0)
    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
    }


def confusion_rates_binary(labels_np: np.ndarray, preds_np: np.ndarray) -> dict:
    """TN, FP, FN, TP, FPR, FNR for binary classification."""
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def metrics_routing_uncertain_as_bird(
    logits: torch.Tensor,
    labels: torch.Tensor,
    low_threshold: float,
) -> dict:
    """Evaluate with pred_bird = (sigmoid(logits) > low_threshold)."""
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np = binary_preds_uncertain_as_bird(probs, low_threshold)
    m = compute_metrics_from_preds(labels_np, preds_np)
    cr = confusion_rates_binary(labels_np, preds_np)
    return {**m, **cr, "low_threshold": float(low_threshold)}


def gated_three_class_predictions(
    recon_errors: np.ndarray,
    tau: float,
    probs: np.ndarray,
    low_t: float,
    high_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """AE OOD gate then MLP three-band routing (matches ``inference.prediction_api``).

    Returns:
        pred: 0 noise, 1 bird, 2 uncertain (only when AE accepts and prob is between bands).
        ae_reject: bool array (OOD — routed to noise without MLP).
    """
    pred_t, ae_t = gated_three_class_pred_tensor(
        torch.as_tensor(probs, dtype=torch.float64),
        torch.as_tensor(recon_errors, dtype=torch.float64),
        float(tau),
        float(low_t),
        float(high_t),
    )
    return pred_t.detach().cpu().numpy().astype(np.int64), ae_t.detach().cpu().numpy().astype(
        bool
    )


def gated_pred_uncertain_as_bird(
    recon_errors: np.ndarray,
    tau: float,
    probs: np.ndarray,
    low_t: float,
) -> np.ndarray:
    """pred bird iff (pass AE) and prob > low_t; OOD → noise."""
    ae_reject = recon_errors > tau
    preds = np.zeros(len(probs), dtype=np.int64)
    for i in range(len(probs)):
        if ae_reject[i]:
            preds[i] = 0
        else:
            preds[i] = 1 if probs[i] > low_t else 0
    return preds


def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """TP, TN, FP, FN for binary 0/1 vectors (research / benchmarks)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Bird-positive binary metrics + counts (research benchmarks)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_total": int(len(y_true)),
    }


def metrics_from_confusion(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Accuracy / precision / recall / F1 / FPR / FNR from confusion counts (baseline scripts)."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def ae_reconstruction_threshold_percentile(
    reconstruction_errors: np.ndarray, percentile: float
) -> float:
    """τ_AE style threshold: percentile of per-sample reconstruction errors."""
    arr = np.asarray(reconstruction_errors, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.01
    return float(np.percentile(arr, float(percentile)))
