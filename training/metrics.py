"""Backward-compatible re-exports; canonical definitions live in ``utils.metrics``."""

from utils.metrics import (  # noqa: F401
    ae_reconstruction_threshold_percentile,
    binary_preds_uncertain_as_bird,
    compute_binary_per_class_metrics,
    compute_confusion_matrix,
    compute_metrics,
    compute_metrics_from_preds,
    confusion_rates_binary,
    find_optimal_threshold,
    gated_pred_uncertain_as_bird,
    gated_three_class_predictions,
    metrics_routing_uncertain_as_bird,
)

__all__ = [
    "ae_reconstruction_threshold_percentile",
    "binary_preds_uncertain_as_bird",
    "compute_binary_per_class_metrics",
    "compute_confusion_matrix",
    "compute_metrics",
    "compute_metrics_from_preds",
    "confusion_rates_binary",
    "find_optimal_threshold",
    "gated_pred_uncertain_as_bird",
    "gated_three_class_predictions",
    "metrics_routing_uncertain_as_bird",
]
