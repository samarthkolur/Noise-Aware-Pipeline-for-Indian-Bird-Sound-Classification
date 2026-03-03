"""
Evaluation Module

Compute classification metrics, generate visualizations (confusion matrix,
ROC curve, PR curve), and run ablation studies comparing different pipeline
configurations.
"""

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0=noise, 1=bird).
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional, needed for AUC).

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_bird": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_bird": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_bird": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr_on_noise": float(
            1 - precision_score(1 - y_true, 1 - y_pred, zero_division=0)
        ) if len(np.unique(y_true)) > 1 else 0.0,
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Print metrics in a readable format."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  Confusion Matrix:")
            cm = np.array(value)
            print(f"    Predicted →  Noise  Bird")
            print(f"    Noise      {cm[0][0]:6d} {cm[0][1]:5d}")
            print(f"    Bird       {cm[1][0]:6d} {cm[1][1]:5d}")
        else:
            print(f"  {key:20s}: {value:.4f}")
    print(f"{'='*60}\n")


def save_metrics(metrics: dict, filename: str = "metrics.json"):
    """Save metrics to JSON file."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Eval] Metrics saved to {path}")


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """Generate and save a confusion matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Noise", "Bird"],
        yticklabels=["Noise", "Bird"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Bird vs Noise — Confusion Matrix", fontsize=14)
    plt.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Eval] Confusion matrix saved to {path}")


def plot_roc_curve(y_true, y_prob, filename="roc_curve.png"):
    """Generate and save a ROC curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Eval] ROC curve saved to {path}")


def plot_pr_curve(y_true, y_prob, filename="pr_curve.png"):
    """Generate and save a Precision–Recall curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Eval] PR curve saved to {path}")


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    tag: str = "full_pipeline",
) -> dict:
    """
    Run full evaluation: compute metrics, print report, generate plots.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        tag: Prefix for output filenames.

    Returns:
        Metrics dictionary.
    """
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, title=f"Evaluation: {tag}")
    save_metrics(metrics, filename=f"{tag}_metrics.json")

    # Classification report
    print(classification_report(y_true, y_pred, target_names=["Noise", "Bird"]))

    # Plots
    plot_confusion_matrix(y_true, y_pred, filename=f"{tag}_confusion_matrix.png")

    if y_prob is not None and len(np.unique(y_true)) > 1:
        plot_roc_curve(y_true, y_prob, filename=f"{tag}_roc.png")
        plot_pr_curve(y_true, y_prob, filename=f"{tag}_pr.png")

    return metrics


def run_ablation_study(results: dict):
    """
    Compare metrics across different pipeline configurations.

    Args:
        results: Dict mapping config name to metrics dict.
            e.g. {"baseline": {...}, "no_ood": {...}, "full": {...}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(results.keys())
    metric_names = ["accuracy", "precision_bird", "recall_bird", "f1_bird"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))

    for i, metric in enumerate(metric_names):
        ax = axes[i] if len(metric_names) > 1 else axes
        values = [results[c].get(metric, 0) for c in configs]
        bars = ax.bar(configs, values, color=plt.cm.Set2(range(len(configs))))
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=9,
            )

    plt.suptitle("Ablation Study — Pipeline Configuration Comparison", fontsize=14)
    plt.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "ablation_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Eval] Ablation comparison saved to {path}")

    # Save as JSON table
    save_metrics(results, "ablation_results.json")


if __name__ == "__main__":
    print("Evaluation: Use run_pipeline.py --stage evaluate")
