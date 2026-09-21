"""ROC/PR curves, confusion matrices, AE MSE histogram, PCA/t-SNE plots
(design.md §9, Figs. 3-8 equivalents)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str = "ROC Curve"
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str = "PR Curve"
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Confusion Matrix"
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["noise", "bird"], ax=ax)
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ae_mse_histogram(mse_values: np.ndarray, tau_ae: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mse_values, bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(tau_ae, color="red", linestyle="--", label=f"tau_AE={tau_ae:.4f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("AE Reconstruction Error Distribution")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_projection(
    embeddings: np.ndarray, labels: np.ndarray, out_path: Path, method: str = "pca"
) -> None:
    if method == "pca":
        proj = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        title = "PCA Projection of BirdNET Embeddings"
    elif method == "tsne":
        proj = TSNE(n_components=2, random_state=42, init="pca").fit_transform(embeddings)
        title = "t-SNE Projection of BirdNET Embeddings"
    else:
        raise ValueError(f"Unknown projection method: {method}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, color, name in [(1, "#4C72B0", "bird"), (0, "#DD8452", "noise")]:
        mask = labels == label
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.6, color=color, label=name)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
