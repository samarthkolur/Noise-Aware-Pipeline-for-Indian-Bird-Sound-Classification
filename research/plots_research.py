"""Research visualizations: PCA, t-SNE, AE errors, feature importance, ROC/PR, confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    PCA = None
    TSNE = None

try:
    from sklearn.metrics import (
        auc,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )
except ImportError:
    auc = None
    confusion_matrix = None
    precision_recall_curve = None
    roc_curve = None


# ═══════════════════════════════════════════════════════════
#  PCA / t-SNE
# ═══════════════════════════════════════════════════════════

def plot_pca_tsne(
    embs: np.ndarray,
    y_true: np.ndarray,
    ood_mask: Optional[np.ndarray],
    out_pca: Path,
    out_tsne: Path,
    path_noise_proxy: Optional[np.ndarray] = None,
    max_points_tsne: int = 800,
    random_seed: int = 42,
    skip_tsne: bool = False,
) -> None:
    """Two-panel PCA (bird vs noise; routing proxy) and t-SNE with OOD highlights."""
    out_pca.parent.mkdir(parents=True, exist_ok=True)
    n = len(embs)
    if PCA is None:
        return

    pca = PCA(n_components=2, random_state=random_seed)
    Z = pca.fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = np.where(y_true == 1, "#2ca02c", "#1f77b4")
    axes[0].scatter(Z[:, 0], Z[:, 1], c=colors, alpha=0.6, s=12, edgecolors="none")
    axes[0].set_title("PCA — ground truth (green=bird, blue=noise)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    if path_noise_proxy is not None:
        c2 = np.where(path_noise_proxy, "#d62728", "#9467bd")
        axes[1].scatter(Z[:, 0], Z[:, 1], c=c2, alpha=0.55, s=12, edgecolors="none")
        axes[1].set_title("PCA — path proxy (red=noise folder / routed noise)")
    elif ood_mask is not None:
        c2 = np.where(ood_mask, "#ff7f0e", colors)
        axes[1].scatter(Z[:, 0], Z[:, 1], c=c2, alpha=0.55, s=12, edgecolors="none")
        axes[1].set_title("PCA — OOD gate (orange=AE rejected)")
    else:
        axes[1].scatter(Z[:, 0], Z[:, 1], c=colors, alpha=0.6, s=12, edgecolors="none")
        axes[1].set_title("PCA — duplicate view")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_pca, dpi=150)
    plt.close(fig)

    if skip_tsne or TSNE is None or n < 10:
        return

    rng = np.random.RandomState(random_seed)
    if n > max_points_tsne:
        sel = rng.choice(n, size=max_points_tsne, replace=False)
    else:
        sel = np.arange(n)

    perplexity = min(30, max(5, len(sel) // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_seed,
        init="pca",
        learning_rate="auto",
    )
    Zt = tsne.fit_transform(embs[sel])
    yt = y_true[sel]
    ood_s = ood_mask[sel] if ood_mask is not None else None

    fig, ax = plt.subplots(figsize=(7, 6))
    col = np.where(yt == 1, "#2ca02c", "#1f77b4")
    ax.scatter(Zt[:, 0], Zt[:, 1], c=col, alpha=0.65, s=14, edgecolors="none")
    if ood_s is not None and ood_s.any():
        ax.scatter(
            Zt[ood_s, 0],
            Zt[ood_s, 1],
            facecolors="none",
            edgecolors="#ff7f0e",
            s=45,
            linewidths=1.2,
            label="AE OOD",
        )
        ax.legend()
    ax.set_title(f"t-SNE (n={len(sel)}, perplexity={perplexity})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out_tsne, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  AE reconstruction error histogram
# ═══════════════════════════════════════════════════════════

def plot_ae_histogram(
    recon_errors: np.ndarray,
    tau: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(recon_errors, bins=50, color="#69b3a2", edgecolor="white", alpha=0.9)
    ax.axvline(tau, color="#d62728", linestyle="--", linewidth=2, label=f"τ_AE = {tau:.5f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder reconstruction error (test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  MLP feature importance (gradient × input + weight magnitude)
# ═══════════════════════════════════════════════════════════

def plot_feature_importance_mlp(
    model: torch.nn.Module,
    sample_embs: np.ndarray,
    device: torch.device,
    out_path: Path,
    n_show: int = 64,
    top_k_annotate: int = 10,
) -> None:
    """Gradient·input importance averaged over a batch + first-layer weight magnitude.

    Top-K most important dimensions are annotated on the plot.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    x = torch.from_numpy(sample_embs[:256]).to(device, dtype=torch.float32)
    x.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x).squeeze(-1)
    loss = logits.sum()
    loss.backward()
    g = x.grad.detach().abs().mean(dim=0).cpu().numpy()

    # First linear weight column L1 norm
    w0 = model.head[0].weight.detach().abs().mean(dim=0).cpu().numpy()
    combined = 0.5 * (g / (g.max() + 1e-12) + w0 / (w0.max() + 1e-12))

    D = len(combined)
    if D > n_show:
        idx = np.linspace(0, D - 1, n_show, dtype=int)
        xc = idx
        yv = combined[idx]
    else:
        xc = np.arange(D)
        yv = combined

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(xc, yv, color="#4c72b0", width=0.8)
    ax.set_xlabel("Embedding dimension index (subsampled)")
    ax.set_ylabel("Normalized importance")
    ax.set_title("MLP attribution: mean |grad·input| + first-layer |W| (normalized)")

    # Annotate top-K most important dimensions
    top_k_idx = np.argsort(combined)[-top_k_annotate:][::-1]
    for rank, dim_idx in enumerate(top_k_idx):
        ax.annotate(
            f"d{dim_idx}",
            xy=(dim_idx, combined[dim_idx]),
            xytext=(0, 8 + rank * 2),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            color="#d62728",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Confusion matrix heatmap (publication quality)
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    out_path: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    """Side-by-side confusion matrix heatmaps for multiple systems.

    Args:
        y_true: Ground truth labels (0/1).
        predictions: Dict mapping system name → prediction array.
        out_path: Output file path.
        class_names: Labels for classes (default: ["Noise", "Bird"]).
    """
    if confusion_matrix is None:
        return
    if class_names is None:
        class_names = ["Noise", "Bird"]

    n_systems = len(predictions)
    fig, axes = plt.subplots(1, n_systems, figsize=(5 * n_systems, 4.5))
    if n_systems == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        # Normalize for display
        cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)

        # Annotate each cell with count + percentage
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  ROC and Precision-Recall curves
# ═══════════════════════════════════════════════════════════

def plot_roc_pr_curves(
    y_true: np.ndarray,
    system_probs: Dict[str, np.ndarray],
    out_roc: Path,
    out_pr: Path,
) -> Dict[str, Dict[str, float]]:
    """Overlay ROC and PR curves for multiple systems.

    Args:
        y_true: Binary labels (0/1).
        system_probs: Dict mapping system name → probability scores.
        out_roc: Output path for ROC curve.
        out_pr: Output path for PR curve.

    Returns:
        Dict of {system_name: {"roc_auc": float, "pr_auc": float}}.
    """
    if roc_curve is None or auc is None:
        return {}

    out_roc.parent.mkdir(parents=True, exist_ok=True)
    out_pr.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    auc_results: Dict[str, Dict[str, float]] = {}

    # ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    for idx, (name, probs) in enumerate(system_probs.items()):
        color = colors[idx % len(colors)]
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc_val = float(auc(fpr, tpr))

        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC={roc_auc_val:.4f})")
        auc_results[name] = {"roc_auc": roc_auc_val}

    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curves — Test Split Comparison", fontsize=13, pad=12)
    ax_roc.legend(fontsize=10, loc="lower right")
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(out_roc, dpi=150)
    plt.close(fig_roc)

    # Precision-Recall curves
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    for idx, (name, probs) in enumerate(system_probs.items()):
        color = colors[idx % len(colors)]
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, probs)
        pr_auc_val = float(auc(recall_arr, precision_arr))

        ax_pr.plot(recall_arr, precision_arr, color=color, linewidth=2,
                   label=f"{name} (AUC={pr_auc_val:.4f})")
        auc_results[name]["pr_auc"] = pr_auc_val

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curves — Test Split Comparison", fontsize=13, pad=12)
    ax_pr.legend(fontsize=10, loc="lower left")
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.02])
    ax_pr.grid(alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(out_pr, dpi=150)
    plt.close(fig_pr)

    return auc_results
