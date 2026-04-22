"""Research visualizations: PCA, t-SNE, AE errors, feature importance."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def plot_feature_importance_mlp(
    model: torch.nn.Module,
    sample_embs: np.ndarray,
    device: torch.device,
    out_path: Path,
    n_show: int = 64,
) -> None:
    """Gradient·input importance averaged over a batch + first-layer weight magnitude."""
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

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(xc, yv, color="#4c72b0", width=0.8)
    ax.set_xlabel("Embedding dimension index (subsampled)")
    ax.set_ylabel("Normalized importance")
    ax.set_title("MLP interpretability: mean |grad·input| + first-layer |W| (normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
