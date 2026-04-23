"""
Embedding Visualisation — PCA and t-SNE

Produces 2-D scatter plots of BirdNET 1024-D embeddings.
Points are coloured by bird (blue) vs noise (red).

PCA plot
--------
A 2-D PCA is fitted on TRAIN embeddings only, then val embeddings are
transformed with the fitted projection.  No val data influences the axes.

t-SNE plot
----------
t-SNE has no "fit/transform" split; it is always optimised on the data
being displayed.  Val embeddings are used directly (standard practice).

Outputs
-------
results/plots/pca_plot.png
results/plots/tsne_plot.png

Usage
-----
    python evaluation/embedding_viz.py
    python evaluation/embedding_viz.py --max-samples 2000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MANIFEST  = Path("data/embeddings/manifest.csv")
PLOTS_DIR = Path("results/plots")


# ---------------------------------------------------------------------------
# Embedding loaders
# ---------------------------------------------------------------------------
def _load_split(manifest: pd.DataFrame, split: str,
                max_samples: int | None = None,
                seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a given split. y: 1=bird, 0=noise."""
    rows = manifest[manifest["split"] == split].reset_index(drop=True)
    if max_samples and len(rows) > max_samples:
        rows = rows.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    embs, labels = [], []
    for _, row in rows.iterrows():
        p = Path(row["path"])
        if p.exists():
            embs.append(np.load(str(p)).astype("float32"))
            labels.append(int(row["binary_label"]))

    if not embs:
        raise RuntimeError(f"[VIZ] No embeddings loaded for split='{split}'.")
    return np.stack(embs), np.array(labels)


# ---------------------------------------------------------------------------
# Scatter helper
# ---------------------------------------------------------------------------
def _scatter(ax, X2d: np.ndarray, y: np.ndarray, title: str) -> None:
    bird_mask  = y == 1
    noise_mask = y == 0
    ax.scatter(X2d[bird_mask,  0], X2d[bird_mask,  1],
               c="steelblue", alpha=0.35, s=6,
               label=f"Bird  (n={bird_mask.sum()})")
    ax.scatter(X2d[noise_mask, 0], X2d[noise_mask, 1],
               c="tomato", alpha=0.65, s=10,
               label=f"Noise (n={noise_mask.sum()})")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)


# ---------------------------------------------------------------------------
# PCA — fit on TRAIN, transform val
# ---------------------------------------------------------------------------
def make_pca_plot(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray,   y_val: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)
    X2d_val = pca.transform(X_val)
    evr     = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, X2d_val, y_val,
             f"PCA of BirdNET Embeddings — Validation Set\n"
             f"(axes fitted on train data only)  "
             f"PC1={evr[0]:.1%}  PC2={evr[1]:.1%}")
    fig.tight_layout()
    out = PLOTS_DIR / "pca_plot.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"[VIZ] PCA plot saved : {out}")


# ---------------------------------------------------------------------------
# t-SNE — fit on val (unavoidable; t-SNE has no transform step)
# ---------------------------------------------------------------------------
def make_tsne_plot(X_val: np.ndarray, y_val: np.ndarray) -> None:
    perp = min(30, len(X_val) - 1)
    tsne = TSNE(n_components=2, perplexity=perp,
                random_state=42, max_iter=1000, init="pca")
    X2d = tsne.fit_transform(X_val)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, X2d, y_val,
             "t-SNE of BirdNET Embeddings — Validation Set")
    fig.tight_layout()
    out = PLOTS_DIR / "tsne_plot.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"[VIZ] t-SNE plot saved : {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def visualise(max_samples: int = 2000) -> None:
    if not MANIFEST.exists():
        print(f"[VIZ] Manifest not found: {MANIFEST}. Run extract_embeddings.py first.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)

    print("[VIZ] Loading train embeddings (for PCA fit) …")
    X_train, y_train = _load_split(manifest, "train", max_samples=max_samples)
    print(f"[VIZ] Train: {len(X_train)} embeddings  "
          f"(bird={y_train.sum()}, noise={(y_train==0).sum()})")

    print("[VIZ] Loading val embeddings …")
    X_val, y_val = _load_split(manifest, "val", max_samples=max_samples)
    print(f"[VIZ] Val  : {len(X_val)} embeddings  "
          f"(bird={y_val.sum()}, noise={(y_val==0).sum()})")

    print("[VIZ] Running PCA (fit on train, transform val) …")
    make_pca_plot(X_train, y_train, X_val, y_val)

    print("[VIZ] Running t-SNE on val embeddings …")
    make_tsne_plot(X_val, y_val)

    print("[VIZ] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PCA + t-SNE embedding plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max samples per split to plot.")
    args = parser.parse_args()
    visualise(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
