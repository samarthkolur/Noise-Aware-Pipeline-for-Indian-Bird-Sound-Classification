"""MLP feature attribution: gradient x input + first-layer |W| (design.md §14, Fig. 8)."""

from __future__ import annotations

import numpy as np
import torch

from pipeline.model import FocalMLP


def gradient_input_attribution(model: FocalMLP, embeddings: torch.Tensor) -> np.ndarray:
    """Per-dimension attribution scores via gradient x input, averaged over a batch.

    Returns an array of shape (embedding_dim,).
    """
    model.eval()
    embeddings = embeddings.clone().requires_grad_(True)
    preds = model(embeddings)
    preds.sum().backward()
    assert embeddings.grad is not None
    attributions = (embeddings.grad * embeddings).detach().abs().mean(dim=0)
    return attributions.numpy()


def first_layer_weight_attribution(model: FocalMLP) -> np.ndarray:
    """Sum of |W| across output units of the first linear layer, per input dimension."""
    first_linear = next(m for m in model.net if isinstance(m, torch.nn.Linear))
    weights = first_linear.weight.detach().abs().sum(dim=0)
    return weights.numpy()


def top_k_dimensions(attribution: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    order = np.argsort(-attribution)[:k]
    return [(int(i), float(attribution[i])) for i in order]
