import torch

from pipeline.attribution import (
    first_layer_weight_attribution,
    gradient_input_attribution,
    top_k_dimensions,
)
from pipeline.config import load_config
from pipeline.model import FocalMLP


def test_gradient_input_attribution_shape():
    cfg = load_config()
    model = FocalMLP(embedding_dim=1024, config=cfg.mlp)
    embeddings = torch.randn(8, 1024)
    attribution = gradient_input_attribution(model, embeddings)
    assert attribution.shape == (1024,)
    assert (attribution >= 0).all()


def test_first_layer_weight_attribution_shape():
    cfg = load_config()
    model = FocalMLP(embedding_dim=1024, config=cfg.mlp)
    attribution = first_layer_weight_attribution(model)
    assert attribution.shape == (1024,)
    assert (attribution >= 0).all()


def test_top_k_dimensions_returns_sorted_descending():
    import numpy as np

    attribution = np.array([0.1, 0.9, 0.3, 0.05, 0.7])
    top = top_k_dimensions(attribution, k=3)
    assert len(top) == 3
    assert top[0][0] == 1  # index of max value
    scores = [score for _, score in top]
    assert scores == sorted(scores, reverse=True)
