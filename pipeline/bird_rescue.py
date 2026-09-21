"""Bird Rescue: MLP-based second-chance classifier (design.md §6.1, §20).

Segments that Noise Segregation V2 + Bird Guard route to the noise class get
one more check: a small binary MLP over their BirdNET embedding. If
P(bird) >= rescue_threshold, the segment is rescued back to the bird class
before the main Focal-loss MLP / AE gate stages run.

This is a genuinely separate, smaller model from the main FocalMLP (design.md
config.yaml: bird_rescue.hidden_dims, default [128, 64]) — its job is a cheap
recheck, not the primary classification decision.
"""

from __future__ import annotations

import torch
from torch import nn

from pipeline.config import BirdRescueConfig


class BirdRescueMLP(nn.Module):
    """Linear(embedding_dim, h1) -> ReLU -> Linear(h1, h2) -> ReLU -> Linear(h2, 1) -> Sigmoid."""

    def __init__(self, embedding_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [embedding_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_bird_rescue(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config: BirdRescueConfig,
    embedding_dim: int,
    epochs: int = 30,
    lr: float = 1e-3,
) -> BirdRescueMLP:
    """Train the Bird Rescue MLP on (embedding, is_bird) pairs.

    Intended to be trained on BirdNET embeddings of segments that Noise
    Segregation V2 + Bird Guard initially routed to the noise class, with
    ground-truth labels from the manifest.
    """
    model = BirdRescueMLP(embedding_dim, config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(embeddings)
        loss = loss_fn(preds, labels.float())
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def rescue_predict(
    embedding: torch.Tensor, model: BirdRescueMLP, rescue_threshold: float
) -> tuple[bool, float]:
    """Return (rescued, probability) for a single embedding."""
    model.eval()
    prob = float(model(embedding.unsqueeze(0)).item())
    return prob >= rescue_threshold, prob
