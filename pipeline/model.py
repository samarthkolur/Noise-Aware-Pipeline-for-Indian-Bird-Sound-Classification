"""FocalMLP classifier and BirdAutoencoder OOD gate (design.md §6.3, §6.4)."""

from __future__ import annotations

import torch
from torch import nn

from pipeline.config import AutoencoderConfig, MLPConfig


class FocalMLP(nn.Module):
    """1024 -> 512 -> BN+ReLU+Dropout -> 256 -> BN+ReLU+Dropout -> 1 -> Sigmoid."""

    def __init__(self, embedding_dim: int, config: MLPConfig):
        super().__init__()
        dims = [embedding_dim, *config.hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BirdAutoencoder(nn.Module):
    """Symmetric bottleneck AE: 1024 -> 512 -> 128 -> 512 -> 1024, MSE reconstruction."""

    def __init__(self, embedding_dim: int, config: AutoencoderConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.bottleneck_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error, shape (batch,)."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=-1)
