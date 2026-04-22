"""
autoencoder.py — Symmetric bottleneck autoencoder for embedding reconstruction.

Trained on bird-only embeddings so that noise samples produce higher
reconstruction error, enabling a gating stage before the classifier.

Architecture:
    Encoder: 1024 → 512 → 256 → 128
    Decoder: 128 → 256 → 512 → 1024
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class EmbeddingAutoencoder(nn.Module):
    """Symmetric autoencoder for 1024-D BirdNET embeddings.

    Forward returns ``(reconstructed, bottleneck)`` so callers can inspect
    the latent representation if needed.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] | None = None,
        latent_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """If ``hidden_dims`` is omitted, it defaults to ``[512, 256, latent_dim]`` or ``128``."""
        super().__init__()
        if hidden_dims is None:
            if latent_dim is not None:
                hidden_dims = [512, 256, int(latent_dim)]
            else:
                hidden_dims = [512, 256, 128]

        # ── Encoder ────────────────────────────────────────
        enc_layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror) ──────────────────────────────
        dec_layers: List[nn.Module] = []
        rev_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        in_dim = hidden_dims[-1]
        for i, h_dim in enumerate(rev_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim))
            if i < len(rev_dims) - 1:
                # Hidden layers get BN + ReLU + Dropout
                dec_layers.extend([
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ])
            # Final layer has no activation (reconstruct raw embedding)
            in_dim = h_dim
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Embedding tensor of shape ``(B, input_dim)``.

        Returns:
            ``(reconstructed, bottleneck)`` — both shape ``(B, D)``.
        """
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck

    @staticmethod
    def compute_reconstruction_error(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample MSE between original and reconstructed embeddings.

        Args:
            original: ``(B, D)``
            reconstructed: ``(B, D)``

        Returns:
            ``(B,)`` tensor of per-sample MSE values.
        """
        return ((original - reconstructed) ** 2).mean(dim=1)
