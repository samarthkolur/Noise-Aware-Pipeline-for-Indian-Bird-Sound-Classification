"""
classifier.py — Lightweight MLP classification head for embeddings.
"""

from typing import List

import torch
import torch.nn as nn


class EmbeddingClassifier(nn.Module):
    """MLP classification head for fixed embeddings.

    Takes a 1D embedding (e.g., 1024D from BirdNET) and passes it through
    a series of linear layers to produce class logits. Supports both
    binary (num_classes=1) and multiclass outputs.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            input_dim: Dimensionality of the incoming embedding.
            num_classes: Number of output classes (1 for binary).
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_dim = h_dim

        # Final projection to logits
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Embedding tensor of shape (B, input_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.head(x)
