"""
attention.py — Attention pooling module for temporal aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learnable attention pooling over the time dimension.

    Takes a sequence of frame-level features and produces a single
    utterance-level representation via weighted averaging.
    """

    def __init__(self, feature_dim: int) -> None:
        """
        Args:
            feature_dim: Dimensionality of each frame-level input vector.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Frame-level features of shape (B, T, D).

        Returns:
            Utterance-level features of shape (B, D).
        """
        # Compute attention weights  (B, T, 1)
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum  (B, D)
        out = (x * attn_weights).sum(dim=1)
        return out
