"""
embedding_model.py — Wrapper around a CNN encoder for embedding extraction.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchaudio


class EmbeddingModel(nn.Module):
    """A CNN-based audio embedding encoder.

    Uses a lightweight convolutional stack that maps a mel-spectrogram
    to a fixed-size embedding vector.  Can later be swapped for a
    pre-trained backbone (e.g., BirdNET weights loaded via state_dict).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.embedding_dim = cfg["embedding"]["embedding_dim"]
        n_mels = cfg["features"]["n_mels"]

        # Simple convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Mel-spectrogram tensor of shape (B, 1, n_mels, T).

        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        features = self.encoder(x)
        embedding = self.projection(features)
        return embedding

    @classmethod
    def from_pretrained(
        cls, cfg: dict, checkpoint_path: str, device: Optional[str] = None
    ) -> "EmbeddingModel":
        """Load a model from a saved checkpoint."""
        model = cls(cfg)
        state_dict = torch.load(
            checkpoint_path,
            map_location=device or "cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model
