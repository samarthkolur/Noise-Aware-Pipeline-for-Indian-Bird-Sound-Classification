"""Focal loss for the binary bird-vs-noise MLP (design.md §6.3, DD-006).

Concentrates gradient signal on hard boundary cases, mitigating the severe
class imbalance between the bird and noise classes without aggressive
upsampling.
"""

from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss: -alpha * (1-p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(self.eps, 1.0 - self.eps)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = -alpha_t * (1.0 - p_t).pow(self.gamma) * torch.log(p_t)
        return loss.mean()
