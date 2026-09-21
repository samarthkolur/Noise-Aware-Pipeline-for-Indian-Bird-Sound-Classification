"""PyTorch Dataset over the HDF5 embedding cache + manifest CSV (design.md §14)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from pipeline.cache import EmbeddingCache
from pipeline.embedding import l2_normalize

BIRD_LABEL = 1
NOISE_LABEL = 0


class EmbeddingDataset(Dataset):
    """Yields (embedding, label) pairs for one manifest split ('train'/'val'/'test')."""

    def __init__(self, manifest_path: str | Path, cache: EmbeddingCache, split: str):
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        self.cache = cache
        self.segment_ids = [r["segment_id"] for r in self.rows]
        self.labels = np.array(
            [BIRD_LABEL if r["class"] == "bird" else NOISE_LABEL for r in self.rows],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seg_id = self.segment_ids[idx]
        embedding = l2_normalize(self.cache.read(seg_id))
        label = self.labels[idx]
        return torch.from_numpy(embedding), torch.tensor(label, dtype=torch.float32)

    def bird_mask(self) -> np.ndarray:
        return self.labels == BIRD_LABEL

    def load_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the entire split into memory as (embeddings, labels) tensors.

        Embeddings are L2-normalised here (design.md §6.3: "L2-normalised at
        inference") so training and evaluation see identically preprocessed
        inputs — the model must never be trained on raw embeddings and
        evaluated on normalised ones (or vice versa).
        """
        if not self.segment_ids:
            return torch.zeros((0, 1)), torch.from_numpy(self.labels)
        raw = self.cache.read_batch(self.segment_ids)
        normalized = np.stack([l2_normalize(e) for e in raw])
        return torch.from_numpy(normalized), torch.from_numpy(self.labels)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """WeightedRandomSampler as the secondary class-imbalance guard (DD-006)."""
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
