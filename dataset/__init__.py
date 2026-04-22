"""dataset — PyTorch Dataset and DataLoader utilities for embeddings."""

from .dataset import (
    EmbeddingDataset,
    LabelEncoder,
    DataSplits,
    create_splits,
    compute_class_weights,
    make_weighted_sampler,
    build_dataloaders,
)

__all__ = [
    "EmbeddingDataset",
    "LabelEncoder",
    "DataSplits",
    "create_splits",
    "compute_class_weights",
    "make_weighted_sampler",
    "build_dataloaders",
]
