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

# Legacy exports (spectrogram-based)
from .bird_dataset import BirdAudioDataset
from .data_utils import get_label_encoder

__all__ = [
    # New embedding-based
    "EmbeddingDataset",
    "LabelEncoder",
    "DataSplits",
    "create_splits",
    "compute_class_weights",
    "make_weighted_sampler",
    "build_dataloaders",
    # Legacy
    "BirdAudioDataset",
    "get_label_encoder",
]
