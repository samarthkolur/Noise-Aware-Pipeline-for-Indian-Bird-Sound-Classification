"""
bird_dataset.py — PyTorch Dataset for loading processed spectrograms + labels.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class BirdAudioDataset(Dataset):
    """Dataset of pre-computed spectrogram features with species labels.

    Expects the following directory layout:
        processed_dir/
            SpeciesA/
                file1.pt
                file2.pt
            SpeciesB/
                ...

    Each .pt file contains a feature tensor (1, n_mels, T).
    """

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        label_map: Dict[str, int],
        transform: Optional[callable] = None,
    ) -> None:
        """
        Args:
            file_paths: List of paths to .pt feature files.
            labels: Corresponding integer labels.
            label_map: Mapping from species name → integer label.
            transform: Optional transform applied to features.
        """
        assert len(file_paths) == len(labels)
        self.file_paths = file_paths
        self.labels = labels
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feature = torch.load(str(self.file_paths[idx]), weights_only=True)
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def species_name(self, label: int) -> str:
        """Convert integer label back to species name."""
        return self.inv_label_map.get(label, "unknown")
