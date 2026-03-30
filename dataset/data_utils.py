"""
data_utils.py — Train/val/test splits, label encoding, and class balancing.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def get_label_encoder(data_dir: Path) -> Dict[str, int]:
    """Create a species → integer label mapping from directory names.

    Args:
        data_dir: Root directory containing one sub-folder per species.

    Returns:
        Sorted dictionary mapping species name → integer label.
    """
    species = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return {name: idx for idx, name in enumerate(species)}


def create_splits(
    data_dir: Path,
    label_map: Dict[str, int],
    val_split: float = 0.15,
    test_split: float = 0.10,
    stratify: bool = True,
    seed: int = 42,
) -> Tuple[
    Tuple[List[Path], List[int]],
    Tuple[List[Path], List[int]],
    Tuple[List[Path], List[int]],
]:
    """Split feature files into train / val / test sets.

    Args:
        data_dir: Directory with species sub-folders of .pt files.
        label_map: Species → integer mapping.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        stratify: Whether to stratify by label.
        seed: Random seed.

    Returns:
        ((train_paths, train_labels),
         (val_paths,   val_labels),
         (test_paths,  test_labels))
    """
    all_paths: List[Path] = []
    all_labels: List[int] = []

    for species, label in label_map.items():
        species_dir = data_dir / species
        if not species_dir.exists():
            continue
        for pt_file in sorted(species_dir.glob("*.pt")):
            all_paths.append(pt_file)
            all_labels.append(label)

    all_labels_np = np.array(all_labels)
    stratify_arr = all_labels_np if stratify else None

    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=test_split,
        stratify=stratify_arr,
        random_state=seed,
    )

    # Second split: train vs val
    relative_val = val_split / (1 - test_split)
    tv_labels_np = np.array(train_val_labels)
    stratify_tv = tv_labels_np if stratify else None

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=relative_val,
        stratify=stratify_tv,
        random_state=seed,
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def compute_class_weights(labels: List[int], num_classes: int) -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        labels: List of integer labels.
        num_classes: Total number of classes.

    Returns:
        Array of shape (num_classes,) with class weights.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights /= weights.sum()  # normalise
    return weights * num_classes
