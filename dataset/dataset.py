"""
dataset.py — PyTorch Dataset and DataLoader for HDF5 embeddings.

Loads pre-extracted embeddings from the HDF5 storage layout produced by
the embedding pipeline and exposes them through standard PyTorch interfaces.

Supports:
  • Multi-class species labels  (species name → integer)
  • Binary bird / noise labels  (any species = 1, "noise" = 0)
  • Stratified train / val / test splits
  • Inverse-frequency class weighting for imbalanced data
  • Manifest-driven or directory-scan loading
  • Memory-mapped and in-memory modes
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from utils.logger import get_logger

logger = get_logger(__name__)

NOISE_LABEL_NAME = "noise"


# ════════════════════════════════════════════════════════════
#  Label encoder
# ════════════════════════════════════════════════════════════

class LabelEncoder:
    """Bidirectional mapping between species names and integer labels.

    If ``binary=True``, every species that is not ``"noise"`` gets label 1
    (bird) and ``"noise"`` gets label 0.
    """

    def __init__(self, species: List[str], binary: bool = False) -> None:
        self.binary = binary
        if binary:
            self._name2id: Dict[str, int] = {
                s: (0 if s == NOISE_LABEL_NAME else 1) for s in sorted(set(species))
            }
            self._id2name: Dict[int, str] = {0: NOISE_LABEL_NAME, 1: "bird"}
            self.num_classes = 2
        else:
            unique = sorted(set(species))
            self._name2id = {name: idx for idx, name in enumerate(unique)}
            self._id2name = {idx: name for name, idx in self._name2id.items()}
            self.num_classes = len(unique)

    def encode(self, name: str) -> int:
        return self._name2id[name]

    def decode(self, label: int) -> str:
        return self._id2name.get(label, "unknown")

    @property
    def name2id(self) -> Dict[str, int]:
        return dict(self._name2id)

    @property
    def id2name(self) -> Dict[int, str]:
        return dict(self._id2name)


# ════════════════════════════════════════════════════════════
#  Core Dataset
# ════════════════════════════════════════════════════════════

class EmbeddingDataset(Dataset):
    """PyTorch Dataset backed by in-memory numpy arrays.

    Each sample returns ``(embedding_tensor, label_int)``.

    Constructed via the classmethods:
      • ``from_directory``  — scan ``embeddings_dir/<species>/embeddings.h5``
      • ``from_manifest``   — read ``manifest.csv`` produced by the pipeline
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        filenames: List[str],
        species_names: List[str],
        label_encoder: LabelEncoder,
    ) -> None:
        assert len(embeddings) == len(labels) == len(filenames) == len(species_names)
        self.embeddings = embeddings        # (N, D) float32
        self.labels = labels                # (N,)   int64
        self.filenames = filenames          # source WAV paths
        self.species_names = species_names  # per-sample species string
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        emb = torch.from_numpy(self.embeddings[idx])
        label = int(self.labels[idx])
        return emb, label

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def num_classes(self) -> int:
        return self.label_encoder.num_classes

    # ── Classmethods ───────────────────────────────────────

    @classmethod
    def from_directory(
        cls,
        embeddings_dir: Path,
        binary: bool = False,
    ) -> "EmbeddingDataset":
        """Load all species embeddings from the HDF5 directory layout.

        Args:
            embeddings_dir: Root dir containing ``<species>/embeddings.h5``.
            binary: If True, map all species → 1 (bird), "noise" → 0.
        """
        embeddings_dir = Path(embeddings_dir)
        all_embs: List[np.ndarray] = []
        all_fnames: List[str] = []
        all_species: List[str] = []

        for species_dir in sorted(embeddings_dir.iterdir()):
            h5_path = species_dir / "embeddings.h5"
            if not h5_path.exists():
                continue
            species = species_dir.name
            with h5py.File(str(h5_path), "r") as h5f:
                embs = h5f["embeddings"][:]  # (N, D)
                fnames = [
                    fn.decode() if isinstance(fn, bytes) else fn
                    for fn in h5f["filenames"][:]
                ]
            all_embs.append(embs)
            all_fnames.extend(fnames)
            all_species.extend([species] * len(fnames))

        if not all_embs:
            raise FileNotFoundError(
                f"No embeddings.h5 files found under {embeddings_dir}"
            )

        emb_matrix = np.vstack(all_embs).astype(np.float32)
        encoder = LabelEncoder(all_species, binary=binary)
        labels = np.array(
            [encoder.encode(s) for s in all_species], dtype=np.int64
        )

        logger.info(
            f"Loaded {len(labels)} embeddings ({emb_matrix.shape[1]}D) "
            f"across {encoder.num_classes} classes from {embeddings_dir}"
        )

        return cls(emb_matrix, labels, all_fnames, all_species, encoder)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        binary: bool = False,
    ) -> "EmbeddingDataset":
        """Load embeddings using the manifest CSV as an index.

        The manifest provides HDF5 file paths and row indices, enabling
        selective or partial loading.
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"Empty manifest: {manifest_path}")

        # Group by HDF5 file to avoid re-opening
        h5_groups: Dict[str, List[Tuple[int, dict]]] = {}
        for row in rows:
            h5_path = row["hdf5_path"]
            h5_row = int(row["hdf5_row"])
            h5_groups.setdefault(h5_path, []).append((h5_row, row))

        all_embs: List[np.ndarray] = []
        all_fnames: List[str] = []
        all_species: List[str] = []

        for h5_path, entries in sorted(h5_groups.items()):
            with h5py.File(h5_path, "r") as h5f:
                emb_ds = h5f["embeddings"]
                for row_idx, row_meta in entries:
                    all_embs.append(emb_ds[row_idx])
                    all_fnames.append(row_meta["source_file"])
                    all_species.append(row_meta["species"])

        emb_matrix = np.stack(all_embs).astype(np.float32)
        encoder = LabelEncoder(all_species, binary=binary)
        labels = np.array(
            [encoder.encode(s) for s in all_species], dtype=np.int64
        )

        logger.info(
            f"Loaded {len(labels)} embeddings ({emb_matrix.shape[1]}D) "
            f"from manifest {manifest_path}"
        )

        return cls(emb_matrix, labels, all_fnames, all_species, encoder)


# ════════════════════════════════════════════════════════════
#  Splitting
# ════════════════════════════════════════════════════════════

@dataclass
class DataSplits:
    """Container for train / val / test index splits."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def create_splits(
    dataset: EmbeddingDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    stratify: bool = True,
    seed: int = 42,
) -> DataSplits:
    """Stratified train / val / test split over an EmbeddingDataset.

    Returns index arrays so the original dataset object is shared (no copy).
    """
    n = len(dataset)
    indices = np.arange(n)
    labels = dataset.labels

    stratify_arr = labels if stratify else None

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        stratify=stratify_arr,
        random_state=seed,
    )

    relative_val = val_frac / (1.0 - test_frac)
    tv_labels = labels[train_val_idx] if stratify else None

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=tv_labels,
        random_state=seed,
    )

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}  (total={n})"
    )

    return DataSplits(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


# ════════════════════════════════════════════════════════════
#  Class weights & sampler
# ════════════════════════════════════════════════════════════

def compute_class_weights(
    labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Inverse-frequency weights for imbalanced datasets.

    Returns:
        Array of shape (num_classes,) — can be passed directly to
        ``torch.nn.CrossEntropyLoss(weight=...)``.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    return (weights * num_classes).astype(np.float32)


def make_weighted_sampler(
    labels: np.ndarray, indices: Optional[np.ndarray] = None
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for a (sub)set of labels.

    Useful for oversampling minority classes during training.
    """
    if indices is not None:
        subset_labels = labels[indices]
    else:
        subset_labels = labels
        indices = np.arange(len(labels))

    class_counts = np.bincount(subset_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    sample_weights = class_weights[subset_labels]

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ════════════════════════════════════════════════════════════
#  DataLoader factory
# ════════════════════════════════════════════════════════════

def build_dataloaders(
    cfg: dict,
    dataset: Optional[EmbeddingDataset] = None,
    binary: bool = False,
    balanced_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """One-call factory: load embeddings → split → build DataLoaders.

    Args:
        cfg: Pipeline config dict.
        dataset: Pre-built dataset. If None, loaded from ``cfg['data']['embeddings_dir']``.
        binary: Use binary (bird/noise) labels.
        balanced_train: Use weighted random sampling for the train loader.

    Returns:
        (train_loader, val_loader, test_loader, label_encoder)
    """
    if dataset is None:
        embeddings_dir = Path(cfg["data"]["embeddings_dir"])
        manifest = embeddings_dir / "manifest.csv"
        if manifest.exists():
            dataset = EmbeddingDataset.from_manifest(manifest, binary=binary)
        else:
            dataset = EmbeddingDataset.from_directory(embeddings_dir, binary=binary)

    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})

    splits = create_splits(
        dataset,
        val_frac=ds_cfg.get("val_split", 0.15),
        test_frac=ds_cfg.get("test_split", 0.10),
        stratify=ds_cfg.get("stratify", True),
        seed=cfg.get("project", {}).get("seed", 42),
    )

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = cfg.get("embedding", {}).get("num_workers", 4)

    # Train loader (optionally balanced)
    train_subset = Subset(dataset, splits.train_idx)
    if balanced_train:
        sampler = make_weighted_sampler(dataset.labels, splits.train_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # Val / Test loaders (no shuffle)
    val_loader = DataLoader(
        Subset(dataset, splits.val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, splits.test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready — batch_size={batch_size}, "
        f"train={len(train_subset)}, val={len(splits.val_idx)}, "
        f"test={len(splits.test_idx)}"
    )

    return train_loader, val_loader, test_loader, dataset.label_encoder
