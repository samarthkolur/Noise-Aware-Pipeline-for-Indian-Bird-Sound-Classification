"""Align test-split embeddings, manifest rows, and BirdNET baseline keys."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from dataset.dataset import EmbeddingDataset, create_splits


def get_test_split_arrays(
    cfg: dict,
    manifest_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
    """Return (embeddings, y_true, manifest_rows, test_indices) for the test split.

    Order is identical to ``DataLoader(Subset(dataset, splits.test_idx))``.
    """
    embeddings_dir = Path(cfg["data"]["embeddings_dir"])
    manifest = manifest_path or (embeddings_dir / "manifest.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    dataset = EmbeddingDataset.from_manifest(manifest, binary=True)
    ds_cfg = cfg.get("dataset", {})
    splits = create_splits(
        dataset,
        val_frac=ds_cfg.get("val_split", 0.15),
        test_frac=ds_cfg.get("test_split", 0.10),
        stratify=ds_cfg.get("stratify", True),
        seed=cfg.get("project", {}).get("seed", 42),
    )

    test_idx = np.asarray(splits.test_idx, dtype=np.int64)
    embs = dataset.embeddings[test_idx].astype(np.float32)
    y = dataset.labels[test_idx].astype(np.int64)

    from compute_baseline_metrics import _get_ordered_manifest_rows

    ordered_rows = _get_ordered_manifest_rows(str(manifest))
    rows = [ordered_rows[int(i)] for i in test_idx]

    return embs, y, rows, test_idx


def rows_to_baseline_keys(rows: List[dict]) -> List[tuple]:
    from compute_baseline_metrics import _build_key_from_manifest_row

    return [_build_key_from_manifest_row(r) for r in rows]
