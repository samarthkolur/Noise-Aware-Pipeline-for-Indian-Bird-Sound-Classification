"""Align evaluation embeddings, manifest rows, and BirdNET baseline keys."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from dataset.dataset import EmbeddingDataset, create_splits


def get_eval_arrays(
    cfg: dict,
    manifest_path: Path | None = None,
    *,
    full_dataset: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
    """Return arrays for the chosen evaluation set.

    Order matches ``DataLoader(dataset_or_subset, shuffle=False)``.
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

    from compute_baseline_metrics import _get_ordered_manifest_rows

    ordered_rows = _get_ordered_manifest_rows(str(manifest))
    if full_dataset:
        eval_idx = np.arange(len(dataset), dtype=np.int64)
    else:
        eval_idx = np.asarray(splits.test_idx, dtype=np.int64)

    embs = dataset.embeddings[eval_idx].astype(np.float32)
    y = dataset.labels[eval_idx].astype(np.int64)
    rows = [ordered_rows[int(i)] for i in eval_idx]

    return embs, y, rows, eval_idx


def get_test_split_arrays(
    cfg: dict,
    manifest_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
    """Backward-compatible wrapper for callers that still want the test split."""
    return get_eval_arrays(cfg, manifest_path, full_dataset=False)


def rows_to_baseline_keys(rows: List[dict]) -> List[tuple]:
    from compute_baseline_metrics import _build_key_from_manifest_row

    return [_build_key_from_manifest_row(r) for r in rows]
