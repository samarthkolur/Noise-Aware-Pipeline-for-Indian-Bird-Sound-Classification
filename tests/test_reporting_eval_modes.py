"""Tests for reporting/evaluation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from research.alignment import get_eval_arrays
from utils.thresholds import resolve_threshold_arg


def _write_manifest_dataset(tmp_path: Path, n_noise: int, n_bird: int, dim: int = 8) -> Path:
    rng = np.random.default_rng(123)
    emb_dir = tmp_path / "embeddings"
    noise_dir = emb_dir / "noise"
    bird_dir = emb_dir / "sparrow"
    noise_dir.mkdir(parents=True)
    bird_dir.mkdir(parents=True)

    h5_noise = noise_dir / "embeddings.h5"
    h5_bird = bird_dir / "embeddings.h5"
    with h5py.File(h5_noise, "w") as f:
        f.create_dataset("embeddings", data=rng.standard_normal((n_noise, dim)).astype(np.float32))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "filenames",
            data=np.array([f"n{i}.wav".encode() for i in range(n_noise)], dtype=dt),
        )
    with h5py.File(h5_bird, "w") as f:
        f.create_dataset("embeddings", data=rng.standard_normal((n_bird, dim)).astype(np.float32))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "filenames",
            data=np.array([f"b{i}.wav".encode() for i in range(n_bird)], dtype=dt),
        )

    manifest = emb_dir / "manifest.csv"
    rows = []
    for i in range(n_noise):
        rows.append(
            {
                "species": "noise",
                "source_file": f"/x/noise/n{i}.wav",
                "segment_index": str(i),
                "embedding_dim": str(dim),
                "model_name": "birdnet_v2.4",
                "sample_rate": "48000",
                "duration_sec": "3.0",
                "hdf5_path": str(h5_noise),
                "hdf5_row": str(i),
            }
        )
    for i in range(n_bird):
        rows.append(
            {
                "species": "sparrow",
                "source_file": f"/x/sparrow/b{i}.wav",
                "segment_index": str(i),
                "embedding_dim": str(dim),
                "model_name": "birdnet_v2.4",
                "sample_rate": "48000",
                "duration_sec": "3.0",
                "hdf5_path": str(h5_bird),
                "hdf5_row": str(i),
            }
        )
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return manifest


def test_get_eval_arrays_full_dataset_returns_all_rows(tmp_path: Path) -> None:
    manifest = _write_manifest_dataset(tmp_path, n_noise=6, n_bird=14)
    cfg = {
        "data": {"embeddings_dir": str(manifest.parent)},
        "dataset": {"val_split": 0.15, "test_split": 0.10, "stratify": True},
        "project": {"seed": 42},
    }

    embs, y_true, rows, eval_idx = get_eval_arrays(cfg, full_dataset=True)

    assert len(embs) == 20
    assert len(y_true) == 20
    assert len(rows) == 20
    assert np.array_equal(eval_idx, np.arange(20))
    assert int((y_true == 0).sum()) == 6
    assert int((y_true == 1).sum()) == 14


def test_resolve_threshold_arg_auto_reads_checkpoint_meta(tmp_path: Path) -> None:
    chkpt_dir = tmp_path / "checkpoints"
    chkpt_dir.mkdir()
    with open(chkpt_dir / "best_model_meta.json", "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": 0.61}, f)

    cfg = {"training": {"checkpoint_dir": str(chkpt_dir)}}
    assert resolve_threshold_arg("auto", cfg) == pytest.approx(0.61)
    assert resolve_threshold_arg("0.42", cfg) == pytest.approx(0.42)
