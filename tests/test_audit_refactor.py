"""Tests for split determinism, AE thresholding, gated inference API, baseline keys."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset.dataset import EmbeddingDataset, create_splits
from inference.prediction_api import decision_binary_gated_single
from models.autoencoder import EmbeddingAutoencoder
from utils.metrics import ae_reconstruction_threshold_percentile


def test_splits_deterministic(tmp_path: Path) -> None:
    """Same seed + config → identical train/val/test index sets."""
    rng = np.random.default_rng(0)
    n = 60
    d = 8
    emb_dir = tmp_path / "embeddings"
    sp_noise = emb_dir / "noise"
    sp_bird = emb_dir / "sparrow"
    sp_noise.mkdir(parents=True)
    sp_bird.mkdir(parents=True)

    h5_noise = sp_noise / "embeddings.h5"
    h5_bird = sp_bird / "embeddings.h5"
    with h5py.File(h5_noise, "w") as f:
        f.create_dataset("embeddings", data=rng.standard_normal((30, d)).astype(np.float32))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "filenames",
            data=np.array([f"n{i}.wav".encode() for i in range(30)], dtype=dt),
        )
    with h5py.File(h5_bird, "w") as f:
        f.create_dataset("embeddings", data=rng.standard_normal((30, d)).astype(np.float32))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "filenames",
            data=np.array([f"b{i}.wav".encode() for i in range(30)], dtype=dt),
        )

    manifest = emb_dir / "manifest.csv"
    rows = []
    for i in range(30):
        rows.append(
            {
                "species": "noise",
                "source_file": f"/x/noise/n{i}.wav",
                "segment_index": str(i),
                "embedding_dim": str(d),
                "model_name": "birdnet_v2.4",
                "sample_rate": "48000",
                "duration_sec": "3.0",
                "hdf5_path": str(h5_noise),
                "hdf5_row": str(i),
            }
        )
    for i in range(30):
        rows.append(
            {
                "species": "sparrow",
                "source_file": f"/x/sparrow/b{i}.wav",
                "segment_index": str(i),
                "embedding_dim": str(d),
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

    cfg = {
        "dataset": {
            "val_split": 0.15,
            "test_split": 0.10,
            "stratify": True,
        },
        "project": {"seed": 42},
    }

    def run_once() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ds = EmbeddingDataset.from_manifest(manifest, binary=True)
        splits = create_splits(
            ds,
            val_frac=cfg["dataset"]["val_split"],
            test_frac=cfg["dataset"]["test_split"],
            stratify=cfg["dataset"]["stratify"],
            seed=cfg["project"]["seed"],
        )
        return splits.train_idx, splits.val_idx, splits.test_idx

    a1, a2, a3 = run_once()
    b1, b2, b3 = run_once()
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2) and np.array_equal(a3, b3)


def test_ae_threshold_percentile() -> None:
    err = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.8], dtype=np.float64)
    q = 90.0
    want = float(np.percentile(err, q))
    got = ae_reconstruction_threshold_percentile(err, q)
    assert abs(got - want) < 1e-9


def test_inference_gate_skips_mlp() -> None:
    """AE gate with τ=-1 forces OOD; classifier forward must not run (recon error ≥ 0)."""

    class CountMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

    ae = EmbeddingAutoencoder(input_dim=4, hidden_dims=[2], latent_dim=1)
    ae.eval()
    clf = CountMLP()
    emb = torch.randn(1, 4)
    decision, _sp, _p, _r, ae_rej = decision_binary_gated_single(
        ae,
        clf,
        emb,
        threshold=-1.0,
        high_threshold=0.7,
        low_threshold=0.3,
    )
    assert ae_rej is True
    assert decision == "noise"
    assert clf.calls == 0


def test_baseline_key_builder_roundtrip(tmp_path: Path) -> None:
    from compute_baseline_metrics import (
        _build_key_from_baseline_jsonl,
        _build_key_from_manifest_row,
        load_baseline_predictions,
    )

    row = {
        "species": "Somebird",
        "source_file": "iBC53/Somebird/recording_seg0006.wav",
        "segment_index": "6",
    }
    key_manifest = _build_key_from_manifest_row(row)

    jsonl = tmp_path / "baseline.jsonl"
    rec = {
        "source_file": "iBC53/Somebird/recording.wav",
        "confidence": 0.9,
        "start_sec": 18.0,
    }
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    bl = load_baseline_predictions(str(jsonl))
    key_jsonl = _build_key_from_baseline_jsonl(rec)

    assert key_manifest == ("somebird", "recording", 6)
    assert key_jsonl == ("somebird", "recording", 6)
    assert key_manifest in bl


def test_baseline_key_matches_processed_segment_path() -> None:
    """JSONL may cite processed ``*_segNNNN.wav`` paths; keys must match manifest."""
    from compute_baseline_metrics import _build_key_from_baseline_jsonl, _build_key_from_manifest_row

    row = {
        "species": "Merops",
        "source_file": "data/processed/Merops/clip_seg0003.wav",
        "segment_index": "3",
    }
    d = {
        "source_file": "data/processed/Merops/clip_seg0003.wav",
        "confidence": 0.8,
        "start_sec": 0.0,
    }
    assert _build_key_from_manifest_row(row) == ("merops", "clip", 3)
    assert _build_key_from_baseline_jsonl(d) == ("merops", "clip", 3)


def test_baseline_key_jsonl_segment_index_overrides_time() -> None:
    from compute_baseline_metrics import _build_key_from_baseline_jsonl, _build_key_from_manifest_row

    row = {
        "species": "Merops",
        "source_file": "data/processed/Merops/clip_seg0003.wav",
        "segment_index": "3",
    }
    d = {
        "source_file": "data/processed/Merops/clip_seg0003.wav",
        "confidence": 0.5,
        "start_sec": 0.0,
        "segment_index": 3,
    }
    assert _build_key_from_manifest_row(row) == _build_key_from_baseline_jsonl(d)


def test_baseline_key_case_insensitive_paths() -> None:
    from compute_baseline_metrics import _build_key_from_baseline_jsonl, _build_key_from_manifest_row

    row = {
        "species": "Merops",
        "source_file": "DATA/processed/MEROPS/CLIP_seg0003.wav",
        "segment_index": "3",
    }
    d = {
        "source_file": "data/processed/Merops/clip_seg0003.wav",
        "confidence": 0.1,
        "start_sec": 0.0,
    }
    assert _build_key_from_manifest_row(row) == _build_key_from_baseline_jsonl(d)


def test_baseline_key_noise_double_underscore_path() -> None:
    from compute_baseline_metrics import _build_key_from_baseline_jsonl, _build_key_from_manifest_row

    row = {
        "species": "noise",
        "source_file": "data/processed/noise/Luscinia__BR150355_seg0004.wav",
        "segment_index": "4",
    }
    d = {
        "source_file": "data/processed/noise/Luscinia__BR150355_seg0004.wav",
        "confidence": 0.3,
        "start_sec": 0.0,
    }
    assert _build_key_from_manifest_row(row) == ("luscinia", "br150355", 4)
    assert _build_key_from_baseline_jsonl(d) == ("luscinia", "br150355", 4)
