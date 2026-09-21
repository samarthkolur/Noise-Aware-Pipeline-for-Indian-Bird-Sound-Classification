import csv

import numpy as np
import pytest

from pipeline.cache import EmbeddingCache
from pipeline.dataset import BIRD_LABEL, NOISE_LABEL, EmbeddingDataset, make_weighted_sampler


@pytest.fixture()
def manifest_and_cache(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    rows = [
        {"segment_id": "bird_1", "class": "bird", "split": "train"},
        {"segment_id": "bird_2", "class": "bird", "split": "train"},
        {"segment_id": "noise_1", "class": "noise", "split": "train"},
        {"segment_id": "bird_3", "class": "bird", "split": "val"},
        {"segment_id": "noise_2", "class": "noise", "split": "test"},
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)

    cache = EmbeddingCache(tmp_path / "embeddings.h5")
    rng = np.random.default_rng(0)
    for row in rows:
        cache.write(row["segment_id"], rng.normal(size=1024).astype(np.float32))

    return manifest_path, cache


def test_train_split_has_expected_rows(manifest_and_cache):
    manifest_path, cache = manifest_and_cache
    ds = EmbeddingDataset(manifest_path, cache, split="train")
    assert len(ds) == 3
    assert set(ds.segment_ids) == {"bird_1", "bird_2", "noise_1"}


def test_labels_match_class_column(manifest_and_cache):
    manifest_path, cache = manifest_and_cache
    ds = EmbeddingDataset(manifest_path, cache, split="train")
    for seg_id, label in zip(ds.segment_ids, ds.labels, strict=True):
        expected = BIRD_LABEL if seg_id.startswith("bird") else NOISE_LABEL
        assert label == expected


def test_getitem_returns_l2_normalized_embedding(manifest_and_cache):
    manifest_path, cache = manifest_and_cache
    ds = EmbeddingDataset(manifest_path, cache, split="train")
    embedding, label = ds[0]
    assert embedding.shape == (1024,)
    norm = float((embedding**2).sum() ** 0.5)
    assert abs(norm - 1.0) < 1e-4
    assert label in (0.0, 1.0)


def test_load_all_shapes(manifest_and_cache):
    manifest_path, cache = manifest_and_cache
    ds = EmbeddingDataset(manifest_path, cache, split="train")
    embeddings, labels = ds.load_all()
    assert embeddings.shape == (3, 1024)
    assert labels.shape == (3,)


def test_bird_mask(manifest_and_cache):
    manifest_path, cache = manifest_and_cache
    ds = EmbeddingDataset(manifest_path, cache, split="train")
    mask = ds.bird_mask()
    assert mask.sum() == 2


def test_empty_split_returns_empty(manifest_and_cache):
    manifest_path, cache = manifest_and_cache
    ds = EmbeddingDataset(manifest_path, cache, split="nonexistent")
    assert len(ds) == 0
    embeddings, labels = ds.load_all()
    assert embeddings.shape[0] == 0


def test_make_weighted_sampler_favors_minority_class():
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # 9 bird, 1 noise
    sampler = make_weighted_sampler(labels)
    assert sampler.num_samples == len(labels)
    weights = sampler.weights.numpy()
    assert weights[-1] > weights[0]  # the single noise example gets more weight
