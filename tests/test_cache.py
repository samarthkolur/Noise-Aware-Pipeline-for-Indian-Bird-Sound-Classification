import numpy as np
import pytest

from pipeline.cache import EmbeddingCache


@pytest.fixture()
def cache(tmp_path):
    return EmbeddingCache(tmp_path / "embeddings.h5")


def test_write_then_read_roundtrip(cache):
    embedding = np.random.default_rng(0).normal(size=1024).astype(np.float32)
    cache.write("seg_001", embedding)
    result = cache.read("seg_001")
    assert result.shape == (1024,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, embedding)


def test_contains_reflects_written_keys(cache):
    assert cache.contains("seg_001") is False
    cache.write("seg_001", np.zeros(1024, dtype=np.float32))
    assert cache.contains("seg_001") is True


def test_write_is_idempotent_does_not_overwrite(cache):
    first = np.ones(1024, dtype=np.float32)
    second = np.zeros(1024, dtype=np.float32)
    cache.write("seg_001", first)
    cache.write("seg_001", second)  # should be a no-op
    np.testing.assert_allclose(cache.read("seg_001"), first)


def test_write_batch_skips_already_cached(cache):
    cache.write("seg_a", np.ones(1024, dtype=np.float32))
    ids = ["seg_a", "seg_b", "seg_c"]
    embeddings = np.stack([np.full(1024, i, dtype=np.float32) for i in range(3)])
    n_written = cache.write_batch(ids, embeddings)
    assert n_written == 2  # seg_a already existed
    np.testing.assert_allclose(cache.read("seg_a"), np.ones(1024, dtype=np.float32))


def test_missing_returns_uncached_ids(cache):
    cache.write("seg_a", np.zeros(1024, dtype=np.float32))
    result = cache.missing(["seg_a", "seg_b", "seg_c"])
    assert result == ["seg_b", "seg_c"]


def test_read_batch_shape(cache):
    ids = ["seg_a", "seg_b"]
    embeddings = np.stack([np.full(1024, i, dtype=np.float32) for i in range(2)])
    cache.write_batch(ids, embeddings)
    result = cache.read_batch(ids)
    assert result.shape == (2, 1024)


def test_rerunning_extraction_on_populated_cache_is_a_noop(cache):
    """Idempotency test mirroring Phase 4's exit criterion (design.md §10)."""
    ids = [f"seg_{i}" for i in range(5)]
    embeddings = np.random.default_rng(0).normal(size=(5, 1024)).astype(np.float32)
    first_written = cache.write_batch(ids, embeddings)
    second_written = cache.write_batch(ids, embeddings)
    assert first_written == 5
    assert second_written == 0
