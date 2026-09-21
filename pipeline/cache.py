"""HDF5 embedding cache: segment_id -> (embedding_dim,) float32 array (DD-003).

Decouples BirdNET inference (the dominant per-segment cost) from classifier
training. Idempotent: re-running extraction skips segment_ids already present
in the cache. Thread-safe read; write is single-writer (no concurrent
training runs, per design.md §20).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class EmbeddingCache:
    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            with h5py.File(self.cache_path, "w"):
                pass  # create an empty file so `contains`/`keys` work before any write

    def contains(self, segment_id: str) -> bool:
        with h5py.File(self.cache_path, "r") as f:
            return segment_id in f

    def keys(self) -> set[str]:
        with h5py.File(self.cache_path, "r") as f:
            return set(f.keys())

    def write(self, segment_id: str, embedding: np.ndarray) -> None:
        """Write one embedding. No-op if segment_id is already cached (idempotent)."""
        with h5py.File(self.cache_path, "a") as f:
            if segment_id in f:
                return
            f.create_dataset(segment_id, data=embedding.astype(np.float32))

    def write_batch(self, segment_ids: list[str], embeddings: np.ndarray) -> int:
        """Write multiple embeddings, skipping any already cached. Returns count written."""
        written = 0
        with h5py.File(self.cache_path, "a") as f:
            for seg_id, emb in zip(segment_ids, embeddings, strict=True):
                if seg_id in f:
                    continue
                f.create_dataset(seg_id, data=emb.astype(np.float32))
                written += 1
        return written

    def read(self, segment_id: str) -> np.ndarray:
        with h5py.File(self.cache_path, "r") as f:
            return np.asarray(f[segment_id][()], dtype=np.float32)

    def read_batch(self, segment_ids: list[str]) -> np.ndarray:
        with h5py.File(self.cache_path, "r") as f:
            return np.stack([np.asarray(f[seg_id][()], dtype=np.float32) for seg_id in segment_ids])

    def missing(self, segment_ids: list[str]) -> list[str]:
        cached = self.keys()
        return [s for s in segment_ids if s not in cached]
