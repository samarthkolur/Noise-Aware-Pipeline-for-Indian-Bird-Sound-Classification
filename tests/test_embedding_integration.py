"""Phase 4 integration test (design.md §10): real BirdNET V2.4 inference over
real iBC53-sourced segments, verifying embedding shape and cache idempotency.

Marked slow: downloads/loads the real TensorFlow-backed BirdNET model on
first run and performs genuine inference, unlike the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.audio import load_audio
from pipeline.cache import EmbeddingCache
from pipeline.config import PROJECT_ROOT, load_config
from pipeline.embedding import extract_embeddings_batch

pytestmark = pytest.mark.slow

SEGMENTED_DIR = PROJECT_ROOT / "data" / "segmented"


def _sample_real_segments(n: int) -> list[Path]:
    if not SEGMENTED_DIR.exists():
        pytest.skip("data/segmented not present in this environment")
    species_dirs = sorted(p for p in SEGMENTED_DIR.iterdir() if p.is_dir())
    files: list[Path] = []
    for species_dir in species_dirs:
        files.extend(sorted(species_dir.glob("*.wav")))
        if len(files) >= n:
            break
    if len(files) < n:
        pytest.skip("not enough real segments available in data/segmented")
    return files[:n]


def test_real_embedding_shape_and_idempotent_cache(tmp_path):
    cfg = load_config()
    files = _sample_real_segments(3)

    audios = [load_audio(str(f), cfg.audio.target_sr)[0] for f in files]
    embeddings = extract_embeddings_batch(audios, cfg.audio.target_sr, cfg.embedding)

    assert embeddings.shape == (3, cfg.embedding.embedding_dim)

    cache = EmbeddingCache(tmp_path / "embeddings.h5")
    segment_ids = [f.stem for f in files]

    first_written = cache.write_batch(segment_ids, embeddings)
    assert first_written == 3

    # Re-running extraction against a populated cache must be a no-op (DD-003).
    second_written = cache.write_batch(segment_ids, embeddings)
    assert second_written == 0

    cached = cache.read_batch(segment_ids)
    assert cached.shape == (3, cfg.embedding.embedding_dim)
