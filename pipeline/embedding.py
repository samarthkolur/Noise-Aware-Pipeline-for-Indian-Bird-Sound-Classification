"""Frozen BirdNET V2.4 embedding extraction (design.md §6.1, §20, DD-009).

Wraps the `birdnet` PyPI package. DD-009 declares a `tflite` vs `tensorflow`
backend flag; the installed `birdnet` package (0.2.x) only ships `tf` and
`pb` backends (no `tflite-runtime` wheel is available for this platform), so
`tflite` resolves to the `tensorflow` fallback path with a logged warning —
exactly the contingency DD-009 documents.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from pipeline.config import EmbeddingConfig

logger = logging.getLogger(__name__)

_BACKEND_MAP = {"tensorflow": "tf", "tf": "tf", "tflite": "tf", "pb": "pb"}


@lru_cache(maxsize=1)
def _load_model(model_version: str, backend: str):
    resolved_backend = _BACKEND_MAP.get(backend)
    if resolved_backend is None:
        raise ValueError(f"Unknown embedding backend: {backend!r}")
    if backend == "tflite" and resolved_backend != "tflite":
        logger.warning(
            "embedding.backend='tflite' requested but no tflite-runtime wheel is "
            "available in this environment; falling back to backend='%s' (DD-009).",
            resolved_backend,
        )
    import birdnet

    # resolved_backend is a runtime str, not a Literal, so it can't satisfy
    # birdnet.load's overloaded Literal-typed signature statically.
    return birdnet.load("acoustic", model_version, resolved_backend)  # type: ignore[call-overload]


def extract_embedding(audio: np.ndarray, sr: int, config: EmbeddingConfig) -> np.ndarray:
    """Extract a single (embedding_dim,) BirdNET embedding from one audio segment."""
    model = _load_model(config.model_version, config.backend)
    result = model.encode_arrays([(audio.astype(np.float32), sr)])
    embedding = np.asarray(result.embeddings)[0].reshape(-1)
    if embedding.shape[0] != config.embedding_dim:
        raise ValueError(f"Expected embedding dim {config.embedding_dim}, got {embedding.shape[0]}")
    return embedding.astype(np.float32)


def extract_embeddings_batch(
    audios: list[np.ndarray], sr: int, config: EmbeddingConfig
) -> np.ndarray:
    """Extract embeddings for a batch of same-length audio segments.

    Returns an array of shape (len(audios), embedding_dim).
    """
    if not audios:
        return np.zeros((0, config.embedding_dim), dtype=np.float32)
    model = _load_model(config.model_version, config.backend)
    result = model.encode_arrays([(a.astype(np.float32), sr) for a in audios])
    embeddings = np.asarray(result.embeddings).reshape(len(audios), -1)
    if embeddings.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Expected embedding dim {config.embedding_dim}, got {embeddings.shape[1]}"
        )
    return embeddings.astype(np.float32)


def extract_max_confidence_batch(
    audios: list[np.ndarray], sr: int, config: EmbeddingConfig
) -> np.ndarray:
    """Max BirdNET species-confidence per segment (raw sigmoid output, no
    thresholding) — used for the BirdNET-baseline benchmark (design.md §6.0,
    §9). A universal 0.5 decision threshold is applied downstream, not here.
    """
    if not audios:
        return np.zeros(0, dtype=np.float32)
    model = _load_model(config.model_version, config.backend)
    result = model.predict_arrays(
        [(a.astype(np.float32), sr) for a in audios],
        top_k=1,
        default_confidence_threshold=0.0,
    )
    df = result.to_dataframe()
    max_conf = np.zeros(len(audios), dtype=np.float32)
    for input_idx, group in df.groupby("input"):
        max_conf[int(input_idx)] = float(group["confidence"].max())
    return max_conf


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """L2-normalise an embedding at inference time (design.md §6.3)."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm
