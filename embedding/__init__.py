"""embedding — Embedding extraction pipeline."""

from .embedding import (
    BaseEncoder,
    BirdNETEncoder,
    PlaceholderEncoder,
    EmbeddingPipeline,
    EmbeddingRecord,
    EmbeddingStore,
    build_encoder,
    extract_embeddings,
    load_species_embeddings,
    load_all_embeddings,
)

__all__ = [
    "BaseEncoder",
    "BirdNETEncoder",
    "PlaceholderEncoder",
    "EmbeddingPipeline",
    "EmbeddingRecord",
    "EmbeddingStore",
    "build_encoder",
    "extract_embeddings",
    "load_species_embeddings",
    "load_all_embeddings",
]
