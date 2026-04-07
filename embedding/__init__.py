"""embedding — Embedding extraction pipeline."""

from .embedding import (
    BaseEncoder,
    BirdNETEncoder,
    BirdNETModelNotFoundError,
    PlaceholderEncoder,
    EmbeddingPipeline,
    EmbeddingRecord,
    EmbeddingStore,
    REQUIRED_BIRDNET_TFLITE_NAME,
    REQUIRED_BIRDNET_VERSION,
    build_encoder,
    extract_embeddings,
    load_species_embeddings,
    load_all_embeddings,
)

__all__ = [
    "BaseEncoder",
    "BirdNETEncoder",
    "BirdNETModelNotFoundError",
    "PlaceholderEncoder",
    "EmbeddingPipeline",
    "EmbeddingRecord",
    "EmbeddingStore",
    "REQUIRED_BIRDNET_TFLITE_NAME",
    "REQUIRED_BIRDNET_VERSION",
    "build_encoder",
    "extract_embeddings",
    "load_species_embeddings",
    "load_all_embeddings",
]
