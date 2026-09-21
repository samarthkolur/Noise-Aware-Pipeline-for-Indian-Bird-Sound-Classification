"""Pydantic-validated loader for config.yaml (DD-004).

Every path, hyperparameter, and threshold used by the pipeline is declared in
config.yaml and loaded through this module. No magic numbers belong in the
pipeline modules themselves.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    data_dir: str
    raw_data_dir: str
    segmented_dir: str
    segments_dir: str
    manifest_path: str
    split_path: str
    embeddings_cache_path: str
    artifacts_dir: str
    outputs_dir: str
    birdnet_model_path: str


class AudioConfig(BaseModel):
    target_sr: int
    segment_length_s: float
    silence_db_threshold: float


class NoiseSegregationWeights(BaseModel):
    zcr: float
    spectral_flatness: float
    centroid_flag: float
    centroid_std: float
    insect_periodicity: float


class NoiseSegregationConfig(BaseModel):
    n_subframes: int
    subframe_duration_s: float
    weights: NoiseSegregationWeights
    insect_band_hz: tuple[int, int]
    zcr_reference: float
    centroid_threshold_hz: float
    centroid_std_reference_hz: float


class BirdGuardConfig(BaseModel):
    harmonic_ratio_threshold: float
    peak_median_threshold: float


class BirdRescueConfig(BaseModel):
    hidden_dims: list[int]
    rescue_threshold: float


class EmbeddingConfig(BaseModel):
    backend: str
    model_version: str
    embedding_dim: int


class SplitConfig(BaseModel):
    train_ratio: float
    val_ratio: float
    test_ratio: float


class MLPConfig(BaseModel):
    hidden_dims: list[int]
    dropout: float
    focal_gamma: float
    focal_alpha: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stop_patience: int


class AutoencoderConfig(BaseModel):
    bottleneck_dim: int
    hidden_dim: int
    ae_percentile: float


class RouterConfig(BaseModel):
    tau_low: float
    tau_high: float


class SyntheticConfig(BaseModel):
    allow_synthetic_noise: bool = False
    noise_types: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    random_seed: int
    log_level: str
    paths: PathsConfig
    audio: AudioConfig
    noise_segregation: NoiseSegregationConfig
    bird_guard: BirdGuardConfig
    bird_rescue: BirdRescueConfig
    embedding: EmbeddingConfig
    split: SplitConfig
    mlp: MLPConfig
    autoencoder: AutoencoderConfig
    router: RouterConfig
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a config-declared path against the project root."""
        p = Path(relative_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@cache
def load_config(config_path: str | None = None) -> PipelineConfig:
    """Load and validate config.yaml (or the path given by $CONFIG_PATH)."""
    path = Path(config_path or os.environ.get("CONFIG_PATH", "config.yaml"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)
