"""
Central configuration for the Noise-Aware Bird Segregation Pipeline V3.

Data-centric architecture with Hard-Negative Mining, Active Learning,
and Multi-Stage Filtering for false-positive suppression.

All paths, hyperparameters, and stage toggles live here.
"""

import os

# ─── Project Root ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Data Paths ─────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_ROOT, "iBC53")
SEGMENTED_DIR = os.path.join(DATA_ROOT, "segmented")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "features", "embeddings")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")
NOISE_AWARE_OUTPUT_DIR = os.path.join(DATA_ROOT, "noise_aware_dataset")

# ─── Hard-Negative Dataset (Stage 3) ────────────────────────────────────────
HARD_NEGATIVE_DIR = os.path.join(DATA_ROOT, "hard_negative_dataset")
DATASET_MANIFEST_PATH = os.path.join(HARD_NEGATIVE_DIR, "manifest.json")
NOISE_CATEGORIES = [
    "traffic", "horn", "construction", "wind", "rain",
    "insects", "urban_ambient", "silence", "unknown_noise",
]
# BirdNET confidence thresholds for pseudo-labeling
BIRD_CONFIDENCE_HIGH = 0.5       # Confident bird: conf >= this
BIRD_CONFIDENCE_LOW = 0.1        # Confident noise: conf < this
# Segments in between are "uncertain" — prime candidates for active learning
SPECTRAL_FLATNESS_NOISE_THRESHOLD = 0.4  # High flatness → broadband noise

# ─── Audio Settings ─────────────────────────────────────────────────────────
TARGET_SR = 48000           # Sample rate (Hz) — must match BirdNET
SEGMENT_LENGTH = 3.0        # Seconds per segment
HOP_LENGTH = 1.0            # Stride between overlapping windows (seconds)
USE_OVERLAP = True          # Enable overlapping segmentation

# ─── Embedding Extraction (Stage 2) ─────────────────────────────────────────
EMBEDDING_MODELS = ["birdnet"]   # BirdNET-only (1024-d embeddings)
EMBEDDING_DIM = 1024             # Expected output dimension

# ─── Binary Classifier (Stage 4) ────────────────────────────────────────────
CLASSIFIER_TYPE = "rf"           # "rf" (primary), "mlp" (optional secondary)

# Random Forest parameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None              # Let trees grow fully
RF_MIN_SAMPLES_LEAF = 2
RF_CLASS_WEIGHT = "balanced"     # Handle class imbalance automatically

# MLP parameters (optional secondary)
MLP_HIDDEN_DIMS = [256, 128, 64]
MLP_DROPOUT = 0.3
MLP_LEARNING_RATE = 1e-3
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 64
MLP_PATIENCE = 10
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.75

# ─── OOD Detection (Stage 5) ────────────────────────────────────────────────
OOD_METHODS = ["mahalanobis", "iforest"]
MAHALANOBIS_THRESHOLD = 25.0         # Chi-squared threshold
IFOREST_CONTAMINATION = 0.05
IFOREST_N_ESTIMATORS = 200

# ─── Active Learning (Stage 6) ──────────────────────────────────────────────
ACTIVE_LEARNING_DIR = os.path.join(DATA_ROOT, "active_learning")
AL_UNCERTAINTY_THRESHOLD = 0.15      # Uncertainty margin: |prob - 0.5| < this
AL_EXPORT_TOP_K = 200                # Max samples to export per round
AL_MAX_ROUNDS = 5                    # Maximum active learning iterations

# ─── Post-Processing (Stage 7) ──────────────────────────────────────────────
# Spectral / HPSS check
HARMONIC_RATIO_THRESHOLD = 0.3       # Minimum harmonic energy ratio

# Temporal smoothing
ENABLE_TEMPORAL_SMOOTHING = True
TEMPORAL_WINDOW_SIZE = 5             # Number of adjacent segments
TEMPORAL_METHOD = "confidence_avg"   # "sliding", "majority", "confidence_avg"

# Ecological priors (optional)
ENABLE_ECOLOGICAL_PRIORS = False
SPECIES_LOCATION_PRIORS = None       # Path to JSON with species-location matrix

# ─── Ensemble Decision (Stage 8) ────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "classifier": 0.35,
    "ood": 0.25,
    "postprocessing": 0.25,
    "birdnet_raw": 0.15,
}
ENSEMBLE_THRESHOLD = 0.5             # Weighted score >= this → bird
ENSEMBLE_BIRDNET_THRESHOLD = 0.05    # Minimum BirdNET confidence

# ─── Noise Labels ───────────────────────────────────────────────────────────
NOISE_LABEL = 0
BIRD_LABEL = 1

# ─── Train/Val Split ────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ─── Stage Enable Flags (for ablation) ──────────────────────────────────────
STAGES_ENABLED = {
    "segmentation": True,
    "embeddings": True,
    "hard_negative_curation": True,
    "classifier": True,
    "ood": True,
    "active_learning": False,        # Off by default (requires human input)
    "postprocessing": True,
    "ensemble": True,
}

# ─── Ensure directories exist ───────────────────────────────────────────────
for _dir in [
    EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR,
    NOISE_AWARE_OUTPUT_DIR, HARD_NEGATIVE_DIR, ACTIVE_LEARNING_DIR,
]:
    os.makedirs(_dir, exist_ok=True)
