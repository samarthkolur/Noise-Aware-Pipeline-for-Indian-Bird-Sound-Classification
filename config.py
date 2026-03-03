"""
Central configuration for the Noise-Aware Bird Segregation Pipeline.
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

# ─── Audio Settings ─────────────────────────────────────────────────────────
TARGET_SR = 48000           # Sample rate (Hz) — must match BirdNET
SEGMENT_LENGTH = 3.0        # Seconds per segment
HOP_LENGTH = 1.0            # Stride between overlapping windows (seconds)
USE_OVERLAP = True          # Enable overlapping segmentation

# ─── Embedding Extraction ───────────────────────────────────────────────────
EMBEDDING_MODELS = ["birdnet", "yamnet", "openl3"]  # Which embeddings to use
OPENL3_EMBEDDING_SIZE = 512          # 512 or 6144
OPENL3_CONTENT_TYPE = "env"          # "env" (environmental) or "music"
FUSE_EMBEDDINGS = True               # Concatenate multi-source embeddings

# ─── Binary Classifier (Stage 3) ────────────────────────────────────────────
CLASSIFIER_TYPE = "mlp"              # "mlp", "logistic", "svm", "rf"
MLP_HIDDEN_DIMS = [256, 128, 64]     # MLP hidden layer sizes
MLP_DROPOUT = 0.3
MLP_LEARNING_RATE = 1e-3
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 64
MLP_PATIENCE = 10                    # Early stopping patience
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.75

# ─── OOD Detection (Stage 4) ────────────────────────────────────────────────
OOD_METHODS = ["mahalanobis", "ocsvm", "iforest", "autoencoder"]
MAHALANOBIS_THRESHOLD = 25.0         # Chi-squared threshold
OCSVM_KERNEL = "rbf"
OCSVM_NU = 0.05
IFOREST_CONTAMINATION = 0.05
IFOREST_N_ESTIMATORS = 200
AE_HIDDEN_DIM = 64
AE_LATENT_DIM = 32
AE_EPOCHS = 50
AE_LEARNING_RATE = 1e-3
AE_RECONSTRUCTION_THRESHOLD = None   # Auto-determined from training data

# ─── Source Separation (Stage 5) ────────────────────────────────────────────
ENABLE_SOURCE_SEPARATION = True
HARMONIC_RATIO_THRESHOLD = 0.6       # Minimum harmonic energy ratio

# ─── Temporal Consistency (Stage 6) ──────────────────────────────────────────
ENABLE_TEMPORAL_SMOOTHING = True
TEMPORAL_WINDOW_SIZE = 5             # Number of adjacent segments
TEMPORAL_METHOD = "confidence_avg"   # "sliding", "majority", "confidence_avg"

# ─── Ensemble Decision (Stage 7) ─────────────────────────────────────────────
ENSEMBLE_CLASSIFIER_THRESHOLD = 0.5
ENSEMBLE_OOD_REQUIRED = True         # Must pass OOD check
ENSEMBLE_HARMONIC_THRESHOLD = 0.5    # Minimum harmonic ratio
ENSEMBLE_BIRDNET_THRESHOLD = 0.3     # Minimum BirdNET confidence
ENSEMBLE_MIN_AGREEMENT = 3           # Min signals that must agree (out of 4)

# ─── Hard-Negative Mining (Stage 8) ──────────────────────────────────────────
HARD_NEGATIVE_ROUNDS = 3             # Iterative retraining rounds
HARD_NEGATIVE_TOP_K = 100            # Number of hardest negatives per round

# ─── Noise Label Generation ─────────────────────────────────────────────────
BIRDNET_NOISE_THRESHOLD = 0.1        # Segments below this BirdNET confidence → noise
NOISE_LABEL = 0
BIRD_LABEL = 1

# ─── Train/Val Split ────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ─── Stage Enable Flags (for ablation) ──────────────────────────────────────
STAGES_ENABLED = {
    "segmentation": True,
    "embeddings": True,
    "classifier": True,
    "ood": True,
    "source_separation": True,
    "temporal": True,
    "ensemble": True,
    "hard_negatives": True,
}

# ─── Ensure directories exist ───────────────────────────────────────────────
for _dir in [EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR, NOISE_AWARE_OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)
