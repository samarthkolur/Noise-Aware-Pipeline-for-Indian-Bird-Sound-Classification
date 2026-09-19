"""
Stage 3: Hard-Negative Dataset Curation

Build a curated binary dataset (bird vs noise) using BirdNET confidence
scores, spectral heuristics, and hard-negative mining. This is the core
of the data-centric approach — quality training data is more important
than model complexity.

Strategy:
    1. Use BirdNET confidence to pseudo-label: high conf → bird, low conf → noise
    2. Apply spectral flatness filter to catch BirdNET false positives
    3. Identify "hard negatives" — noise that BirdNET incorrectly calls bird
    4. Generate a balanced, versioned train/val split
    5. Save a JSON manifest with per-sample metadata

Output:
    data/hard_negative_dataset/manifest.json
    features/embeddings/binary_labels.npy
    features/embeddings/X_train.npy, X_val.npy
    features/embeddings/y_train.npy, y_val.npy
"""

import os
import json
import numpy as np
import librosa
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_spectral_flatness(file_path: str) -> float:
    """
    Compute spectral flatness for an audio segment.

    Spectral flatness measures how noise-like a signal is.
    Values close to 1.0 = pure noise (flat spectrum).
    Values close to 0.0 = tonal (peaked spectrum, likely bird call).

    Args:
        file_path: Path to a .wav segment.

    Returns:
        Mean spectral flatness (0.0–1.0). Returns 0.5 on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=config.TARGET_SR)
        flatness = librosa.feature.spectral_flatness(y=y)
        return float(np.mean(flatness))
    except Exception:
        return 0.5


def compute_harmonic_ratio(file_path: str) -> float:
    """
    Compute harmonic energy ratio using HPSS.

    Bird calls are predominantly harmonic (ratio > 0.5).
    Noise is predominantly percussive or broadband (ratio < 0.3).

    Args:
        file_path: Path to a .wav segment.

    Returns:
        Harmonic ratio in [0, 1]. Returns 0.5 on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=config.TARGET_SR)
        harmonic, percussive = librosa.effects.hpss(y)
        h_energy = np.sum(harmonic ** 2)
        p_energy = np.sum(percussive ** 2)
        eps = 1e-10
        return float(h_energy / (h_energy + p_energy + eps))
    except Exception:
        return 0.5


def pseudo_label_segments(
    file_paths: np.ndarray,
    birdnet_confidences: np.ndarray,
) -> dict:
    """
    Assign pseudo-labels using BirdNET confidence and spectral heuristics.

    Labeling strategy:
        - BirdNET conf >= BIRD_CONFIDENCE_HIGH → bird (confident)
        - BirdNET conf < BIRD_CONFIDENCE_LOW → noise (confident)
        - In between → uncertain (flagged for active learning)

    Additional filter: if spectral flatness is very high, override bird
    label to noise (catches BirdNET false positives on broadband noise).

    Args:
        file_paths: Array of file paths.
        birdnet_confidences: Array of BirdNET max confidence scores.

    Returns:
        Dict with keys: 'labels', 'certainty', 'hard_negatives_mask'
    """
    n = len(file_paths)
    labels = np.full(n, -1, dtype=np.int64)  # -1 = uncertain
    certainty = np.zeros(n, dtype=np.float32)
    hard_negatives_mask = np.zeros(n, dtype=bool)

    print(f"[Stage 3] Pseudo-labeling {n} segments...")
    print(f"  Bird threshold: >= {config.BIRD_CONFIDENCE_HIGH}")
    print(f"  Noise threshold: < {config.BIRD_CONFIDENCE_LOW}")

    n_spectral_overrides = 0

    for i in range(n):
        conf = birdnet_confidences[i]

        if conf >= config.BIRD_CONFIDENCE_HIGH:
            # High confidence bird — but verify with spectral check
            flatness = compute_spectral_flatness(str(file_paths[i]))

            if flatness > config.SPECTRAL_FLATNESS_NOISE_THRESHOLD:
                # BirdNET says bird, but spectrum says noise → hard negative!
                labels[i] = config.NOISE_LABEL
                hard_negatives_mask[i] = True
                n_spectral_overrides += 1
            else:
                labels[i] = config.BIRD_LABEL

            certainty[i] = conf

        elif conf < config.BIRD_CONFIDENCE_LOW:
            labels[i] = config.NOISE_LABEL
            certainty[i] = 1.0 - conf

        else:
            # Uncertain — leave as -1 for active learning
            labels[i] = -1
            certainty[i] = 0.0

        if (i + 1) % 500 == 0:
            print(f"    Labeled {i+1}/{n}...")

    n_bird = int((labels == config.BIRD_LABEL).sum())
    n_noise = int((labels == config.NOISE_LABEL).sum())
    n_uncertain = int((labels == -1).sum())
    n_hard_neg = int(hard_negatives_mask.sum())

    print(f"[Stage 3] Pseudo-labels: {n_bird} bird, {n_noise} noise, "
          f"{n_uncertain} uncertain")
    print(f"[Stage 3] Hard negatives found: {n_hard_neg} "
          f"(spectral overrides: {n_spectral_overrides})")

    return {
        "labels": labels,
        "certainty": certainty,
        "hard_negatives_mask": hard_negatives_mask,
    }


def create_versioned_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    file_paths: np.ndarray,
    birdnet_confidences: np.ndarray,
) -> dict:
    """
    Create a train/val split from labeled data (excluding uncertain samples).

    Args:
        embeddings: (N, D) embedding matrix.
        labels: (N,) label array (-1 = uncertain, 0 = noise, 1 = bird).
        file_paths: (N,) file paths.
        birdnet_confidences: (N,) confidence scores.

    Returns:
        Dict with train/val arrays and metadata.
    """
    # Only use confidently labeled samples for training
    labeled_mask = labels >= 0
    X_labeled = embeddings[labeled_mask]
    y_labeled = labels[labeled_mask]
    paths_labeled = file_paths[labeled_mask]
    confs_labeled = birdnet_confidences[labeled_mask]

    print(f"[Stage 3] Creating train/val split from {len(X_labeled)} labeled samples")

    if len(np.unique(y_labeled)) < 2:
        print("  [WARN] Only one class present — cannot stratify. Using random split.")
        stratify = None
    else:
        stratify = y_labeled

    idx = np.arange(len(X_labeled))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=1 - config.TRAIN_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=stratify,
    )

    X_train, X_val = X_labeled[idx_train], X_labeled[idx_val]
    y_train, y_val = y_labeled[idx_train], y_labeled[idx_val]

    # Save splits
    np.save(os.path.join(config.EMBEDDINGS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.EMBEDDINGS_DIR, "idx_train.npy"), idx_train)
    np.save(os.path.join(config.EMBEDDINGS_DIR, "idx_val.npy"), idx_val)

    n_bird_train = int((y_train == config.BIRD_LABEL).sum())
    n_noise_train = int((y_train == config.NOISE_LABEL).sum())
    n_bird_val = int((y_val == config.BIRD_LABEL).sum())
    n_noise_val = int((y_val == config.NOISE_LABEL).sum())

    print(f"  Train: {len(X_train)} ({n_bird_train} bird, {n_noise_train} noise)")
    print(f"  Val:   {len(X_val)} ({n_bird_val} bird, {n_noise_val} noise)")

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "idx_train": idx_train, "idx_val": idx_val,
        "paths_train": paths_labeled[idx_train],
        "paths_val": paths_labeled[idx_val],
    }


def generate_dataset_manifest(
    file_paths: np.ndarray,
    labels: np.ndarray,
    birdnet_confidences: np.ndarray,
    hard_negatives_mask: np.ndarray,
    split_info: dict,
) -> dict:
    """
    Save a JSON manifest documenting the curated dataset.

    Args:
        file_paths: All file paths.
        labels: All labels (-1/0/1).
        birdnet_confidences: All confidence scores.
        hard_negatives_mask: Boolean mask for hard negatives.
        split_info: Train/val split metadata.

    Returns:
        Manifest dictionary.
    """
    manifest = {
        "version": "3.0.0",
        "created": datetime.now().isoformat(),
        "total_segments": len(file_paths),
        "labeled_segments": int((labels >= 0).sum()),
        "uncertain_segments": int((labels == -1).sum()),
        "bird_segments": int((labels == config.BIRD_LABEL).sum()),
        "noise_segments": int((labels == config.NOISE_LABEL).sum()),
        "hard_negatives": int(hard_negatives_mask.sum()),
        "train_size": len(split_info["X_train"]),
        "val_size": len(split_info["X_val"]),
        "config": {
            "bird_confidence_high": config.BIRD_CONFIDENCE_HIGH,
            "bird_confidence_low": config.BIRD_CONFIDENCE_LOW,
            "spectral_flatness_threshold": config.SPECTRAL_FLATNESS_NOISE_THRESHOLD,
            "train_ratio": config.TRAIN_RATIO,
            "random_seed": config.RANDOM_SEED,
        },
    }

    os.makedirs(config.HARD_NEGATIVE_DIR, exist_ok=True)
    manifest_path = config.DATASET_MANIFEST_PATH

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Stage 3] Manifest saved to {manifest_path}")
    return manifest


def curate_hard_negative_dataset() -> dict:
    """
    Main entry point: build the curated hard-negative dataset.

    Loads embeddings and BirdNET confidences from Stage 2, applies
    pseudo-labeling with spectral filters, creates train/val split,
    and saves a versioned manifest.

    Returns:
        Dict with split info and manifest.
    """
    print("\n" + "=" * 70)
    print("  STAGE 3: Hard-Negative Dataset Curation")
    print("=" * 70)

    # Load Stage 2 outputs
    embeddings = np.load(os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy"))
    file_paths = np.load(os.path.join(config.EMBEDDINGS_DIR, "paths.npy"), allow_pickle=True)
    birdnet_confs = np.load(os.path.join(config.EMBEDDINGS_DIR, "birdnet_confidences.npy"))

    print(f"  Loaded {len(embeddings)} embeddings ({embeddings.shape})")

    # Step 1: Pseudo-label with spectral filtering
    label_result = pseudo_label_segments(file_paths, birdnet_confs)
    labels = label_result["labels"]
    hard_negatives_mask = label_result["hard_negatives_mask"]

    # Save binary labels (including -1 for uncertain)
    np.save(os.path.join(config.EMBEDDINGS_DIR, "binary_labels.npy"), labels)

    # Step 2: Create train/val split (excludes uncertain samples)
    split_info = create_versioned_split(
        embeddings, labels, file_paths, birdnet_confs
    )

    # Step 3: Save manifest
    manifest = generate_dataset_manifest(
        file_paths, labels, birdnet_confs, hard_negatives_mask, split_info
    )

    return {
        "split_info": split_info,
        "manifest": manifest,
        "labels": labels,
        "hard_negatives_mask": hard_negatives_mask,
    }


if __name__ == "__main__":
    curate_hard_negative_dataset()
