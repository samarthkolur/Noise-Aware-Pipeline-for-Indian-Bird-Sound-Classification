"""
Stage 8: Hard-Negative Mining

Iteratively identify noise segments that the current model misclassifies
as bird, add them to the training set, and retrain to tighten decision
boundaries.
"""

import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def find_hard_negatives(
    classifier,
    noise_embeddings: np.ndarray,
    top_k: int = None,
) -> tuple:
    """
    Identify the most confidently misclassified noise segments.

    These are noise samples that the classifier predicts as bird
    with highest probability — the "hardest" negatives.

    Args:
        classifier: Trained classifier with .predict(X) method.
        noise_embeddings: Embedding vectors of known-noise segments.
        top_k: Number of hardest negatives to select.

    Returns:
        Tuple of (hard_negative_indices, hard_negative_scores).
    """
    top_k = top_k or config.HARD_NEGATIVE_TOP_K

    # Get classifier predictions on noise data
    probabilities = classifier.predict(noise_embeddings)

    # Sort by descending probability (most confident false positives first)
    sorted_indices = np.argsort(probabilities)[::-1]

    # Select top-K
    hard_indices = sorted_indices[:min(top_k, len(sorted_indices))]
    hard_scores = probabilities[hard_indices]

    n_misclassified = np.sum(probabilities > 0.5)
    print(
        f"  [Stage 8] {n_misclassified}/{len(noise_embeddings)} noise segments "
        f"misclassified as bird. Selected top-{len(hard_indices)} hardest."
    )

    return hard_indices, hard_scores


def augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hard_negative_embeddings: np.ndarray,
) -> tuple:
    """
    Add hard negatives to the training set.

    Args:
        X_train: Current training embeddings.
        y_train: Current training labels.
        hard_negative_embeddings: Embedding vectors of hard negatives.

    Returns:
        Tuple of (augmented_X, augmented_y).
    """
    hard_labels = np.zeros(len(hard_negative_embeddings), dtype=y_train.dtype)

    X_augmented = np.concatenate([X_train, hard_negative_embeddings], axis=0)
    y_augmented = np.concatenate([y_train, hard_labels], axis=0)

    # Shuffle
    perm = np.random.RandomState(config.RANDOM_SEED).permutation(len(X_augmented))
    X_augmented = X_augmented[perm]
    y_augmented = y_augmented[perm]

    print(
        f"  [Stage 8] Augmented training set: {len(X_augmented)} samples "
        f"({int(y_augmented.sum())} bird, {int((1-y_augmented).sum())} noise)"
    )

    return X_augmented, y_augmented


def iterative_retrain(
    classifier_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noise_embeddings: np.ndarray,
    n_rounds: int = None,
) -> tuple:
    """
    Iteratively mine hard negatives and retrain the classifier.

    Each round:
        1. Predict on noise data with current model
        2. Identify top-K hardest misclassified noise samples
        3. Add them to training set
        4. Retrain classifier from scratch

    Args:
        classifier_factory: Callable that creates a fresh classifier.
        X_train: Initial training embeddings.
        y_train: Initial training labels.
        X_val: Validation embeddings.
        y_val: Validation labels.
        noise_embeddings: Pool of noise embeddings to mine from.
        n_rounds: Number of mining rounds.

    Returns:
        Tuple of (final_classifier, final_X_train, final_y_train, history).
    """
    n_rounds = n_rounds or config.HARD_NEGATIVE_ROUNDS

    current_X = X_train.copy()
    current_y = y_train.copy()
    history = []

    # Initial training
    print(f"\n[Stage 8] Starting hard-negative mining ({n_rounds} rounds)")
    classifier = classifier_factory()
    train_hist = classifier.train_model(current_X, current_y, X_val, y_val)
    history.append({"round": 0, "type": "initial", **train_hist})

    for round_num in range(1, n_rounds + 1):
        print(f"\n[Stage 8] ── Round {round_num}/{n_rounds} ──")

        # Find hard negatives
        hard_indices, hard_scores = find_hard_negatives(classifier, noise_embeddings)

        if len(hard_indices) == 0:
            print("  [Stage 8] No hard negatives found. Stopping early.")
            break

        # Augment training set
        hard_embeddings = noise_embeddings[hard_indices]
        current_X, current_y = augment_training_set(current_X, current_y, hard_embeddings)

        # Retrain
        classifier = classifier_factory()
        train_hist = classifier.train_model(current_X, current_y, X_val, y_val)
        history.append({
            "round": round_num,
            "n_hard_negatives": len(hard_indices),
            **train_hist,
        })

    print(f"\n[Stage 8] Hard-negative mining complete. Final training set: {len(current_X)} samples.")

    return classifier, current_X, current_y, history


if __name__ == "__main__":
    print("Stage 8: Use run_pipeline.py to run hard-negative mining.")
