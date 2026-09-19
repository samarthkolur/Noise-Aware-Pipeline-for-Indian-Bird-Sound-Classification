"""
Stage 6: Active Learning / Expert-in-the-Loop Feedback

Implements "Agile Modeling" — an iterative feedback loop where the most
uncertain predictions are exported for expert review, and the expert
labels are fed back to retrain the classifier.

Research shows this provides a "speed-up factor," reaching high accuracy
with ~25% of the typical training budget. A new classifier version can
be trained and validated in under a minute (with RF).

Workflow:
    1. Identify segments where classifier is least certain (near boundary)
    2. Export uncertain samples as CSV with audio paths for expert review
    3. Import expert corrections
    4. Retrain classifier with augmented labels
    5. Repeat until convergence or max rounds

Output:
    data/active_learning/review_round_N.csv
    data/active_learning/progress.json
"""

import os
import json
import csv
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class UncertaintySampler:
    """
    Identify the most uncertain predictions for expert review.

    Uses margin-based uncertainty: samples where the classifier's
    predicted probability is closest to 0.5 (maximum uncertainty).
    """

    def __init__(
        self,
        uncertainty_threshold: float = None,
        top_k: int = None,
    ):
        self.uncertainty_threshold = uncertainty_threshold or config.AL_UNCERTAINTY_THRESHOLD
        self.top_k = top_k or config.AL_EXPORT_TOP_K

    def find_uncertain_samples(
        self,
        classifier,
        X: np.ndarray,
    ) -> tuple:
        """
        Find samples closest to the decision boundary.

        Uncertainty = |predicted_probability - 0.5|
        Lower uncertainty → closer to boundary → harder to classify.

        Args:
            classifier: Trained classifier with .predict(X) method.
            X: Embedding matrix (N, D).

        Returns:
            Tuple of (uncertain_indices, uncertainty_scores, predicted_probs).
        """
        probs = classifier.predict(X)
        uncertainty = np.abs(probs - 0.5)

        # Find samples within uncertainty threshold
        uncertain_mask = uncertainty < self.uncertainty_threshold
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) > self.top_k:
            # Take the top_k most uncertain
            sorted_by_uncertainty = np.argsort(uncertainty[uncertain_indices])
            uncertain_indices = uncertain_indices[sorted_by_uncertainty[:self.top_k]]

        uncertainty_scores = uncertainty[uncertain_indices]
        predicted_probs = probs[uncertain_indices]

        print(f"  [Stage 6] Found {len(uncertain_indices)} uncertain samples "
              f"(threshold: ±{self.uncertainty_threshold} from 0.5)")

        return uncertain_indices, uncertainty_scores, predicted_probs

    def export_for_review(
        self,
        uncertain_indices: np.ndarray,
        uncertainty_scores: np.ndarray,
        predicted_probs: np.ndarray,
        file_paths: np.ndarray,
        round_num: int = 1,
    ) -> str:
        """
        Export uncertain samples as CSV for expert review.

        CSV columns: file_path, uncertainty_score, predicted_label,
                     predicted_prob, expert_label (empty, to be filled)

        Args:
            uncertain_indices: Indices of uncertain samples.
            uncertainty_scores: Uncertainty scores.
            predicted_probs: Classifier probabilities.
            file_paths: All file paths (indexed by uncertain_indices).
            round_num: Active learning round number.

        Returns:
            Path to the exported CSV file.
        """
        os.makedirs(config.ACTIVE_LEARNING_DIR, exist_ok=True)
        csv_path = os.path.join(
            config.ACTIVE_LEARNING_DIR, f"review_round_{round_num}.csv"
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file_path", "uncertainty_score", "predicted_prob",
                "predicted_label", "expert_label"
            ])

            for idx, unc, prob in zip(
                uncertain_indices, uncertainty_scores, predicted_probs
            ):
                predicted_label = "bird" if prob >= 0.5 else "noise"
                writer.writerow([
                    str(file_paths[idx]),
                    f"{unc:.4f}",
                    f"{prob:.4f}",
                    predicted_label,
                    "",  # Expert fills this in
                ])

        print(f"  [Stage 6] Exported {len(uncertain_indices)} samples to {csv_path}")
        print(f"  [Stage 6] → Have an expert review and fill the 'expert_label' column")
        return csv_path

    @staticmethod
    def import_expert_labels(csv_path: str) -> dict:
        """
        Import expert-reviewed labels from CSV.

        Args:
            csv_path: Path to the reviewed CSV.

        Returns:
            Dict mapping file_path → expert_label (0 or 1).
        """
        expert_labels = {}
        n_labeled = 0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_str = row.get("expert_label", "").strip().lower()
                if label_str in ("bird", "1"):
                    expert_labels[row["file_path"]] = config.BIRD_LABEL
                    n_labeled += 1
                elif label_str in ("noise", "0"):
                    expert_labels[row["file_path"]] = config.NOISE_LABEL
                    n_labeled += 1
                # Skip empty or invalid labels

        print(f"  [Stage 6] Imported {n_labeled} expert labels from {csv_path}")
        return expert_labels


class ActiveLearningLoop:
    """
    Manage the iterative active learning process.

    Each round:
        1. Train classifier on current labeled data
        2. Find uncertain samples in unlabeled pool
        3. Export for expert review
        4. Import expert labels → augment training set
        5. Track accuracy improvement
    """

    def __init__(self, max_rounds: int = None):
        self.max_rounds = max_rounds or config.AL_MAX_ROUNDS
        self.progress = []
        self.sampler = UncertaintySampler()

    def run_round(
        self,
        classifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_pool: np.ndarray,
        pool_paths: np.ndarray,
        round_num: int,
    ) -> dict:
        """
        Execute one round of active learning.

        Args:
            classifier: Fresh classifier instance.
            X_train: Current training embeddings.
            y_train: Current training labels.
            X_val: Validation embeddings.
            y_val: Validation labels.
            X_pool: Unlabeled pool embeddings.
            pool_paths: File paths for pool samples.
            round_num: Round number (1-indexed).

        Returns:
            Dict with round metrics and export CSV path.
        """
        print(f"\n  ── Active Learning Round {round_num} ──")

        # Train classifier
        metrics = classifier.train_model(X_train, y_train, X_val, y_val)

        # Find uncertain samples in the pool
        uncertain_idx, unc_scores, pred_probs = self.sampler.find_uncertain_samples(
            classifier, X_pool
        )

        if len(uncertain_idx) == 0:
            print("  [Stage 6] No uncertain samples found — model is confident!")
            return {"round": round_num, "n_uncertain": 0, "csv_path": None, **metrics}

        # Export for review
        csv_path = self.sampler.export_for_review(
            uncertain_idx, unc_scores, pred_probs, pool_paths, round_num
        )

        round_result = {
            "round": round_num,
            "n_uncertain": len(uncertain_idx),
            "csv_path": csv_path,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.progress.append(round_result)

        return round_result

    def retrain_with_feedback(
        self,
        expert_labels: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_pool: np.ndarray,
        pool_paths: np.ndarray,
    ) -> tuple:
        """
        Augment training set with expert-labeled samples and prepare for retrain.

        Args:
            expert_labels: Dict mapping file_path → label.
            X_train: Current training embeddings.
            y_train: Current training labels.
            X_pool: Pool embeddings.
            pool_paths: Pool file paths.

        Returns:
            Tuple of (augmented_X_train, augmented_y_train, remaining_X_pool, remaining_pool_paths).
        """
        new_X = []
        new_y = []
        keep_mask = np.ones(len(X_pool), dtype=bool)

        for i, path in enumerate(pool_paths):
            if str(path) in expert_labels:
                new_X.append(X_pool[i])
                new_y.append(expert_labels[str(path)])
                keep_mask[i] = False

        if new_X:
            X_augmented = np.concatenate([X_train, np.array(new_X)], axis=0)
            y_augmented = np.concatenate([y_train, np.array(new_y)], axis=0)
        else:
            X_augmented = X_train
            y_augmented = y_train

        remaining_X_pool = X_pool[keep_mask]
        remaining_paths = pool_paths[keep_mask]

        n_new = len(new_X)
        n_bird = int((np.array(new_y) == config.BIRD_LABEL).sum()) if new_y else 0
        n_noise = n_new - n_bird

        print(f"  [Stage 6] Added {n_new} expert labels ({n_bird} bird, {n_noise} noise)")
        print(f"  [Stage 6] New training size: {len(X_augmented)}, "
              f"remaining pool: {len(remaining_X_pool)}")

        return X_augmented, y_augmented, remaining_X_pool, remaining_paths

    def save_progress(self):
        """Save active learning progress to JSON."""
        progress_path = os.path.join(config.ACTIVE_LEARNING_DIR, "progress.json")
        with open(progress_path, "w") as f:
            json.dump(self.progress, f, indent=2)
        print(f"  [Stage 6] Progress saved to {progress_path}")

    @staticmethod
    def load_progress() -> list:
        """Load active learning progress from JSON."""
        progress_path = os.path.join(config.ACTIVE_LEARNING_DIR, "progress.json")
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                return json.load(f)
        return []
