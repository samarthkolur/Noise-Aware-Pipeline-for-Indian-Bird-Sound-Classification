#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation of the trained classifier.

Loads the test split from HDF5 embeddings, runs classifier inference,
and prints detailed metrics including confusion matrix and threshold curve.

Usage:
    python evaluate.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset import EmbeddingDataset, create_splits
from models.classifier import EmbeddingClassifier
from training.metrics import (
    compute_binary_per_class_metrics,
    compute_metrics,
    compute_confusion_matrix,
    find_optimal_threshold,
)
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _json_sanitize(obj: object) -> object:
    """Replace NaN with None for JSON export."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def evaluate(cfg: dict) -> None:
    """Run full evaluation on the held-out test set."""
    device_str = cfg.get("project", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    binary = cfg.get("model", {}).get("binary", False)

    # 1. Load dataset and get test split
    embeddings_dir = Path(cfg["data"]["embeddings_dir"])
    manifest = embeddings_dir / "manifest.csv"
    if manifest.exists():
        dataset = EmbeddingDataset.from_manifest(manifest, binary=binary)
    else:
        dataset = EmbeddingDataset.from_directory(embeddings_dir, binary=binary)

    n_noise_all = int((dataset.labels == 0).sum()) if binary else 0
    n_bird_all = int((dataset.labels == 1).sum()) if binary else 0
    if binary and n_noise_all == 0:
        logger.error(
            "Evaluation: zero noise samples (label 0) in the full embedding set — "
            "bird vs noise metrics are invalid. Use pipeline.mode: full or add raw noise/."
        )

    ds_cfg = cfg.get("dataset", {})
    splits = create_splits(
        dataset,
        val_frac=ds_cfg.get("val_split", 0.15),
        test_frac=ds_cfg.get("test_split", 0.10),
        stratify=ds_cfg.get("stratify", True),
        seed=cfg.get("project", {}).get("seed", 42),
    )

    test_subset = Subset(dataset, splits.test_idx)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # 2. Load model
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    meta_path = chkpt_dir / "best_model_meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    num_classes = 1 if meta["binary"] else len(meta["label_map"])
    model = EmbeddingClassifier(
        input_dim=cfg["embedding"]["embedding_dim"],
        num_classes=num_classes,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)

    chkpt = torch.load(chkpt_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()

    # 3. Run inference on test set
    all_logits, all_labels = [], []

    with torch.no_grad():
        for embs, labels in test_loader:
            embs = embs.to(device)
            logits = model(embs)
            if binary and logits.ndim > 1:
                logits = logits.squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)

    n_noise_test = int((cat_labels == 0).sum()) if binary else 0
    n_bird_test = int((cat_labels == 1).sum()) if binary else 0
    binary_eval_valid = True
    if binary and (n_noise_test == 0 or n_bird_test == 0):
        binary_eval_valid = False
        logger.error(
            f"Binary test split has only one class (noise={n_noise_test}, bird={n_bird_test}) — "
            "confusion matrix, F1, and FPR on noise are not valid. "
            "Increase data or adjust val/test splits."
        )

    # 4. Metrics at default threshold (0.5)
    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)

    metrics_05 = compute_metrics(cat_logits, cat_labels, binary=binary, threshold=0.5)
    print(f"\n--- Metrics at threshold=0.50 ---")
    for k, v in metrics_05.items():
        print(f"  {k:>10s}: {v:.4f}")

    cm_05 = compute_confusion_matrix(cat_logits, cat_labels, binary=binary, threshold=0.5)
    print(f"\n{cm_05}")

    metrics_payload: dict = {"threshold_0.5": metrics_05}
    if binary:
        metrics_payload.update(
            {
                "binary_eval_valid": binary_eval_valid,
                "dataset_noise_count": n_noise_all,
                "dataset_bird_count": n_bird_all,
                "test_noise_count": n_noise_test,
                "test_bird_count": n_bird_test,
            }
        )

    if binary and binary_eval_valid:
        pc05 = compute_binary_per_class_metrics(cat_logits, cat_labels, threshold=0.5)
        print("\n--- Per-class (threshold=0.50) ---")
        for k, v in pc05.items():
            if k == "fpr_noise" and v != v:  # nan
                print(f"  {k:>12s}: nan (no noise in test)")
            else:
                print(f"  {k:>12s}: {v:.4f}")
        metrics_payload["per_class_0.5"] = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in pc05.items()}

    if binary:
        # 5. Optimal threshold search
        result = find_optimal_threshold(cat_logits, cat_labels, metric="f1", steps=50)
        opt_thresh = result["best_threshold"]

        print(f"\n--- Optimal Threshold (max F1) ---")
        print(f"  Threshold: {opt_thresh:.3f}")

        metrics_opt = compute_metrics(cat_logits, cat_labels, binary=True, threshold=opt_thresh)
        for k, v in metrics_opt.items():
            print(f"  {k:>10s}: {v:.4f}")

        cm_opt = compute_confusion_matrix(
            cat_logits, cat_labels, binary=True, threshold=opt_thresh
        )
        print(f"\n{cm_opt}")

        if binary_eval_valid:
            pc_opt = compute_binary_per_class_metrics(
                cat_logits, cat_labels, threshold=opt_thresh
            )
            print(f"\n--- Per-class (optimal F1 threshold={opt_thresh:.3f}) ---")
            for k, v in pc_opt.items():
                print(f"  {k:>12s}: {v:.4f}")
            metrics_payload["optimal_f1_threshold"] = opt_thresh
            metrics_payload["metrics_at_optimal_f1"] = metrics_opt
            metrics_payload["per_class_optimal_f1"] = pc_opt
        else:
            metrics_payload["optimal_f1_threshold"] = opt_thresh
            metrics_payload["metrics_at_optimal_f1"] = metrics_opt

        # 6. Threshold curve
        print(f"\n--- Threshold Curve ---")
        print(f"  {'Thresh':>7s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
        for entry in result["curve"][::5]:  # Every 5th entry
            print(
                f"  {entry['threshold']:7.3f}  "
                f"{entry['precision']:6.4f}  "
                f"{entry['recall']:6.4f}  "
                f"{entry['f1']:6.4f}"
            )

        # 7. Recall-at-precision sweep
        print(f"\n--- Recall at Minimum Precision ---")
        for min_prec in [0.9, 0.8, 0.7, 0.6, 0.5]:
            r = find_optimal_threshold(
                cat_logits, cat_labels,
                metric="recall_at_precision",
                min_precision=min_prec,
                steps=50,
            )
            print(
                f"  min_prec={min_prec:.1f} → threshold={r['best_threshold']:.3f}, "
                f"recall={r['best_value']:.4f}"
            )

    # 8. Per-class breakdown (binary)
    if binary:
        probs = torch.sigmoid(cat_logits).numpy()
        labels_np = cat_labels.numpy()
        bird_probs = probs[labels_np == 1]
        noise_probs = probs[labels_np == 0]

        print(f"\n--- Probability Distribution ---")
        print(f"  Bird  samples: n={len(bird_probs)}, "
              f"mean_prob={bird_probs.mean():.4f}, std={bird_probs.std():.4f}")
        print(f"  Noise samples: n={len(noise_probs)}, "
              f"mean_prob={noise_probs.mean():.4f}, std={noise_probs.std():.4f}")

    results_dir = Path(cfg.get("evaluation", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(metrics_payload), f, indent=2)
    logger.info(f"Wrote metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg)
