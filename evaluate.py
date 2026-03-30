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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset import EmbeddingDataset, create_splits
from models.classifier import EmbeddingClassifier
from training.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    find_optimal_threshold,
)
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


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

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg)
