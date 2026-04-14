#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation of the trained classifier.

Loads embeddings from manifest.csv, runs classifier inference, and prints
detailed metrics including confusion matrix and threshold curve.

By default uses the held-out **test** split (same seed as training).
Use ``--full-dataset`` to evaluate on **every row** in the manifest (matches
``compute_baseline_metrics.py`` apples-to-apples with BirdNET baseline).

Usage:
    python evaluate.py --config config.yaml
    python evaluate.py --config config.yaml --full-dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset import EmbeddingDataset, create_splits
from models.autoencoder import EmbeddingAutoencoder
from models.classifier import EmbeddingClassifier
from training.metrics import (
    compute_binary_per_class_metrics,
    compute_metrics,
    compute_confusion_matrix,
    find_optimal_threshold,
    metrics_routing_uncertain_as_bird,
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


def evaluate(cfg: dict, *, full_dataset: bool = False) -> None:
    """Run evaluation on the test split (default) or the full embedding manifest."""
    device_str = cfg.get("project", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    binary = cfg.get("model", {}).get("binary", True)

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

    if full_dataset:
        eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        eval_label = "full manifest"
        n_eval = len(dataset)
    else:
        test_subset = Subset(dataset, splits.test_idx)
        eval_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        eval_label = "test split"
        n_eval = len(splits.test_idx)

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

    # 3. Run inference on evaluation subset
    all_logits, all_labels = [], []

    with torch.no_grad():
        for embs, labels in eval_loader:
            embs = embs.to(device, non_blocking=True)
            logits = model(embs)
            if binary and logits.ndim > 1:
                logits = logits.squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)

    n_noise_eval = int((cat_labels == 0).sum()) if binary else 0
    n_bird_eval = int((cat_labels == 1).sum()) if binary else 0
    binary_eval_valid = True
    if binary and (n_noise_eval == 0 or n_bird_eval == 0):
        binary_eval_valid = False
        logger.error(
            f"Binary {eval_label} has only one class (noise={n_noise_eval}, bird={n_bird_eval}) — "
            "confusion matrix, F1, and FPR on noise are not valid. "
            + (
                "Increase data or adjust val/test splits."
                if not full_dataset
                else "Increase data or check manifest labels."
            )
        )

    # 4. Metrics at default threshold (0.5)
    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT  ({eval_label}, n={n_eval})")
    print("=" * 60)

    metrics_05 = compute_metrics(cat_logits, cat_labels, binary=binary, threshold=0.5)
    print(f"\n--- Metrics at threshold=0.50 ---")
    for k, v in metrics_05.items():
        print(f"  {k:>10s}: {v:.4f}")

    cm_05 = compute_confusion_matrix(cat_logits, cat_labels, binary=binary, threshold=0.5)
    print(f"\n{cm_05}")

    metrics_payload: dict = {
        "threshold_0.5": metrics_05,
        "eval_split": "full" if full_dataset else "test",
    }
    if binary:
        metrics_payload.update(
            {
                "binary_eval_valid": binary_eval_valid,
                "dataset_noise_count": n_noise_all,
                "dataset_bird_count": n_bird_all,
                "eval_noise_count": n_noise_eval,
                "eval_bird_count": n_bird_eval,
                # Legacy keys: same counts as eval_* (historically test-split only)
                "test_noise_count": n_noise_eval,
                "test_bird_count": n_bird_eval,
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

        # Routing eval: uncertain → bird (binary pred=1 iff prob > low_threshold)
        inf_cfg = cfg.get("inference", {})
        low_t = float(inf_cfg.get("low_threshold", 0.2))
        high_t = float(inf_cfg.get("high_threshold", 0.6))
        rout = metrics_routing_uncertain_as_bird(cat_logits, cat_labels, low_threshold=low_t)
        print(f"\n--- Routing eval: uncertain → bird (pred bird if prob > low_threshold={low_t}) ---")
        print(f"  (high_threshold={high_t} sets inference folders; eval binary uses low_t only)")
        for k in ("acc", "prec", "rec", "f1"):
            print(f"  {k:>10s}: {rout[k]:.4f}")
        print(f"  TN={rout['tn']} FP={rout['fp']} FN={rout['fn']} TP={rout['tp']}")
        print(f"  FPR (noise→bird): {rout['fpr']:.4f}  FNR (bird missed): {rout['fnr']:.4f}")
        metrics_payload["routing_uncertain_as_bird"] = {
            "low_threshold": low_t,
            "high_threshold": high_t,
            "description": "pred_bird = 1 iff sigmoid(logit) > low_threshold",
            **{k: rout[k] for k in ("acc", "prec", "rec", "f1", "tn", "fp", "fn", "tp", "fpr", "fnr")},
        }

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

    # ── Autoencoder reconstruction error (if enabled) ────────────
    ae_cfg = cfg.get("autoencoder", {})
    if ae_cfg.get("enabled", False):
        ae_chkpt = Path(ae_cfg.get("checkpoint_path", "./checkpoints/autoencoder.pt"))
        ae_meta_path = ae_chkpt.with_name("autoencoder_meta.json")
        if ae_chkpt.exists():
            emb_dim = int(cfg["embedding"]["embedding_dim"])
            ae_model = EmbeddingAutoencoder(input_dim=emb_dim).to(device)
            ae_chk = torch.load(ae_chkpt, map_location=device, weights_only=True)
            ae_model.load_state_dict(ae_chk["model_state_dict"])
            ae_model.eval()

            ae_threshold = 0.01
            if ae_meta_path.exists():
                with open(ae_meta_path, "r") as f:
                    ae_meta = json.load(f)
                ae_threshold = float(ae_meta.get("recon_threshold", 0.01))

            bird_errors, noise_errors = [], []
            with torch.no_grad():
                for embs_batch, labels_batch in eval_loader:
                    embs_batch = embs_batch.to(device, non_blocking=True)
                    recon, _ = ae_model(embs_batch)
                    errors = EmbeddingAutoencoder.compute_reconstruction_error(
                        embs_batch, recon
                    ).cpu()
                    for err_val, lbl in zip(errors.tolist(), labels_batch.tolist()):
                        if lbl == 1:
                            bird_errors.append(err_val)
                        else:
                            noise_errors.append(err_val)

            import numpy as np_ae

            recon_report = {
                "bird_mean": float(np_ae.mean(bird_errors)) if bird_errors else None,
                "bird_std": float(np_ae.std(bird_errors)) if bird_errors else None,
                "noise_mean": float(np_ae.mean(noise_errors)) if noise_errors else None,
                "noise_std": float(np_ae.std(noise_errors)) if noise_errors else None,
                "threshold": ae_threshold,
            }
            metrics_payload["reconstruction_error"] = recon_report

            print("\n--- Autoencoder Reconstruction Error ---")
            if bird_errors:
                print(
                    f"  Bird  samples: n={len(bird_errors)}, "
                    f"mean_err={recon_report['bird_mean']:.6f}, "
                    f"std={recon_report['bird_std']:.6f}"
                )
            if noise_errors:
                print(
                    f"  Noise samples: n={len(noise_errors)}, "
                    f"mean_err={recon_report['noise_mean']:.6f}, "
                    f"std={recon_report['noise_std']:.6f}"
                )
            print(f"  Threshold:     {ae_threshold:.6f}")
        else:
            logger.warning(
                f"Autoencoder enabled but checkpoint not found at {ae_chkpt}. "
                "Skipping reconstruction error metrics."
            )

    results_dir = Path(cfg.get("evaluation", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    prev_path = results_dir / "metrics_previous_run.json"
    if metrics_path.exists():
        try:
            shutil.copy2(metrics_path, prev_path)
            logger.info(f"Backed up previous metrics to {prev_path}")
        except OSError as e:
            logger.warning(f"Could not backup previous metrics: {e}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(metrics_payload), f, indent=2)
    logger.info(f"Wrote metrics to {metrics_path}")

    if prev_path.exists() and metrics_payload.get("routing_uncertain_as_bird"):
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_r = prev_data.get("routing_uncertain_as_bird") or {}
            cur_r = metrics_payload.get("routing_uncertain_as_bird") or {}
            if prev_r.get("fnr") is not None and cur_r.get("fnr") is not None:
                d_fnr = float(prev_r["fnr"]) - float(cur_r["fnr"])
                print("\n--- vs previous run (metrics_previous_run.json) ---")
                print(f"  ΔFNR (routing eval): {d_fnr:+.4f}  (positive = fewer missed birds)")
                print(f"  prev FNR: {prev_r['fnr']:.4f}  →  now FNR: {cur_r['fnr']:.4f}")
                if prev_r.get("fpr") is not None and cur_r.get("fpr") is not None:
                    print(f"  prev FPR: {prev_r['fpr']:.4f}  →  now FPR: {cur_r['fpr']:.4f}")
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug("Could not compare to previous metrics: %s", e)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluate on all manifest rows (not only the held-out test split).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, full_dataset=args.full_dataset)
