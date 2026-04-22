#!/usr/bin/env python3
"""
evaluate.py — Evaluation of the AE-gated classifier (deployment-aligned).

Loads embeddings from manifest.csv, applies autoencoder OOD rejection (τ_AE),
then runs the MLP only on in-distribution embeddings with three-band routing.

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

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

from dataset.dataset import EmbeddingDataset, create_splits
from inference.prediction_api import predict_embeddings_ae_gate, predict_embeddings_mlp
from models.classifier import EmbeddingClassifier
from utils.metrics import (
    compute_metrics_from_preds,
    confusion_rates_binary,
    gated_pred_uncertain_as_bird,
    gated_three_class_predictions,
)
from utils.ae_checkpoint import load_autoencoder_state
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
    """Run evaluation on the test split (default) or the full embedding manifest.

    Ground truth is binary (noise=0, bird=1). The gated head can emit three
    prediction labels (noise / bird / uncertain). **Primary** bird–noise scores
    use standard binary F1 (see ``gated_binary_excluding_uncertain`` and
    ``gated_routing_uncertain_as_bird``), not a macro F1 over three classes,
    because macro F1 with ``labels=[0,1,2]`` mixes an abstention bucket with
    two-way supervision and is not a conventional generalization metric here.
    **Abstention** is reported separately as the fraction of samples routed to
    ``uncertain`` (pred label 2) among all eval rows.
    """
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

    # 2. Load classifier + mandatory autoencoder (binary bird/noise only)
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    meta_path = chkpt_dir / "best_model_meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if not meta.get("binary", True):
        raise RuntimeError(
            "Evaluation with mandatory AE OOD gating requires a binary bird/noise checkpoint "
            "(model.binary=true)."
        )

    model = EmbeddingClassifier(
        input_dim=cfg["embedding"]["embedding_dim"],
        num_classes=1,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)

    chkpt = torch.load(chkpt_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()

    ae_model, tau_ae, ae_thr_meta = load_autoencoder_state(cfg, device)

    inf_cfg = cfg.get("inference", {})
    low_t = float(inf_cfg.get("low_threshold", 0.3))
    high_t = float(inf_cfg.get("high_threshold", 0.7))

    # 3. Gated inference: AE → reject or MLP (MLP not run on OOD embeddings)
    all_probs_list, all_labels_list, all_err_list = [], [], []

    with torch.no_grad():
        for embs, labels in eval_loader:
            embs = embs.to(device, non_blocking=True)
            _ood, recon_err = predict_embeddings_ae_gate(
                ae_model, embs, threshold=tau_ae
            )
            probs_b = torch.zeros(embs.size(0), device=device, dtype=torch.float32)
            pass_m = recon_err <= tau_ae
            if pass_m.any():
                probs_b[pass_m] = predict_embeddings_mlp(
                    model, embs[pass_m], binary=True
                )
            all_probs_list.append(probs_b.cpu())
            all_err_list.append(recon_err.cpu())
            all_labels_list.append(labels.cpu())

    cat_probs_t = torch.cat(all_probs_list)
    cat_labels = torch.cat(all_labels_list)
    recon_errors = torch.cat(all_err_list).numpy().astype(np.float64)

    probs = cat_probs_t.numpy().astype(np.float64)
    pass_ae = recon_errors <= tau_ae

    pred_3, ae_reject = gated_three_class_predictions(
        recon_errors, tau_ae, probs, low_t, high_t
    )
    n_ood = int(ae_reject.sum())
    ood_rate = float(n_ood / len(ae_reject)) if len(ae_reject) else 0.0
    n_pass_mlp = int((~ae_reject).sum())

    y_true = cat_labels.numpy().astype(np.int64)

    n_noise_eval = int((cat_labels == 0).sum())
    n_bird_eval = int((cat_labels == 1).sum())
    binary_eval_valid = True
    if n_noise_eval == 0 or n_bird_eval == 0:
        binary_eval_valid = False
        logger.error(
            f"Binary {eval_label} has only one class (noise={n_noise_eval}, bird={n_bird_eval}) — "
            "gated metrics are not valid. "
            + (
                "Increase data or adjust val/test splits."
                if not full_dataset
                else "Increase data or check manifest labels."
            )
        )

    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT  ({eval_label}, n={n_eval})")
    print("  Pipeline: Embedding → AE → (OOD reject | MLP) → three-band routing")
    print("=" * 60)

    print("\n--- OOD gating (autoencoder) ---")
    print(f"  τ_AE:                  {tau_ae:.6f}")
    print(f"  Samples rejected:      {n_ood}  ({100.0 * ood_rate:.2f}% of eval set)")
    print(f"  Passed to MLP:         {n_pass_mlp}")

    metrics_payload: dict = {
        "pipeline": "ae_ood_gate_then_mlp",
        "eval_split": "full" if full_dataset else "test",
        "tau_ae": tau_ae,
        "autoencoder_threshold_meta": ae_thr_meta,
        "ood_gating": {
            "n_rejected": n_ood,
            "rejection_rate": ood_rate,
            "n_passed_to_mlp": n_pass_mlp,
            "pct_rejected_before_classifier": 100.0 * ood_rate,
        },
        "binary_eval_valid": binary_eval_valid,
        "dataset_noise_count": n_noise_all,
        "dataset_bird_count": n_bird_all,
        "eval_noise_count": n_noise_eval,
        "eval_bird_count": n_bird_eval,
        "test_noise_count": n_noise_eval,
        "test_bird_count": n_bird_eval,
    }

    acc_3 = accuracy_score(y_true, pred_3)
    cm3 = confusion_matrix(y_true, pred_3, labels=[0, 1, 2])
    abstention_rate = float((pred_3 == 2).mean()) if len(pred_3) else 0.0

    print("\n--- Gated 3-class (noise / bird / uncertain) ---")
    print(f"  accuracy: {acc_3:.4f}")
    print(f"  abstention_rate (pred uncertain): {abstention_rate:.4f}")
    print("  Confusion matrix (rows=true 0=noise,1=bird; cols=pred 0=noise,1=bird,2=uncertain):")
    print(cm3)

    metrics_payload["gated_three_class"] = {
        "accuracy": float(acc_3),
        "abstention_rate": abstention_rate,
        "confusion_matrix_labels": [0, 1, 2],
        "confusion_matrix": cm3.tolist(),
    }

    if binary_eval_valid:
        mask_ex = pred_3 != 2
        if mask_ex.any():
            m_bin = compute_metrics_from_preds(y_true[mask_ex], pred_3[mask_ex])
            cr_bin = confusion_rates_binary(y_true[mask_ex], pred_3[mask_ex])
            print("\n--- Binary subset (excluding uncertain predictions) ---")
            for k, v in m_bin.items():
                print(f"  {k:>10s}: {v:.4f}")
            print(f"  TN={cr_bin['tn']} FP={cr_bin['fp']} FN={cr_bin['fn']} TP={cr_bin['tp']}")
            metrics_payload["gated_binary_excluding_uncertain"] = {**m_bin, **cr_bin}

        rout_g = gated_pred_uncertain_as_bird(recon_errors, tau_ae, probs, low_t)
        cr_r = confusion_rates_binary(y_true, rout_g)
        m_r = compute_metrics_from_preds(y_true, rout_g)
        print(f"\n--- Routing eval (AE + pred bird if prob > low_threshold={low_t}) ---")
        for k in ("acc", "prec", "rec", "f1"):
            print(f"  {k:>10s}: {m_r[k]:.4f}")
        print(f"  TN={cr_r['tn']} FP={cr_r['fp']} FN={cr_r['fn']} TP={cr_r['tp']}")
        print(f"  FPR (noise→bird): {cr_r['fpr']:.4f}  FNR (bird missed): {cr_r['fnr']:.4f}")
        metrics_payload["gated_routing_uncertain_as_bird"] = {
            "low_threshold": low_t,
            "high_threshold": high_t,
            "description": "OOD→noise; else pred_bird = 1 iff sigmoid(logit) > low_threshold",
            **{k: m_r[k] for k in ("acc", "prec", "rec", "f1")},
            **{k: cr_r[k] for k in ("tn", "fp", "fn", "tp", "fpr", "fnr")},
        }

    if pass_ae.any():
        pp = probs[pass_ae]
        lt = y_true[pass_ae]
        bird_probs = pp[lt == 1]
        noise_probs = pp[lt == 0]
        print(f"\n--- MLP probability (AE-passed only) ---")
        if len(bird_probs):
            print(f"  Bird  samples: n={len(bird_probs)}, "
                  f"mean_prob={bird_probs.mean():.4f}, std={bird_probs.std():.4f}")
        else:
            print("  Bird  samples: n=0")
        if len(noise_probs):
            print(f"  Noise samples: n={len(noise_probs)}, "
                  f"mean_prob={noise_probs.mean():.4f}, std={noise_probs.std():.4f}")
        else:
            print("  Noise samples: n=0")

    bird_errors = recon_errors[y_true == 1]
    noise_errors = recon_errors[y_true == 0]
    recon_report = {
        "bird_mean": float(np.mean(bird_errors)) if bird_errors.size else None,
        "bird_std": float(np.std(bird_errors)) if bird_errors.size else None,
        "noise_mean": float(np.mean(noise_errors)) if noise_errors.size else None,
        "noise_std": float(np.std(noise_errors)) if noise_errors.size else None,
        "threshold_tau_ae": tau_ae,
    }
    metrics_payload["reconstruction_error"] = recon_report

    print("\n--- Reconstruction error (all eval samples) ---")
    if bird_errors.size:
        print(f"  Bird  samples: n={bird_errors.size}, mean_err={recon_report['bird_mean']:.6f}")
    if noise_errors.size:
        print(f"  Noise samples: n={noise_errors.size}, mean_err={recon_report['noise_mean']:.6f}")

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

    if prev_path.exists() and metrics_payload.get("gated_routing_uncertain_as_bird"):
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_r = prev_data.get("gated_routing_uncertain_as_bird") or {}
            cur_r = metrics_payload.get("gated_routing_uncertain_as_bird") or {}
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
