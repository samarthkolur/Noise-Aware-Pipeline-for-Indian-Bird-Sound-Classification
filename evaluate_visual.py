#!/usr/bin/env python3
"""
evaluate_visual.py — Visual evaluation: plots + JSON export.

Reuses training.metrics (compute_metrics, compute_confusion_matrix, find_optimal_threshold).
Does not modify training, embeddings, or BirdNET.

Usage:
    python evaluate_visual.py --config config.yaml
    python evaluate_visual.py --config config.yaml --compare-json results/metrics_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

from dataset.dataset import EmbeddingDataset, create_splits
from models.classifier import EmbeddingClassifier
from training.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    find_optimal_threshold,
)
from utils.config import load_config


def _resolve_device(cfg: dict) -> torch.device:
    device_str = cfg.get("project", {}).get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model_and_logits(cfg: dict) -> Tuple[torch.Tensor, torch.Tensor, bool, torch.device]:
    """Load test split, run classifier, return logits, labels, binary flag, device."""
    device = _resolve_device(cfg)
    binary = cfg.get("model", {}).get("binary", False)

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

    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    with open(chkpt_dir / "best_model_meta.json", "r") as f:
        meta = json.load(f)

    num_classes = 1 if meta["binary"] else len(meta["label_map"])
    model = EmbeddingClassifier(
        input_dim=cfg["embedding"]["embedding_dim"],
        num_classes=num_classes,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)

    chkpt = torch.load(
        chkpt_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for embs, labels in test_loader:
            embs = embs.to(device)
            logits = model(embs)
            if binary and logits.ndim > 1:
                logits = logits.squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels), binary, device


def _metrics_to_user_dict(m: Dict[str, float]) -> Dict[str, float]:
    """Map internal keys (acc, prec, rec, f1) to JSON-friendly names."""
    return {
        "accuracy": float(m["acc"]),
        "precision": float(m["prec"]),
        "recall": float(m["rec"]),
        "f1": float(m["f1"]),
    }


def _numeric_confusion_matrix(
    logits: torch.Tensor, labels: torch.Tensor, binary: bool, threshold: float
) -> np.ndarray:
    """2D array for heatmap (rows = actual, cols = predicted)."""
    labels_np = labels.cpu().numpy()
    if binary:
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds_np = (probs > threshold).cpu().numpy().astype(int)
        return confusion_matrix(labels_np, preds_np, labels=[0, 1])
    preds_np = logits.argmax(dim=1).cpu().numpy()
    u = np.unique(np.concatenate([labels_np, preds_np]))
    u.sort()
    return confusion_matrix(labels_np, preds_np, labels=u)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    binary: bool,
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Rows = actual, cols = predicted. Binary: 0=noise, 1=bird."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0)

    if binary:
        tick_labels = ["Noise (0)", "Bird (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    else:
        n = cm.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([str(i) for i in range(n)])
        ax.set_yticklabels([str(i) for i in range(n)])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics: Dict[str, float], out_path: Path, title: str) -> None:
    names = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["acc", "prec", "rec", "f1"]
    values = [metrics[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=["#4472c4", "#ed7d31", "#a5a5a5", "#ffc000"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_comparison(
    m05: Dict[str, float],
    m_best: Dict[str, float],
    out_path: Path,
) -> None:
    """Grouped bars: X = ['0.5 threshold', 'Best F1 threshold']; 4 bars per group."""
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["acc", "prec", "rec", "f1"]
    m05_vals = [m05[k] for k in keys]
    m_best_vals = [m_best[k] for k in keys]

    x_groups = np.arange(2)
    group_names = ["0.5 threshold", "Best F1 threshold"]
    fig, ax = plt.subplots(figsize=(9, 6))
    n_metrics = 4
    width = 0.18
    for i, ml in enumerate(metric_labels):
        offset = (i - (n_metrics - 1) / 2.0) * width
        vals = [m05_vals[i], m_best_vals[i]]
        ax.bar(x_groups + offset, vals, width, label=ml)

    ax.set_xticks(x_groups)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Threshold comparison")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_debug_cm(cm: np.ndarray, label: str) -> None:
    print(f"\n[DEBUG] {label} (rows=actual, cols=predicted):")
    print(np.array2string(cm, separator=", "))


def _print_debug_metrics(m: Dict[str, float], label: str) -> None:
    print(f"\n[DEBUG] {label}:")
    for k, v in m.items():
        print(f"  {k}: {v:.6f}")


def _optional_compare(current: Dict[str, Any], baseline_path: Path) -> None:
    """Print side-by-side comparison if baseline metrics.json exists."""
    if not baseline_path.is_file():
        print(f"[compare-json] File not found: {baseline_path}")
        return
    with open(baseline_path, "r") as f:
        base = json.load(f)

    print("\n" + "=" * 60)
    print("  OPTIONAL: comparison vs", baseline_path)
    print("=" * 60)
    for section in ("threshold_0.5", "best_threshold"):
        if section not in current or section not in base:
            continue
        print(f"\n--- {section} ---")
        cur_s = current[section]
        base_s = base[section]
        for key in ("accuracy", "precision", "recall", "f1"):
            if key in cur_s and key in base_s:
                d = float(cur_s[key]) - float(base_s[key])
                print(f"  {key}: baseline={base_s[key]:.4f}  current={cur_s[key]:.4f}  Δ={d:+.4f}")
        if section == "best_threshold" and "threshold" in cur_s and "threshold" in base_s:
            print(
                f"  threshold: baseline={base_s.get('threshold')}  "
                f"current={cur_s.get('threshold')}"
            )


def run_visual_evaluation(cfg: dict, compare_json: Optional[Path] = None) -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    cat_logits, cat_labels, binary, _ = _load_model_and_logits(cfg)

    metrics_05 = compute_metrics(cat_logits, cat_labels, binary=binary, threshold=0.5)
    cm05 = _numeric_confusion_matrix(cat_logits, cat_labels, binary, 0.5)

    print(
        "\n[DEBUG] compute_confusion_matrix() text @ threshold=0.5:\n",
        compute_confusion_matrix(cat_logits, cat_labels, binary=binary, threshold=0.5),
    )
    _print_debug_cm(cm05, "Numeric confusion matrix @ threshold=0.5")
    _print_debug_metrics(metrics_05, "Metrics @ threshold=0.5")

    # Heatmap (use 0.5 threshold CM for consistency with threshold_0.5 block)
    plot_confusion_matrix_heatmap(
        cm05,
        binary=binary,
        out_path=results_dir / "confusion_matrix.png",
        title="Confusion Matrix",
    )

    # Bar chart: use best-F1 metrics when binary (clearer headline performance); else 0.5
    if binary:
        result = find_optimal_threshold(cat_logits, cat_labels, metric="f1", steps=50)
        opt_thresh = float(result["best_threshold"])
        metrics_opt = compute_metrics(
            cat_logits, cat_labels, binary=True, threshold=opt_thresh
        )
        cm_opt = _numeric_confusion_matrix(cat_logits, cat_labels, binary, opt_thresh)

        print(
            f"\n[DEBUG] compute_confusion_matrix() text @ best F1 threshold ({opt_thresh:.4f}):\n",
            compute_confusion_matrix(
                cat_logits, cat_labels, binary=True, threshold=opt_thresh
            ),
        )
        _print_debug_cm(cm_opt, f"Numeric confusion matrix @ best F1 threshold ({opt_thresh:.4f})")
        _print_debug_metrics(metrics_opt, "Metrics @ best F1 threshold")

        plot_metrics_bar(
            metrics_opt,
            results_dir / "metrics_bar_chart.png",
            title="Model Performance Metrics",
        )

        plot_threshold_comparison(
            metrics_05,
            metrics_opt,
            results_dir / "threshold_comparison.png",
        )

        payload: Dict[str, Any] = {
            "threshold_0.5": _metrics_to_user_dict(metrics_05),
            "best_threshold": {
                "threshold": opt_thresh,
                **_metrics_to_user_dict(metrics_opt),
            },
        }
    else:
        plot_metrics_bar(
            metrics_05,
            results_dir / "metrics_bar_chart.png",
            title="Model Performance Metrics",
        )
        # Multiclass: no single F1-optimal threshold in this script
        m_user = _metrics_to_user_dict(metrics_05)
        payload = {
            "threshold_0.5": m_user,
            "best_threshold": {
                "threshold": None,
                **m_user,
            },
        }
        print(
            "\n[Note] Multiclass mode: threshold_comparison.png not created "
            "(comparison applies to binary bird/noise only)."
        )

    out_json = results_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("  VISUAL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Saved: {out_json}")
    print(f"  Saved: {results_dir / 'confusion_matrix.png'}")
    print(f"  Saved: {results_dir / 'metrics_bar_chart.png'}")
    if binary:
        print(f"  Saved: {results_dir / 'threshold_comparison.png'}")
    print()

    if compare_json is not None:
        _optional_compare(payload, compare_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual evaluation with plots and metrics JSON"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--compare-json",
        type=str,
        default=None,
        help="Optional path to another metrics.json for a printed baseline comparison",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    compare_path = Path(args.compare_json) if args.compare_json else None
    run_visual_evaluation(cfg, compare_json=compare_path)


if __name__ == "__main__":
    main()
