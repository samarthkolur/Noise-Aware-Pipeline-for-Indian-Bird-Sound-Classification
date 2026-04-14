#!/usr/bin/env python3
"""
compute_baseline_metrics.py
Compare BirdNET baseline vs Noise-Aware Pipeline (strict full-manifest comparison).

Baseline predictions: comparison/baseline_normalized.jsonl (confidence thresholded)
Pipeline metrics:     recomputed from trained classifier on FULL manifest.csv
Ground truth:         data/embeddings/manifest.csv (noise/ → 0, species → 1)

Key mapping fix: noise segments in the manifest are filed under
  data/processed/noise/{OrigSpecies}__{OrigFileNum}_segXXXX.wav
but BirdNET JSONL keys use the original path structure:
  iBC53/{OrigSpecies}/{OrigFileNum}.wav
The loader recovers the original (species, filenum, seg_idx) key for noise entries.
"""

import json
import argparse
from pathlib import Path
from typing import Tuple
from collections import Counter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None
import numpy as np


def load_ground_truth(manifest_path: str) -> dict:
    """Load binary labels keyed to match BirdNET JSONL structure.

    Bird segments:  key = (species_folder, file_stem, seg_idx)
    Noise segments: filename encodes original source as SpeciesName__FileNum;
                    key = (OrigSpecies, OrigFileNum, seg_idx) to align with JSONL.
    """
    ground_truth = {}
    parse_warnings = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            species = parts[0]
            src_file = parts[1]
            seg_idx = int(parts[2])

            label = 0 if species.lower() == "noise" else 1

            p = Path(src_file)
            folder = p.parent.name
            file_stem = p.name.split("_seg")[0]

            if label == 0 and "__" in file_stem:
                orig_species, orig_filenum = file_stem.rsplit("__", 1)
                key = (orig_species, orig_filenum, seg_idx)
            else:
                key = (folder, file_stem, seg_idx)

            if key in ground_truth and ground_truth[key] != label:
                parse_warnings += 1
            ground_truth[key] = label

    if parse_warnings:
        print(f"  ⚠ {parse_warnings} key collisions between bird/noise segments")
    return ground_truth


def load_baseline_predictions(jsonl_path: str) -> dict:
    """Load BirdNET detections, keyed by (folder, file_stem, seg_idx).
    Multiple detections per segment → keep max confidence."""
    predictions = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            src = d["source_file"].replace("\\", "/")
            parts = src.split("/")
            if len(parts) < 2:
                continue
            folder = parts[-2]
            file_stem = parts[-1].split(".")[0]
            seg_idx = int(float(d["start_sec"]) // 3)
            conf = float(d["confidence"])

            key = (folder, file_stem, seg_idx)
            predictions[key] = max(predictions.get(key, 0.0), conf)
    return predictions


def compute_confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, fpr=fpr, fnr=fnr,
                tp=tp, tn=tn, fp=fp, fn=fn)


def compute_pipeline_metrics(config_path: str = "config.yaml") -> Tuple[dict, int, int, int]:
    """Recompute pipeline metrics from the trained classifier on the FULL manifest.

    This is required for a strict apples-to-apples comparison, because the BirdNET
    baseline metrics are computed over every key in manifest.csv.
    """
    import torch
    from dataset.dataset import EmbeddingDataset
    from models.classifier import EmbeddingClassifier
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix as sklearn_cm

    cfg_mod = __import__("utils.config", fromlist=["load_config"])
    cfg = cfg_mod.load_config(config_path)
    device = torch.device("cpu")

    ds = EmbeddingDataset.from_manifest(
        Path(cfg["data"]["embeddings_dir"]) / "manifest.csv", binary=True)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    model = EmbeddingClassifier(
        input_dim=cfg["embedding"]["embedding_dim"],
        num_classes=1,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)
    chk = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for embs, labels in loader:
            logits = model(embs.to(device))
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    probs = torch.sigmoid(logits_t).numpy()
    y = labels_t.numpy()
    # Match inference + evaluate.py: uncertain → bird (pred bird iff prob > low_threshold)
    low_thr = float(cfg.get("inference", {}).get("low_threshold", 0.2))
    preds = (probs > low_thr).astype(int)

    tn, fp, fn, tp = sklearn_cm(y, preds).ravel()
    result = metrics_from_confusion(int(tp), int(tn), int(fp), int(fn))
    n_noise = int((y == 0).sum())
    n_bird = int((y == 1).sum())
    n_total = int(len(y))
    return result, n_noise, n_bird, n_total


def plot_metrics_comparison(baseline: dict, pipeline: dict, out_path: str):
    if plt is None:
        print("  [skip plots] matplotlib not installed")
        return
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    b_vals = [baseline[k] for k in ("accuracy", "precision", "recall", "f1")]
    p_vals = [pipeline[k] for k in ("accuracy", "precision", "recall", "f1")]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline (Raw BirdNET)", color="#4C72B0")
    bars_p = ax.bar(x + w / 2, p_vals, w, label="Noise-Aware Pipeline", color="#55A868")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance: Raw BirdNET vs Noise-Aware Pipeline", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)

    for bars in (bars_b, bars_p):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_error_comparison(baseline: dict, pipeline: dict, out_path: str):
    if plt is None:
        print("  [skip plots] matplotlib not installed")
        return
    labels = ["False Positive Rate\n(Noise → Bird)", "False Negative Rate\n(Bird Missed)"]
    b_vals = [baseline.get("fpr", 0), baseline.get("fnr", 0)]
    p_vals = [pipeline.get("fpr", 0), pipeline.get("fnr", 0)]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline (BirdNET)", color="#C44E52")
    bars_p = ax.bar(x + w / 2, p_vals, w, label="Noise-Aware Pipeline", color="#8172B3")

    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Error Rate Comparison", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)

    for bars in (bars_b, bars_p):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BirdNET baseline vs Pipeline comparison")
    parser.add_argument("--manifest", default="data/embeddings/manifest.csv")
    parser.add_argument("--baseline", default="comparison/baseline_normalized.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for baseline BirdNET (default: 0.5)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Pipeline config (for pipeline-side low_threshold)")
    args = parser.parse_args()

    print("=" * 60)
    print("  BASELINE vs PIPELINE COMPARISON")
    print("=" * 60)

    # ── 1. Load & validate data ──────────────────────────────────
    gt = load_ground_truth(args.manifest)
    bl = load_baseline_predictions(args.baseline)

    gt_labels = Counter(gt.values())
    print(f"\n--- Step 1: Validate Data ---")
    print(f"  Total samples:         {len(gt)}")
    print(f"  Bird  (y_true == 1):   {gt_labels[1]}")
    print(f"  Noise (y_true == 0):   {gt_labels[0]}")
    if gt_labels[0] == 0:
        print("  STOP: FPR cannot be computed — zero noise samples")
        return

    # ── 2. Alignment ─────────────────────────────────────────────
    gt_keys = set(gt.keys())
    bl_keys = set(bl.keys())
    matched = gt_keys & bl_keys
    noise_matched = sum(1 for k in matched if gt[k] == 0)
    noise_unmatched = gt_labels[0] - noise_matched

    print(f"\n--- Step 2: Alignment ---")
    print(f"  GT ∩ Baseline:         {len(matched)}")
    print(f"  GT-only (no detect):   {len(gt_keys - bl_keys)}")
    print(f"  Baseline-only:         {len(bl_keys - gt_keys)}")
    print(f"  Noise segs matched:    {noise_matched}/{gt_labels[0]}")
    print(f"  Noise segs unmatched:  {noise_unmatched} (default conf=0 → pred=0)")

    # ── 3. Build aligned predictions ─────────────────────────────
    y_true, y_pred_baseline = [], []
    for key, label in gt.items():
        conf = bl.get(key, 0.0)
        pred = 1 if conf >= args.threshold else 0
        y_true.append(label)
        y_pred_baseline.append(pred)

    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred_baseline)

    print(f"\n--- Step 2b: Validate Predictions ---")
    print(f"  Unique y_true:         {sorted(set(y_true))}")
    print(f"  Unique y_pred_base:    {sorted(set(y_pred_baseline))}")
    print(f"  y_true  distribution:  {{0: {true_dist[0]}, 1: {true_dist[1]}}}")
    print(f"  y_pred  distribution:  {{0: {pred_dist[0]}, 1: {pred_dist[1]}}}")
    assert 0 in pred_dist and 1 in pred_dist, "y_pred must contain both 0 and 1"

    # ── 4. Baseline confusion matrix ─────────────────────────────
    tp, tn, fp, fn = compute_confusion(y_true, y_pred_baseline)
    base = metrics_from_confusion(tp, tn, fp, fn)

    print(f"\n--- Step 3: Baseline Confusion Matrix ---")
    print(f"  [[TN={tn}, FP={fp}],")
    print(f"   [FN={fn}, TP={tp}]]")

    # ── 5. Verify logic ──────────────────────────────────────────
    print(f"\n--- Step 4: Verify Logic ---")
    print(f"  FP={fp} (noise predicted as bird): {'> 0 ✓' if fp > 0 else '= 0 ⚠'}")

    # ── 6. Baseline metrics ──────────────────────────────────────
    print(f"\n--- Step 5: Baseline Error Rates (threshold={args.threshold}) ---")
    print(f"  Accuracy:  {base['accuracy']:.4f}")
    print(f"  Precision: {base['precision']:.4f}")
    print(f"  Recall:    {base['recall']:.4f}")
    print(f"  F1:        {base['f1']:.4f}")
    if (fp + tn) == 0:
        print(f"  FPR:       N/A (no noise samples in denominator)")
    else:
        print(f"  FPR:       {base['fpr']:.4f}  = {fp}/({fp}+{tn})")
    if (fn + tp) == 0:
        print(f"  FNR:       N/A (no bird samples in denominator)")
    else:
        print(f"  FNR:       {base['fnr']:.4f}  = {fn}/({fn}+{tp})")

    # ── 7. Pipeline metrics (recomputed from model) ──────────────
    from utils.config import load_config as _lc

    _cfg = _lc(args.config)
    low_thr = float(_cfg.get("inference", {}).get("low_threshold", 0.2))
    print(
        f"\n--- Step 5b: Pipeline Metrics (recomputed from classifier, pred bird if prob > {low_thr}) ---"
    )
    pipe, pipe_n_noise, pipe_n_bird, pipe_n_total = compute_pipeline_metrics(args.config)
    pipe_tp, pipe_tn = pipe["tp"], pipe["tn"]
    pipe_fp, pipe_fn = pipe["fp"], pipe["fn"]

    print(f"  Full set: {pipe_n_bird} bird + {pipe_n_noise} noise (total={pipe_n_total})")
    print(f"  [[TN={pipe_tn}, FP={pipe_fp}],")
    print(f"   [FN={pipe_fn}, TP={pipe_tp}]]")
    print(f"  Accuracy:  {pipe['accuracy']:.4f}")
    print(f"  Precision: {pipe['precision']:.4f}")
    print(f"  Recall:    {pipe['recall']:.4f}")
    print(f"  F1:        {pipe['f1']:.4f}")
    print(f"  FPR:       {pipe['fpr']:.4f}  = {pipe_fp}/({pipe_fp}+{pipe_tn})")
    print(f"  FNR:       {pipe['fnr']:.4f}  = {pipe_fn}/({pipe_fn}+{pipe_tp})")

    # ── 8. Sanity check ──────────────────────────────────────────
    print(f"\n--- Step 6: Sanity Check ---")
    print(f"  Baseline FPR: {base['fpr']:.4f}")
    print(f"  Pipeline FPR: {pipe['fpr']:.4f}")
    if base["fpr"] > pipe["fpr"]:
        print(f"  ✓ Baseline FPR > Pipeline FPR (pipeline rejects noise better)")
    else:
        print(f"  ⚠ Baseline FPR ≤ Pipeline FPR (see explanation below)")

    print(f"\n  Baseline FNR: {base['fnr']:.4f}")
    print(f"  Pipeline FNR: {pipe['fnr']:.4f}")
    if base["fnr"] > pipe["fnr"]:
        print(f"  ✓ Baseline FNR > Pipeline FNR (pipeline finds more birds)")

    # ── 9. Save JSON outputs ─────────────────────────────────────
    Path("comparison").mkdir(exist_ok=True)
    Path("results/comparison_graphs").mkdir(parents=True, exist_ok=True)

    with open("comparison/baseline_metrics.json", "w") as f:
        json.dump(base, f, indent=2)

    pipe_out = {k: pipe[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr",
                                       "tp", "tn", "fp", "fn")}
    with open("comparison/pipeline_metrics.json", "w") as f:
        json.dump(pipe_out, f, indent=2)

    comp = {"Baseline (BirdNET)": base, "Pipeline (Noise-Aware)": pipe_out}
    with open("comparison/comparison_table.json", "w") as f:
        json.dump(comp, f, indent=2)

    print(f"\n--- Saved ---")
    print(f"  comparison/baseline_metrics.json")
    print(f"  comparison/pipeline_metrics.json")
    print(f"  comparison/comparison_table.json")

    # ── 10. Plots ─────────────────────────────────────────────────
    print(f"\n--- Step 7: Plots ---")
    plot_metrics_comparison(base, pipe_out, "results/comparison_graphs/metrics_comparison.png")
    plot_error_comparison(base, pipe_out, "results/comparison_graphs/error_comparison.png")

    print(f"\n{'=' * 60}")
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
