#!/usr/bin/env python3
"""
compute_real_baseline.py — Generate BirdNET baseline predictions using the
BirdNET V2.4 TFLite classifier (not just embeddings).

Uses the same model file the pipeline uses, but reads the **classification
output** (not the embedding layer).  For each processed segment, the top
BirdNET species confidence is treated as the "bird detection confidence."
If max(confidence) >= threshold → pred=bird (1), else pred=noise (0).

This provides a fair baseline: same audio segments, same model, same threshold.

Outputs:
    comparison/baseline_normalized.jsonl   — per-segment JSONL (for research suite)
    comparison/baseline_metrics.json       — summary metrics
    results/comparison_graphs/             — bar chart comparisons
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config import load_config
from utils.metrics import metrics_from_confusion, confusion_binary


def _resolve_device():
    """Same device resolution as the pipeline."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_birdnet_classifier(cfg: dict):
    """Load the BirdNET TFLite model for classification (full output, not embeddings)."""
    from embedding.embedding import _resolve_birdnet_model_path
    import ai_edge_litert.interpreter as tflite

    model_path = str(_resolve_birdnet_model_path(cfg))
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _classify_segment(interpreter, waveform: np.ndarray, sr: int) -> float:
    """Run BirdNET classification and return max confidence across all species.

    BirdNET V2.4 outputs a probability distribution over ~6k species.
    For binary bird/noise detection, we take max(confidence) — if any species
    is detected with confidence >= threshold, the segment is "bird."
    """
    # Prepare input: BirdNET expects 48kHz, 3s, mono
    expected_samples = 48000 * 3
    if len(waveform) < expected_samples:
        waveform = np.pad(waveform, (0, expected_samples - len(waveform)))
    elif len(waveform) > expected_samples:
        waveform = waveform[:expected_samples]

    waveform = waveform.astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = waveform.reshape(input_details[0]["shape"])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Get classification output (last output tensor = species probabilities)
    output = interpreter.get_tensor(output_details[0]["index"])
    probs = output.flatten()

    # Return max confidence across all species
    return float(np.max(probs))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BirdNET baseline on processed segments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = args.threshold

    # Load manifest
    manifest_path = Path(cfg["data"]["embeddings_dir"]) / "manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("Run the pipeline (preprocess → embed) first.", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r") as f:
        rows = list(csv.DictReader(f))

    print(f"Dataset: {len(rows)} segments from {manifest_path}")

    # Load BirdNET model
    print("Loading BirdNET V2.4 TFLite classifier...")
    interpreter = _load_birdnet_classifier(cfg)
    print("BirdNET loaded ✓")

    # Process each segment
    import soundfile as sf
    from tqdm import tqdm

    jsonl_records: List[dict] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    confidences: List[float] = []
    errors = 0

    for row in tqdm(rows, desc="BirdNET baseline"):
        source_file = row["source_file"]
        species = row["species"]
        is_bird_true = 0 if species.lower() == "noise" else 1

        try:
            waveform, sr = sf.read(source_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            conf = _classify_segment(interpreter, waveform, sr)
            is_bird_pred = 1 if conf >= threshold else 0

            y_true_all.append(is_bird_true)
            y_pred_all.append(is_bird_pred)
            confidences.append(conf)

            # Build JSONL record compatible with compute_baseline_metrics.py
            seg_idx = int(row.get("segment_index", 0))
            jsonl_records.append({
                "run": "baseline",
                "source_file": source_file,
                "segment_index": seg_idx,
                "start_sec": float(seg_idx) * 3.0,
                "end_sec": float(seg_idx + 1) * 3.0,
                "confidence": conf,
                "species_code": species,
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {Path(source_file).name}: {e}")

    if errors > 5:
        print(f"  ... and {errors - 5} more errors")

    # Write JSONL
    comp_dir = Path("comparison")
    comp_dir.mkdir(exist_ok=True)
    jsonl_path = comp_dir / "baseline_normalized.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(jsonl_records)} records → {jsonl_path}")

    # Compute metrics
    y_true = np.array(y_true_all, dtype=np.int64)
    y_pred = np.array(y_pred_all, dtype=np.int64)
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)
    base = metrics_from_confusion(tp, tn, fp, fn)

    n_bird = int((y_true == 1).sum())
    n_noise = int((y_true == 0).sum())

    print(f"\n{'='*60}")
    print(f"  BirdNET V2.4 BASELINE (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  Total segments: {len(y_true)} (bird={n_bird}, noise={n_noise})")
    print(f"  [[TN={tn:4d}, FP={fp:4d}],")
    print(f"   [FN={fn:4d}, TP={tp:4d}]]")
    print(f"  Accuracy:  {base['accuracy']:.4f}")
    print(f"  Precision: {base['precision']:.4f}")
    print(f"  Recall:    {base['recall']:.4f}")
    print(f"  F1:        {base['f1']:.4f}")
    print(f"  FPR:       {base['fpr']:.4f}")
    print(f"  FNR:       {base['fnr']:.4f}")

    # Save baseline metrics
    base_out = {**base, "threshold": threshold, "n_total": len(y_true),
                "n_bird": n_bird, "n_noise": n_noise}
    with open(comp_dir / "baseline_metrics.json", "w") as f:
        json.dump(base_out, f, indent=2)

    # Load pipeline metrics from evaluate.py output
    pipeline_metrics_path = Path("results/metrics.json")
    if pipeline_metrics_path.exists():
        with open(pipeline_metrics_path) as f:
            pipe_raw = json.load(f)

        # Use gated routing metrics for comparison
        pipe_routing = pipe_raw.get("gated_routing_uncertain_as_bird", {})
        pipe = {
            "accuracy": pipe_routing.get("acc", 0),
            "precision": pipe_routing.get("prec", 0),
            "recall": pipe_routing.get("rec", 0),
            "f1": pipe_routing.get("f1", 0),
            "fpr": pipe_routing.get("fpr", 0),
            "fnr": pipe_routing.get("fnr", 0),
        }
    else:
        print("  WARNING: results/metrics.json not found, using zeros for pipeline")
        pipe = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "fpr": 0, "fnr": 0}

    with open(comp_dir / "pipeline_metrics.json", "w") as f:
        json.dump(pipe, f, indent=2)

    comp = {
        "note": f"Both evaluated on same segments. BirdNET threshold={threshold}.",
        "Baseline (BirdNET)": base_out,
        "Pipeline (Noise-Aware)": pipe,
    }
    with open(comp_dir / "comparison_table.json", "w") as f:
        json.dump(comp, f, indent=2)

    # Generate comparison graphs
    _generate_graphs(base, pipe, threshold, len(y_true))

    print(f"\n  Saved → comparison/ and results/comparison_graphs/")
    print(f"{'='*60}")


def _generate_graphs(base: dict, pipe: dict, threshold: float, n_total: int):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plots] matplotlib not installed")
        return

    Path("results/comparison_graphs").mkdir(parents=True, exist_ok=True)

    # Metrics comparison
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    b_vals = [base[k] for k in ("accuracy", "precision", "recall", "f1")]
    p_vals = [pipe[k] for k in ("accuracy", "precision", "recall", "f1")]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_b = ax.bar(x - w/2, b_vals, w, label="Baseline (Raw BirdNET @0.5)", color="#4C72B0")
    bars_p = ax.bar(x + w/2, p_vals, w, label="Noise-Aware Pipeline", color="#55A868")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Performance: Raw BirdNET vs Noise-Aware Pipeline (N={n_total})",
                 fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)

    for bars in (bars_b, bars_p):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    plt.savefig("results/comparison_graphs/metrics_comparison.png", dpi=150)
    plt.close()

    # Error comparison
    labels_err = ["False Positive Rate\n(Noise → Bird)", "False Negative Rate\n(Bird Missed)"]
    b_errs = [base.get("fpr", 0), base.get("fnr", 0)]
    p_errs = [pipe.get("fpr", 0), pipe.get("fnr", 0)]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    x2 = np.arange(len(labels_err))
    bars_b2 = ax2.bar(x2 - w/2, b_errs, w, label="Baseline (BirdNET)", color="#C44E52")
    bars_p2 = ax2.bar(x2 + w/2, p_errs, w, label="Noise-Aware Pipeline", color="#8172B3")

    ax2.set_ylabel("Rate", fontsize=12)
    ax2.set_title(f"Error Rate Comparison (N={n_total})", fontsize=14, pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_err, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, max(max(b_errs), max(p_errs)) * 1.25 + 0.05)

    for bars in (bars_b2, bars_p2):
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
    fig2.tight_layout()
    plt.savefig("results/comparison_graphs/error_comparison.png", dpi=150)
    plt.close()

    print("  Saved graphs → results/comparison_graphs/")


if __name__ == "__main__":
    main()
