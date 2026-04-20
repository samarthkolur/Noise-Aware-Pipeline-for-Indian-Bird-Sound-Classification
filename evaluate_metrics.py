"""
Metrics Computation & Comparison

Reads baseline_predictions.json and processed_predictions.json, computes:

  Stage 1 — Bird Detection (processed only, requires is_noise labels):
    - detection_precision, detection_recall, detection_f1, detection_rate
    - False Positive Rate (noise segments flagged as bird)

  Stage 2 — Species Classification:
    - Accuracy, accuracy_among_detected, genus_accuracy
    - Precision (macro), Recall (macro), F1-score (macro)

Writes:
  results/baseline_metrics.json
  results/processed_metrics.json
  results/comparison.json
  results/plots/confusion_matrix_baseline.png
  results/plots/confusion_matrix_processed.png
  results/plots/metrics_bar_chart.png
  results/plots/detection_metrics_bar.png
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -- Paths ---------------------------------------------------------------------
RESULTS_DIR        = "results"
BASELINE_PREDS     = os.path.join(RESULTS_DIR, "baseline_predictions.json")
PROCESSED_PREDS    = os.path.join(RESULTS_DIR, "processed_predictions.json")
BASELINE_METRICS   = os.path.join(RESULTS_DIR, "baseline_metrics.json")
PROCESSED_METRICS  = os.path.join(RESULTS_DIR, "processed_metrics.json")
COMPARISON_FILE    = os.path.join(RESULTS_DIR, "comparison.json")
PLOTS_DIR          = os.path.join(RESULTS_DIR, "plots")


# Labels that cannot be matched by BirdNET and must be excluded from scoring.
# "mystery mystery" is an IBC53 artefact folder with no corresponding species.
EXCLUDED_LABELS = {"mystery mystery"}


def _get_genus(name):
    """Return the first token (genus) of a scientific name, or None."""
    if not name:
        return None
    return name.split()[0]


def _compute_detection_metrics(records):
    """
    Stage 1 — Binary bird-vs-noise detection evaluation using BirdNET output.
    Requires is_noise labels; only meaningful for processed records.
    BirdNET firing (top_prediction is not None) = predicted bird.
    Returns a dict of detection metrics, or {} if no labelled noise records exist.
    """
    scored = [
        r for r in records
        if r.get("is_noise") is not None
        and r["ground_truth"] not in EXCLUDED_LABELS
    ]
    if not scored:
        return {}

    y_true = ["noise" if r["is_noise"] else "bird" for r in scored]
    y_pred = ["noise" if r["top_prediction"] is None else "bird" for r in scored]

    det_prec = round(precision_score(y_true, y_pred, pos_label="bird", zero_division=0), 4)
    det_rec  = round(recall_score(y_true, y_pred, pos_label="bird", zero_division=0), 4)
    det_f1   = round(f1_score(y_true, y_pred, pos_label="bird", zero_division=0), 4)

    bird_total    = sum(1 for r in scored if not r["is_noise"])
    bird_detected = sum(1 for r in scored if not r["is_noise"] and r["top_prediction"] is not None)
    det_rate = round(bird_detected / bird_total, 4) if bird_total else 0.0

    return {
        "detection_precision": det_prec,
        "detection_recall":    det_rec,
        "detection_f1":        det_f1,
        "detection_rate":      det_rate,
    }


# -- Helpers -------------------------------------------------------------------

def _load_predictions(path):
    if not os.path.isfile(path):
        print(f"[ERROR] Predictions file not found: '{path}'")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_fpr(records):
    """
    Binary FPR: among segments whose ground truth is 'noise',
    what fraction did BirdNET falsely flag as a bird species?
    """
    noise_records = [r for r in records if r.get("is_noise", r["ground_truth"] == "noise")]
    if not noise_records:
        return None, 0, 0
    fp = sum(1 for r in noise_records if r["top_prediction"] is not None)
    tn = len(noise_records) - fp
    return round(fp / len(noise_records), 4), fp, tn


def _build_label_arrays(records):
    """
    Build (y_true, y_pred) arrays for species-level classification.
    Noise segments and EXCLUDED_LABELS are omitted -- FPR handles noise separately.
    Segments with no BirdNET detection get prediction = '__no_detection__'.
    """
    y_true, y_pred = [], []
    skipped = 0
    for r in records:
        if r.get("is_noise", r["ground_truth"] == "noise"):
            continue
        if r["ground_truth"] in EXCLUDED_LABELS:
            skipped += 1
            continue
        y_true.append(r["ground_truth"])
        y_pred.append(r["top_prediction"] if r["top_prediction"] else "__no_detection__")
    if skipped:
        print(f"[WARN] Excluded {skipped} records with unrecognised labels "
              f"({sorted(EXCLUDED_LABELS)}) from species metrics.")
    return y_true, y_pred


def compute_metrics(records, label=""):
    """
    Compute full metrics dict from a predictions list.
    """
    y_true, y_pred = _build_label_arrays(records)
    fpr, fp, tn    = _compute_fpr(records)

    noise_records  = [r for r in records
                      if r.get("is_noise", r["ground_truth"] == "noise")]
    bird_records   = [r for r in records
                      if not r.get("is_noise", r["ground_truth"] == "noise")]

    if not y_true:
        print(f"[WARN] No bird segments found in '{label}' predictions -- "
              "cannot compute species metrics.")
        return {}

    labels = sorted(set(y_true))

    acc  = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred, average="macro",
                                 labels=labels, zero_division=0), 4)
    rec  = round(recall_score(y_true, y_pred, average="macro",
                              labels=labels, zero_division=0), 4)
    f1   = round(f1_score(y_true, y_pred, average="macro",
                          labels=labels, zero_division=0), 4)

    no_det = sum(1 for r in bird_records if r["top_prediction"] is None)

    # Scored records: same filter as _build_label_arrays (exclude noise + excluded labels)
    scored = [
        r for r in bird_records
        if r["ground_truth"] not in EXCLUDED_LABELS
    ]

    # accuracy_among_detected: species accuracy restricted to files where BirdNET fired
    detected = [r for r in scored if r["top_prediction"] is not None]
    acc_det = round(sum(r["correct"] for r in detected) / len(detected), 4) if detected else 0.0

    # genus_accuracy: genus of ground truth matches genus of top prediction (all scored files)
    genus_correct = sum(
        1 for r in scored
        if _get_genus(r["ground_truth"]) == _get_genus(r["top_prediction"])
        and r["top_prediction"] is not None
    )
    genus_acc = round(genus_correct / len(scored), 4) if scored else 0.0

    det_metrics = _compute_detection_metrics(records)

    metrics = {
        "total_records":           len(records),
        "bird_segments":           len(bird_records),
        "noise_segments":          len(noise_records),
        "no_detection_count":      no_det,
        # Stage 1 — detection
        **det_metrics,
        "fpr":                     fpr,
        "noise_false_positives":   fp,
        "noise_true_negatives":    tn,
        # Stage 2 — species classification
        "accuracy":                acc,
        "accuracy_among_detected": acc_det,
        "genus_accuracy":          genus_acc,
        "precision_macro":         prec,
        "recall_macro":            rec,
        "f1_macro":                f1,
    }

    print(f"\n{'-'*56}  {label}")
    if det_metrics:
        print(f"  [Stage 1 - Detection]")
        print(f"  Detection precision     : {det_metrics['detection_precision']:.4f}")
        print(f"  Detection recall        : {det_metrics['detection_recall']:.4f}")
        print(f"  Detection F1            : {det_metrics['detection_f1']:.4f}")
        print(f"  Detection rate          : {det_metrics['detection_rate']:.4f}")
    if fpr is not None:
        print(f"  FPR (noise->bird)       : {fpr:.4f}  ({fp} FP / {fp+tn} noise segs)")
    print(f"  [Stage 2 - Classification]")
    print(f"  Accuracy                : {acc:.4f}")
    print(f"  Accuracy (detected only): {acc_det:.4f}  ({len(detected)} files with detections)")
    print(f"  Genus accuracy          : {genus_acc:.4f}")
    print(f"  Precision (macro)       : {prec:.4f}")
    print(f"  Recall (macro)          : {rec:.4f}")
    print(f"  F1 (macro)              : {f1:.4f}")
    print(f"{'-'*56}\n")

    return metrics


# -- Plots ---------------------------------------------------------------------

def _plot_confusion_matrix(records, label, out_path):
    y_true, y_pred = _build_label_arrays(records)
    if not y_true:
        return

    labels = sorted(set(y_true))
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    fig_size = max(10, len(labels) // 3)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix -- {label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[PLOT] Confusion matrix saved: '{out_path}'")


def _plot_metrics_bar(baseline_m, processed_m, out_path):
    metrics_to_plot = [
        "accuracy", "accuracy_among_detected", "genus_accuracy",
        "precision_macro", "recall_macro", "f1_macro", "fpr",
    ]
    labels = ["Accuracy", "Acc (detected)", "Genus Acc", "Precision", "Recall", "F1", "FPR"]

    baseline_vals  = [baseline_m.get(k, 0)  or 0 for k in metrics_to_plot]
    processed_vals = [processed_m.get(k, 0) or 0 for k in metrics_to_plot]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals,  width, label="Baseline (raw)",      color="#4C72B0")
    bars2 = ax.bar(x + width / 2, processed_vals, width, label="Processed (filtered)", color="#DD8452")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Processed -- BirdNET Metrics")
    ax.legend()

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[PLOT] Bar chart saved: '{out_path}'")


def _plot_detection_bar(baseline_m, processed_m, out_path):
    """Stage 1 detection metrics bar chart (processed only; baseline has no noise ground truth)."""
    det_keys   = ["detection_precision", "detection_recall", "detection_f1", "detection_rate", "fpr"]
    det_labels = ["Det. Precision", "Det. Recall", "Det. F1", "Det. Rate", "FPR"]

    processed_vals = [processed_m.get(k) or 0 for k in det_keys]

    x     = np.arange(len(det_labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, processed_vals, width, color="#DD8452", label="Processed (filtered)")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(det_labels)
    ax.set_ylabel("Score")
    ax.set_title("Stage 1 — Bird Detection Metrics (Processed Pipeline)")
    ax.legend()

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[PLOT] Detection bar chart saved: '{out_path}'")


# -- Main ----------------------------------------------------------------------

def evaluate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("[METRICS] Loading predictions ...")
    baseline_records  = _load_predictions(BASELINE_PREDS)
    processed_records = _load_predictions(PROCESSED_PREDS)

    print("[METRICS] Computing baseline metrics ...")
    baseline_m  = compute_metrics(baseline_records,  label="BASELINE")

    print("[METRICS] Computing processed metrics ...")
    processed_m = compute_metrics(processed_records, label="PROCESSED")

    # -- Write per-run metrics --------------------------------------------------
    with open(BASELINE_METRICS, "w") as f:
        json.dump(baseline_m,  f, indent=2)
    with open(PROCESSED_METRICS, "w") as f:
        json.dump(processed_m, f, indent=2)

    # -- Build comparison ------------------------------------------------------
    metric_keys = [
        # Stage 1
        "detection_precision", "detection_recall", "detection_f1", "detection_rate", "fpr",
        # Stage 2
        "accuracy", "accuracy_among_detected", "genus_accuracy",
        "precision_macro", "recall_macro", "f1_macro",
    ]
    comparison  = {}
    for k in metric_keys:
        b = baseline_m.get(k)
        p = processed_m.get(k)
        delta = round(p - b, 4) if (b is not None and p is not None) else None
        comparison[k] = {
            "baseline":  b,
            "processed": p,
            "delta":     delta,
        }

    with open(COMPARISON_FILE, "w") as f:
        json.dump(comparison, f, indent=2)

    # -- Print comparison table -------------------------------------------------
    print(f"\n{'='*62}")
    print("  COMPARISON  (processed - baseline)")
    print(f"{'='*62}")
    print(f"  {'Metric':<22} {'Baseline':>10} {'Processed':>10} {'Delta':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for k, v in comparison.items():
        sign = "+" if (v['delta'] or 0) > 0 else ""
        delta_str = f"{sign}{v['delta']:.4f}" if v['delta'] is not None else "N/A"
        print(f"  {k:<22} {str(v['baseline']):>10} {str(v['processed']):>10} {delta_str:>10}")
    print(f"{'='*62}\n")

    print(f"[METRICS] baseline_metrics.json  -> '{BASELINE_METRICS}'")
    print(f"[METRICS] processed_metrics.json -> '{PROCESSED_METRICS}'")
    print(f"[METRICS] comparison.json        -> '{COMPARISON_FILE}'")

    # -- Plots ------------------------------------------------------------------
    print("\n[PLOTS] Generating visualisations ...")
    _plot_confusion_matrix(
        baseline_records, "Baseline",
        os.path.join(PLOTS_DIR, "confusion_matrix_baseline.png")
    )
    _plot_confusion_matrix(
        processed_records, "Processed",
        os.path.join(PLOTS_DIR, "confusion_matrix_processed.png")
    )
    _plot_metrics_bar(
        baseline_m, processed_m,
        os.path.join(PLOTS_DIR, "metrics_bar_chart.png")
    )
    _plot_detection_bar(
        baseline_m, processed_m,
        os.path.join(PLOTS_DIR, "detection_metrics_bar.png")
    )


if __name__ == "__main__":
    evaluate_all()
