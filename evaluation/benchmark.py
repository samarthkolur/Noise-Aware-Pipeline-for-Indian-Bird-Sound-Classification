"""
Four-Model Benchmark Comparison

Evaluates all four models on the val split using a unified metric function.

Models
------
1. BirdNET Baseline      — baseline_predictions.json
   Run on full raw IBC53 files (not 3-s segments); no noise ground-truth.
   Scope: all 1368 original recordings (bird species only; FPR = N/A).

2. Noise-Aware BirdNET   — processed_predictions.json
   Run on 3-s preprocessed segments; filtered to val split (2928 records).

3. MLP only              — mlp_predictions.json   (2928 val records)
4. MLP + AE gate         — mlp_ae_predictions.json (2928 val records)

Metrics (binary detection — bird vs noise)
------------------------------------------
  accuracy   = (TP + TN) / N
  precision  = TP / (TP + FP)
  recall     = TP / (TP + FN)     [= bird detection rate]
  f1         = harmonic mean of precision and recall
  fpr        = FP / (FP + TN)     [noise-as-bird rate; N/A for baseline]
  fnr        = FN / (FN + TP)     [missed-bird rate]

Outputs
-------
results/benchmark_table.csv
results/benchmark_comparison.json
results/plots/benchmark_bar.png

Usage
-----
    python evaluation/benchmark.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_metrics import EXCLUDED_LABELS

RESULTS   = Path("results")
PLOTS_DIR = RESULTS / "plots"
VAL_CSV   = Path("splits/val.csv")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list:
    if not path.exists():
        print(f"[BENCH] File not found, skipping: {path}")
        return []
    with open(path) as f:
        return json.load(f)


def _filter_to_val(records: list, val_norm: set) -> list:
    """Keep only records whose normalised file path is in val_norm."""
    return [r for r in records
            if Path(r["file"]).as_posix().lower() in val_norm]


# ---------------------------------------------------------------------------
# Unified binary-detection metric computation
# ---------------------------------------------------------------------------
def _compute_metrics(records: list, name: str) -> dict:
    """
    Compute binary detection metrics from a predictions list.

    Detection rule: top_prediction is not None  =>  predicted bird.
    Ground truth  : is_noise field (if present) else ground_truth == "noise".

    Works for records with or without noise ground-truth labels.
    """
    # Exclude mystery-mystery artefact labels
    records = [r for r in records if r["ground_truth"] not in EXCLUDED_LABELS]

    is_noise_gt = [
        r.get("is_noise") if r.get("is_noise") is not None
        else (r["ground_truth"].lower() == "noise")
        for r in records
    ]
    # sentinel "__bird__" means "MLP predicted bird, BirdNET had no species"
    pred_bird = [r["top_prediction"] is not None for r in records]

    TP = sum(1 for g, p in zip(is_noise_gt, pred_bird) if not g and p)
    TN = sum(1 for g, p in zip(is_noise_gt, pred_bird) if g and not p)
    FP = sum(1 for g, p in zip(is_noise_gt, pred_bird) if g and p)
    FN = sum(1 for g, p in zip(is_noise_gt, pred_bird) if not g and not p)

    n_bird  = TP + FN
    n_noise = FP + TN
    N       = len(records)

    accuracy  = round((TP + TN) / N, 4)          if N       else 0.0
    precision = round(TP / (TP + FP), 4)          if TP + FP else 0.0
    recall    = round(TP / (TP + FN), 4)          if TP + FN else 0.0
    f1        = round(2 * TP / (2 * TP + FP + FN), 4) if 2 * TP + FP + FN else 0.0
    fpr       = round(FP / n_noise, 4)            if n_noise else None
    fnr       = round(FN / n_bird,  4)            if n_bird  else None

    row = {
        "name":      name,
        "n_records": N,
        "n_bird":    n_bird,
        "n_noise":   n_noise,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "fpr":       fpr,
        "fnr":       fnr,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }

    fpr_str = f"{fpr:.4f}" if fpr is not None else "N/A"
    fnr_str = f"{fnr:.4f}" if fnr is not None else "N/A"
    print(f"\n  [{name}]")
    print(f"    Records  : {N}  (bird={n_bird}, noise={n_noise})")
    print(f"    Accuracy : {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
    print(f"    FPR      : {fpr_str}          FNR   : {fnr_str}")

    return row


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
def _plot_benchmark(rows: list) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    names   = [r["name"] for r in rows]
    metrics = ["precision", "recall", "f1", "fpr", "fnr"]
    labels  = ["Precision", "Recall", "F1", "FPR", "FNR"]
    colors  = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]

    x     = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = []
        for r in rows:
            v = r.get(metric)
            vals.append(float(v) if v is not None else 0.0)
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Four-Model Benchmark — Bird/Noise Detection (Validation Set)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "benchmark_bar.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"\n[BENCH] Bar chart saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_benchmark() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build val-split normalised path set (used to filter processed predictions)
    if not VAL_CSV.exists():
        print(f"[BENCH] Val CSV not found: {VAL_CSV}")
        sys.exit(1)
    val_df   = pd.read_csv(VAL_CSV)
    val_norm = {Path(p).as_posix().lower() for p in val_df["file_path"]}

    print("\n[BENCH] Computing metrics for each model …")
    print("=" * 62)

    results = []

    # 1. BirdNET Baseline — raw IBC53 files, no val filtering, no noise GT
    recs = _load_json(RESULTS / "baseline_predictions.json")
    if recs:
        results.append(_compute_metrics(recs, "BirdNET Baseline"))

    # 2. Noise-Aware BirdNET — filtered to val split
    recs = _load_json(RESULTS / "processed_predictions.json")
    if recs:
        val_recs = _filter_to_val(recs, val_norm)
        print(f"[BENCH] Noise-Aware BirdNET: {len(val_recs)} val records "
              f"(from {len(recs)} total)")
        results.append(_compute_metrics(val_recs, "Noise-Aware BirdNET"))

    # 3. MLP only — already val split
    recs = _load_json(RESULTS / "mlp_predictions.json")
    if recs:
        results.append(_compute_metrics(recs, "MLP only"))

    # 4. MLP + AE gate — already val split
    recs = _load_json(RESULTS / "mlp_ae_predictions.json")
    if recs:
        results.append(_compute_metrics(recs, "MLP + AE"))

    print("=" * 62)

    if not results:
        print("[BENCH] No results to save.")
        return

    # CSV — drop raw TP/TN/FP/FN columns for readability
    display_cols = ["name", "n_records", "n_bird", "n_noise",
                    "accuracy", "precision", "recall", "f1", "fpr", "fnr"]
    df = pd.DataFrame(results)[display_cols]
    csv_path = RESULTS / "benchmark_table.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"[BENCH] Table saved : {csv_path}")
    print("\n" + df.to_string(index=False))

    # JSON
    json_path = RESULTS / "benchmark_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[BENCH] JSON saved  : {json_path}")

    _plot_benchmark(results)


if __name__ == "__main__":
    run_benchmark()
