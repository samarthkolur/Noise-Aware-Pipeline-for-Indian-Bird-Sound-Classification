#!/usr/bin/env python3
"""
Research suite: 3-way benchmark, paired statistics, error mining, plots, report text.

Run from repository root:
    python research/run_research_suite.py --config config.yaml

Outputs (default under ./results/):
    benchmark_comparison.json, benchmark_table.csv
    statistical_tests.json
    error_analysis.json, error_samples/...
    plots/pca_plot.png, tsne_plot.png, ae_error_distribution.png
    plots/feature_importance.png, confusion_matrices.png
    plots/roc_curves.png, pr_curves.png
    report_snippets.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root on sys.path (supports `python research/run_research_suite.py`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.classifier import EmbeddingClassifier
from research.alignment import get_test_split_arrays, rows_to_baseline_keys
from research.error_mining import mine_and_copy
from research.metrics_common import metrics_dict
from research.plots_research import (
    plot_ae_histogram,
    plot_confusion_matrix_heatmap,
    plot_feature_importance_mlp,
    plot_pca_tsne,
    plot_roc_pr_curves,
)
from research.predictors import ae_mlp_predictions, baseline_predictions, mlp_predictions
from research.stats_tests import paired_tests
from utils.ae_checkpoint import load_tau_ae_and_meta
from utils.config import load_config


def _device(cfg: dict) -> torch.device:
    s = cfg.get("project", {}).get("device", "auto")
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _write_benchmark_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _report_snippets(
    bench: dict,
    stats: dict,
    n_test: int,
    threshold: float,
    auc_results: dict,
) -> str:
    """Formal academic-style paragraphs (filled from computed JSON)."""
    sysm = bench["systems"]
    b = sysm["birdnet_baseline"]
    m = sysm["noise_aware_mlp_only"]
    a = sysm["noise_aware_mlp_with_ae_gate"]

    p_bm = stats["baseline_vs_mlp"]["paired_ttest"]["pvalue"]
    p_ma = stats["mlp_vs_mlp_ae"]["paired_ttest"]["pvalue"]
    w_bm = stats["baseline_vs_mlp"]["wilcoxon_signed_rank"].get("pvalue")
    w_ma = stats["mlp_vs_mlp_ae"]["wilcoxon_signed_rank"].get("pvalue")
    w_bm_s = f"{w_bm:.4e}" if w_bm is not None else "n/a"
    w_ma_s = f"{w_ma:.4e}" if w_ma is not None else "n/a"

    # AUC summary
    auc_lines = []
    for sys_name, aucs in auc_results.items():
        roc = aucs.get("roc_auc", 0)
        pr = aucs.get("pr_auc", 0)
        auc_lines.append(f"{sys_name}: ROC-AUC {roc:.4f}, PR-AUC {pr:.4f}")
    auc_text = "; ".join(auc_lines) if auc_lines else "AUC computation skipped."

    text = f"""
### Benchmark comparison (held-out test split, N={n_test})

Table \\ref{{tab:benchmark}} summarizes binary classification performance for three systems evaluated on the identical stratified test partition (matching training/validation splits). The BirdNET baseline applies a fixed confidence threshold of {threshold} on exported analyzer scores. The noise-aware MLP operates on BirdNET embeddings with identical thresholding ({threshold}) for binary decisions. The full pipeline applies autoencoder-based out-of-distribution rejection prior to the MLP; binary predictions follow the same {threshold} rule on classifier logits for non-rejected embeddings, while rejected embeddings are assigned the noise class.

Observed metrics — Baseline: accuracy {b['accuracy']:.4f}, F1 {b['f1']:.4f}, FPR {b['fpr']:.4f}, FNR {b['fnr']:.4f}; MLP-only: accuracy {m['accuracy']:.4f}, F1 {m['f1']:.4f}, FPR {m['fpr']:.4f}, FNR {m['fnr']:.4f}; MLP+AE: accuracy {a['accuracy']:.4f}, F1 {a['f1']:.4f}, FPR {a['fpr']:.4f}, FNR {a['fnr']:.4f}.

### Area under curve

{auc_text}

ROC and Precision-Recall curves are provided in Figure \\ref{{fig:roc}} and Figure \\ref{{fig:pr}} respectively.

### Statistical validation

We assessed paired differences in per-segment correctness (indicator equal to one if the predicted label matches the manifest-derived bird/noise label). A paired t-test and a Wilcoxon signed-rank test were applied to compare (i) BirdNET baseline versus the embedding MLP, and (ii) the MLP without versus with the autoencoder gate. For baseline versus MLP, the paired t-test yielded p = {p_bm:.4e} (Wilcoxon p = {w_bm_s}); for MLP versus MLP+AE, the paired t-test yielded p = {p_ma:.4e} (Wilcoxon p = {w_ma_s}). Values below the conventional α = 0.05 threshold are interpreted as evidence of a statistically significant shift in paired accuracy between systems on the same segments.

### Error analysis and qualitative review

Automated mining identifies high-confidence false positives, high-uncertainty false negatives, segments recovered relative to the BirdNET baseline, and segments rejected by the autoencoder gate. Accompanying heuristic descriptors (zero-crossing rate, spectral flatness, low-frequency energy fraction, RMS level) provide coarse tags for insect-like, wind-like, or low-energy segments to guide qualitative discussion. Audio exemplars are copied under `results/error_samples/` for manual audition.

### Visualizations

PCA and t-SNE projections of BirdNET embeddings visualize class structure and highlight autoencoder-rejected points relative to bird and noise labels. The reconstruction-error histogram situates the learned threshold τ_AE within the empirical error distribution. Confusion matrix heatmaps show per-class accuracy for all three systems. The bar chart summarizes an approximate MLP interpretability signal combining gradient magnitude with respect to inputs and first-layer weight magnitudes across embedding dimensions (not a CNN Grad-CAM map, but a standard post-hoc attribution for MLPs on fixed embeddings).
"""
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Research benchmark + statistics + plots")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--baseline-jsonl",
        type=str,
        default=None,
        help="BirdNET baseline JSONL (default: comparison/baseline_normalized.jsonl)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="BirdNET & MLP binary threshold")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = _ROOT
    results_dir = project_root / args.results_dir
    plots_dir = results_dir / "plots"
    err_dir = results_dir / "error_samples"
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_jsonl or "comparison/baseline_normalized.jsonl")
    if not baseline_path.is_file():
        baseline_path = project_root / baseline_path
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline JSONL not found: {baseline_path}. "
            "Generate BirdNET exports or pass --baseline-jsonl."
        )

    device = _device(cfg)
    embs, y_true, rows, test_idx = get_test_split_arrays(cfg)
    keys = rows_to_baseline_keys(rows)

    pred_b, prob_b = baseline_predictions(keys, baseline_path, threshold=args.threshold)
    pred_m, prob_m = mlp_predictions(cfg, embs, device, threshold=args.threshold)
    pred_a, prob_a, recon_err, ood = ae_mlp_predictions(cfg, embs, device, mlp_threshold=args.threshold)

    mb = metrics_dict(y_true, pred_b)
    mm = metrics_dict(y_true, pred_m)
    ma = metrics_dict(y_true, pred_a)

    bench_payload = {
        "eval": "test_split",
        "threshold": args.threshold,
        "n_test": int(len(y_true)),
        "seed": cfg.get("project", {}).get("seed", 42),
        "systems": {
            "birdnet_baseline": {k: mb[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "tp", "tn", "fp", "fn")},
            "noise_aware_mlp_only": {k: mm[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "tp", "tn", "fp", "fn")},
            "noise_aware_mlp_with_ae_gate": {k: ma[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "tp", "tn", "fp", "fn")},
        },
    }

    with open(results_dir / "benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(bench_payload, f, indent=2)

    csv_rows = [
        {"system": "BirdNET baseline (0.5)", **{k: mb[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr")}},
        {"system": "Noise-aware MLP only (0.5)", **{k: mm[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr")}},
        {"system": "Noise-aware MLP + AE gate (0.5)", **{k: ma[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr")}},
    ]
    _write_benchmark_csv(results_dir / "benchmark_table.csv", csv_rows)

    correct_b = (pred_b == y_true).astype(np.float64)
    correct_m = (pred_m == y_true).astype(np.float64)
    correct_a = (pred_a == y_true).astype(np.float64)

    stats_payload = {
        "baseline_vs_mlp": paired_tests(correct_b, correct_m, "BirdNET_baseline", "MLP_only"),
        "mlp_vs_mlp_ae": paired_tests(correct_m, correct_a, "MLP_only", "MLP_plus_AE"),
    }
    with open(results_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    # Load heuristic overrides from config
    heuristic_overrides = cfg.get("research", {}).get("heuristics", None)

    err_report = mine_and_copy(
        y_true,
        pred_b,
        pred_m,
        pred_a,
        prob_b,
        prob_m,
        recon_err,
        ood,
        rows,
        project_root,
        err_dir,
        top_k=args.top_k,
        heuristic_overrides=heuristic_overrides,
    )
    err_full = {
        **err_report,
        "note": "Heuristic tags are indicative only; faint bird calls may overlap insect-like cues in short clips.",
    }
    with open(results_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(err_full, f, indent=2)

    # Path proxy: noise routed folder in source path
    path_proxy = np.array(["/noise/" in (r.get("source_file") or "").replace("\\\\", "/") for r in rows])

    auc_results: dict = {}

    if not args.skip_plots:
        ae_cfg = cfg.get("autoencoder", {})
        chk = Path(ae_cfg.get("checkpoint_path", "./checkpoints/autoencoder.pt"))
        if not chk.is_absolute():
            chk = project_root / chk
        tau, _ = load_tau_ae_and_meta(chk)

        plot_pca_tsne(
            embs,
            y_true,
            ood,
            plots_dir / "pca_plot.png",
            plots_dir / "tsne_plot.png",
            path_noise_proxy=path_proxy,
            max_points_tsne=800,
            random_seed=int(cfg.get("project", {}).get("seed", 42)),
            skip_tsne=args.skip_tsne,
        )

        plot_ae_histogram(recon_err, tau, plots_dir / "ae_error_distribution.png")

        # Confusion matrix heatmaps (new)
        plot_confusion_matrix_heatmap(
            y_true,
            {
                "BirdNET Baseline": pred_b,
                "MLP Only": pred_m,
                "MLP + AE Gate": pred_a,
            },
            plots_dir / "confusion_matrices.png",
        )

        # ROC and PR curves (new)
        auc_results = plot_roc_pr_curves(
            y_true,
            {
                "BirdNET Baseline": prob_b,
                "MLP Only": prob_m,
                "MLP + AE Gate": prob_a,
            },
            plots_dir / "roc_curves.png",
            plots_dir / "pr_curves.png",
        )

        # Add AUC to benchmark payload
        if auc_results:
            bench_payload["auc_scores"] = auc_results
            with open(results_dir / "benchmark_comparison.json", "w", encoding="utf-8") as f:
                json.dump(bench_payload, f, indent=2)

            # Also add to CSV rows
            for row in csv_rows:
                sys_name = row["system"]
                for auc_name, auc_vals in auc_results.items():
                    if auc_name.lower().replace(" ", "_") in sys_name.lower().replace(" ", "_"):
                        row["roc_auc"] = auc_vals.get("roc_auc", "")
                        row["pr_auc"] = auc_vals.get("pr_auc", "")
            _write_benchmark_csv(results_dir / "benchmark_table.csv", csv_rows)

        chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
        if not chkpt_dir.is_absolute():
            chkpt_dir = project_root / chkpt_dir
        model = EmbeddingClassifier(
            input_dim=int(cfg["embedding"]["embedding_dim"]),
            num_classes=1,
            hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
        ).to(device)
        cpt = torch.load(chkpt_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(cpt["model_state_dict"])
        model.eval()
        plot_feature_importance_mlp(model, embs, device, plots_dir / "feature_importance.png")

    snippets = _report_snippets(
        bench_payload, stats_payload, int(len(y_true)), args.threshold, auc_results
    )
    with open(results_dir / "report_snippets.txt", "w", encoding="utf-8") as f:
        f.write(snippets)

    print(f"Wrote benchmarks → {results_dir / 'benchmark_comparison.json'}")
    print(f"Wrote table      → {results_dir / 'benchmark_table.csv'}")
    print(f"Wrote stats      → {results_dir / 'statistical_tests.json'}")
    print(f"Wrote errors     → {results_dir / 'error_analysis.json'}")
    print(f"Wrote report text → {results_dir / 'report_snippets.txt'}")
    if not args.skip_plots:
        print(f"Wrote plots      → {plots_dir}/")


if __name__ == "__main__":
    main()
