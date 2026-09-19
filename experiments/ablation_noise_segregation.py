#!/usr/bin/env python3
"""
Ablation Study: Noise Segregation V2 — Standalone Noise Feature Evaluation

Evaluates how well the 3 hand-crafted noise features (spectral flatness,
ZCR, insect periodicity) discriminate bird vs noise segments **on their own**,
independent of BirdNET embeddings.

This answers the question: "If we only had these acoustic features
(no deep learning), how well could we classify bird vs noise?"

Approach:
    1. Extract the 3 noise features for all labeled segments
    2. Train a separate lightweight classifier (RF) on JUST the 3 features
    3. Run 6 configurations:
       - baseline_3feat: All 3 raw features (3-d input)
       - original:       Weighted features (original weights)
       - equal_weights:  Weighted features (equal weights)
       - no_flatness:    2 features (ZCR + insect periodicity)
       - no_zcr:         2 features (flatness + insect periodicity)
       - no_insect_periodicity: 2 features (flatness + ZCR)
    4. Compare each config's performance to quantify each feature's
       standalone contribution

The key difference from the previous approach: we do NOT include
BirdNET embeddings. This isolates the noise features' contribution.

Usage:
    python experiments/ablation_noise_segregation.py
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.noise_segregation_v2 import (
    compute_spectral_flatness,
    compute_zcr_normalized,
    compute_insect_periodicity,
    ABLATION_CONFIGS,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


FEATURE_NAMES = ["spectral_flatness", "zcr", "insect_periodicity"]


def extract_noise_features_batch(file_paths, cache_path=None):
    """Extract all 3 noise features. Caches to disk."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached noise features from {cache_path}")
        return np.load(cache_path)

    import librosa
    n = len(file_paths)
    features = np.zeros((n, 3), dtype=np.float32)

    print(f"  Extracting noise features for {n} segments...")
    for i, fpath in enumerate(file_paths):
        try:
            y, sr = librosa.load(str(fpath), sr=config.TARGET_SR)
            features[i, 0] = compute_spectral_flatness(y, sr)
            features[i, 1] = compute_zcr_normalized(y, sr)
            features[i, 2] = compute_insect_periodicity(y, sr)
        except Exception:
            features[i] = [0.5, 0.5, 0.0]

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n}...")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, features)
        print(f"  Cached noise features to {cache_path}")

    return features


def get_feature_columns(config_name):
    """Return which feature columns to include for a config."""
    if config_name in ("baseline_3feat", "original", "equal_weights"):
        return [0, 1, 2]  # All 3
    elif config_name == "no_flatness":
        return [1, 2]  # ZCR + insect
    elif config_name == "no_zcr":
        return [0, 2]  # flatness + insect
    elif config_name == "no_insect_periodicity":
        return [0, 1]  # flatness + ZCR
    else:
        return [0, 1, 2]


def get_config_description(config_name):
    """Return human-readable description."""
    descs = {
        "baseline_3feat": "All 3 features (unweighted, raw values)",
        "original": "All 3 features (original weights: 0.5/0.3/0.2)",
        "equal_weights": "All 3 features (equal weights: 1/3 each)",
        "no_flatness": "Without spectral flatness (2 features: ZCR + insect)",
        "no_zcr": "Without ZCR (2 features: flatness + insect)",
        "no_insect_periodicity": "Without insect periodicity (2 features: flatness + ZCR)",
    }
    return descs.get(config_name, config_name)


def build_feature_matrix(noise_features, config_name):
    """Build the feature matrix for a given config."""
    cols = get_feature_columns(config_name)
    X = noise_features[:, cols].copy()

    # Apply weighting for weighted configs
    if config_name == "original":
        weights = [0.5, 0.3, 0.2]
        for j in range(3):
            X[:, j] *= weights[j]
    elif config_name == "equal_weights":
        for j in range(3):
            X[:, j] *= (1/3)

    return X


def compute_metrics(y_true, y_pred):
    """Compute all 4 required metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_single_config(config_name, noise_feat_train, y_train, noise_feat_val, y_val, verbose=True):
    """Run a single ablation configuration."""
    desc = get_config_description(config_name)
    cols = get_feature_columns(config_name)
    active_features = [FEATURE_NAMES[c] for c in cols]

    if verbose:
        print(f"\n{'═' * 65}")
        print(f"  Configuration: {config_name}")
        print(f"  {desc}")
        print(f"  Active features: {active_features}")
        print(f"{'═' * 65}")

    start = time.time()

    X_train = build_feature_matrix(noise_feat_train, config_name)
    X_val = build_feature_matrix(noise_feat_val, config_name)

    if verbose:
        print(f"  Input shape: train={X_train.shape}, val={X_val.shape}")
        for j, feat in enumerate(active_features):
            col_train = X_train[:, j]
            print(f"    {feat}: mean={col_train.mean():.4f}, std={col_train.std():.4f}, "
                  f"range=[{col_train.min():.4f}, {col_train.max():.4f}]")

    # Train RF with multiple seeds for stability
    all_metrics = []
    seeds = [42, 123, 456, 789, 1024]

    for seed in seeds:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        m = compute_metrics(y_val, y_pred)
        all_metrics.append(m)

    # Average across seeds
    avg_metrics = {}
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = round(float(np.mean(values)), 4)
        avg_metrics[f"{key}_std"] = round(float(np.std(values)), 4)

    # Also get feature importances from the first RF
    clf_main = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf_main.fit(X_train, y_train)
    y_pred_main = clf_main.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_main).tolist()

    feat_importances = {}
    for j, feat in enumerate(active_features):
        feat_importances[feat] = round(float(clf_main.feature_importances_[j]), 6)

    # Cross-validation on train set
    cv_scores = cross_val_score(clf_main, X_train, y_train, cv=5, scoring="f1")

    elapsed = time.time() - start

    result = {
        "config_name": config_name,
        "description": desc,
        "active_features": active_features,
        "n_features": len(active_features),
        **avg_metrics,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_importances": feat_importances,
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "n_seeds": len(seeds),
        "elapsed_seconds": round(elapsed, 2),
    }

    if verbose:
        print(f"\n  Results for '{config_name}' (avg over {len(seeds)} seeds):")
        print(f"    Accuracy:  {avg_metrics['accuracy']:.4f} (±{avg_metrics['accuracy_std']:.4f})")
        print(f"    Precision: {avg_metrics['precision']:.4f} (±{avg_metrics['precision_std']:.4f})")
        print(f"    Recall:    {avg_metrics['recall']:.4f} (±{avg_metrics['recall_std']:.4f})")
        print(f"    F1 Score:  {avg_metrics['f1_score']:.4f} (±{avg_metrics['f1_score_std']:.4f})")
        print(f"    CV F1:     {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"    Feature importances: {feat_importances}")
        print(f"    Time: {elapsed:.1f}s")

    return result


def compute_deltas(results, baseline_key="original"):
    """Compute deltas from baseline."""
    baseline = results[baseline_key]
    deltas = {}
    for name, r in results.items():
        deltas[name] = {
            k: round(r[k] - baseline[k], 6)
            for k in ["accuracy", "precision", "recall", "f1_score"]
        }
    return deltas


def rank_features(deltas):
    """Rank features by F1 impact."""
    mapping = {
        "spectral_flatness": "no_flatness",
        "zcr": "no_zcr",
        "insect_periodicity": "no_insect_periodicity",
    }
    impacts = []
    for feat, cfg in mapping.items():
        if cfg in deltas:
            impacts.append((feat, -deltas[cfg]["f1_score"]))
    impacts.sort(key=lambda x: x[1], reverse=True)
    return impacts


def format_table(results, deltas):
    """Format ASCII results table."""
    lines = []
    lines.append("=" * 105)
    lines.append("  NOISE SEGREGATION V2 — ABLATION STUDY RESULTS")
    lines.append("  Method: Standalone RF classifier on noise features only (no BirdNET embeddings)")
    lines.append("  Metrics averaged over 5 random seeds for stability")
    lines.append("=" * 105)
    lines.append("")

    order = ["baseline_3feat", "original", "equal_weights",
             "no_flatness", "no_zcr", "no_insect_periodicity"]

    hdr = f"{'Configuration':<32s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1 Score':>10s} {'#Feat':>6s}"
    lines.append("  " + hdr)
    lines.append("  " + "─" * len(hdr))

    for cfg in order:
        if cfg in results:
            r = results[cfg]
            lines.append(
                f"  {cfg:<32s} "
                f"{r['accuracy']:>10.4f} "
                f"{r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} "
                f"{r['f1_score']:>10.4f} "
                f"{r['n_features']:>6d}"
            )

    lines.append("")
    lines.append("  Deltas from 'original':")
    lines.append("  " + "─" * len(hdr))

    for cfg in order:
        if cfg in deltas and cfg != "original":
            d = deltas[cfg]
            lines.append(
                f"  {cfg:<32s} "
                f"{d['accuracy']:>+10.4f} "
                f"{d['precision']:>+10.4f} "
                f"{d['recall']:>+10.4f} "
                f"{d['f1_score']:>+10.4f}"
            )

    lines.append("")
    lines.append("=" * 105)
    return "\n".join(lines)


def plot_results(results, output_path):
    """Generate grouped bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["baseline_3feat", "original", "equal_weights",
             "no_flatness", "no_zcr", "no_insect_periodicity"]
    configs = [c for c in order if c in results]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels_map = {"accuracy": "Accuracy", "precision": "Precision",
                  "recall": "Recall", "f1_score": "F1 Score"}

    x = np.arange(len(configs))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (m, color) in enumerate(zip(metrics, colors)):
        vals = [results[c][m] for c in configs]
        stds = [results[c].get(f"{m}_std", 0) for c in configs]
        bars = ax.bar(x + offsets[i]*width, vals, width, yerr=stds,
                      label=labels_map[m], color=color, alpha=0.85,
                      capsize=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    xlabels = {
        "baseline_3feat": "Baseline\n(3 raw feat.)",
        "original": "Original\nWeights",
        "equal_weights": "Equal\nWeights",
        "no_flatness": "Without\nFlatness",
        "no_zcr": "Without\nZCR",
        "no_insect_periodicity": "Without\nInsect Per.",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([xlabels.get(c, c) for c in configs], fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Noise Segregation V2 — Feature Ablation Study\n"
        "Standalone RF on noise features only (no BirdNET embeddings) · Avg over 5 seeds",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved to {output_path}")


def run_ablation_study(verbose=True):
    """Execute the complete ablation study."""
    print("\n" + "#" * 70)
    print("  NOISE SEGREGATION V2 — ABLATION STUDY")
    print("  Standalone noise feature evaluation (no BirdNET embeddings)")
    print("#" * 70)

    total_start = time.time()

    # ─── Load Data ───────────────────────────────────────────────────────
    print("\n  Loading data...")
    y_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))
    all_paths = np.load(os.path.join(config.EMBEDDINGS_DIR, "paths.npy"), allow_pickle=True)
    idx_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "idx_train.npy"))
    idx_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "idx_val.npy"))
    binary_labels = np.load(os.path.join(config.EMBEDDINGS_DIR, "binary_labels.npy"))

    print(f"  Train: {len(y_train)} ({(y_train==1).sum()} bird, {(y_train==0).sum()} noise)")
    print(f"  Val:   {len(y_val)} ({(y_val==1).sum()} bird, {(y_val==0).sum()} noise)")

    # ─── Extract Noise Features ──────────────────────────────────────────
    cache_path = os.path.join(config.EMBEDDINGS_DIR, "noise_features_all.npy")
    all_feats = extract_noise_features_batch(all_paths, cache_path=cache_path)

    labeled_mask = binary_labels >= 0
    labeled_indices = np.where(labeled_mask)[0]
    noise_feat_train = all_feats[labeled_indices[idx_train]]
    noise_feat_val = all_feats[labeled_indices[idx_val]]

    print(f"\n  Feature distributions (TRAIN):")
    for j, name in enumerate(FEATURE_NAMES):
        col = noise_feat_train[:, j]
        print(f"    {name:25s}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"range=[{col.min():.4f}, {col.max():.4f}]")

    print(f"\n  Feature distributions (VAL):")
    for j, name in enumerate(FEATURE_NAMES):
        col = noise_feat_val[:, j]
        print(f"    {name:25s}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"range=[{col.min():.4f}, {col.max():.4f}]")

    # ─── Run All Configurations ──────────────────────────────────────────
    order = ["baseline_3feat", "original", "equal_weights",
             "no_flatness", "no_zcr", "no_insect_periodicity"]
    results = OrderedDict()

    for cfg_name in order:
        result = run_single_config(
            cfg_name, noise_feat_train, y_train, noise_feat_val, y_val, verbose
        )
        results[cfg_name] = result

    # ─── Compute Deltas & Rankings ───────────────────────────────────────
    deltas = compute_deltas(results, "original")
    impacts = rank_features(deltas)

    print(f"\n{'═' * 65}")
    print("  FEATURE IMPORTANCE RANKING (by F1 drop when removed):")
    print(f"{'═' * 65}")
    for rank, (feat, drop) in enumerate(impacts, 1):
        tag = "MOST" if rank == 1 else ("SECOND" if rank == 2 else "LEAST")
        print(f"    {rank}. {feat:<25s}  F1 drop: {drop:+.4f}  ({tag} important)")

    # ─── Print Table ─────────────────────────────────────────────────────
    table = format_table(results, deltas)
    print(f"\n{table}")

    # ─── Save Outputs ────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    chart_path = os.path.join(config.RESULTS_DIR, "noise_segregation_ablation_chart.png")
    plot_results(results, chart_path)

    table_path = os.path.join(config.RESULTS_DIR, "noise_segregation_ablation_table.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"  Table saved to {table_path}")

    total_elapsed = time.time() - total_start

    report = {
        "study_name": "Noise Segregation V2 — Ablation Study",
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "method": (
            "Standalone RF classifier trained on noise features ONLY "
            "(no BirdNET embeddings). This isolates the contribution of "
            "each hand-crafted feature. Metrics averaged over 5 random seeds."
        ),
        "dataset": {
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_bird_train": int((y_train == 1).sum()),
            "n_noise_train": int((y_train == 0).sum()),
            "n_bird_val": int((y_val == 1).sum()),
            "n_noise_val": int((y_val == 0).sum()),
        },
        "configurations": dict(results),
        "deltas_from_original": deltas,
        "feature_importance_ranking": [
            {"rank": i+1, "feature": f, "f1_drop": round(d, 6)}
            for i, (f, d) in enumerate(impacts)
        ],
        "summary": {
            "most_important_feature": impacts[0][0] if impacts else "N/A",
            "least_important_feature": impacts[-1][0] if impacts else "N/A",
            "best_config": max(results.keys(), key=lambda k: results[k]["f1_score"]),
            "worst_config": min(results.keys(), key=lambda k: results[k]["f1_score"]),
        },
    }

    report_path = os.path.join(config.RESULTS_DIR, "noise_segregation_ablation_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to {report_path}")

    print(f"\n{'#' * 70}")
    print(f"  Ablation study completed in {total_elapsed:.1f}s")
    print(f"{'#' * 70}\n")

    return report


if __name__ == "__main__":
    run_ablation_study(verbose=True)
