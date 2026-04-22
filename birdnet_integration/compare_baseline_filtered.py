#!/usr/bin/env python3
"""Compute comparison metrics, summary.json, and matplotlib figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from birdnet_integration.integration_config import experiment_fingerprint, load_experiment_config, resolve_under_root
from birdnet_integration.align_segments import align


def _parse_json_list(s: str) -> List[str]:
    return json.loads(s) if isinstance(s, str) else []


def _parse_json_det(s: str) -> List[Dict[str, Any]]:
    return json.loads(s) if isinstance(s, str) else []


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    bird = df[df["pipeline_bucket"] == "bird"]
    noise = df[df["pipeline_bucket"] == "noise"]

    baseline_all = int(df["baseline_detection_count"].sum())
    baseline_on_bird = int(bird["baseline_detection_count"].sum())
    baseline_on_noise = int(noise["baseline_detection_count"].sum())

    filtered_total = int(bird["filtered_detection_count"].sum())

    # Species richness (from per-row species sets)
    sp_b_all: set[str] = set()
    sp_b_bird: set[str] = set()
    sp_f: set[str] = set()
    for _, row in df.iterrows():
        sp_b_all.update(_parse_json_list(row["baseline_species"]))
        if row["pipeline_bucket"] == "bird":
            sp_b_bird.update(_parse_json_list(row["baseline_species"]))
            sp_f.update(_parse_json_list(row["filtered_species"]))

    # Retained recall per bird segment
    recalls: List[float] = []
    for _, row in bird.iterrows():
        b_dets = _parse_json_det(row["baseline_detections_json"])
        f_dets = _parse_json_det(row["filtered_detections_json"])
        b_spec = {d["species_code"] for d in b_dets}
        f_spec = {d["species_code"] for d in f_dets}
        if not b_spec:
            continue
        inter = len(b_spec & f_spec)
        recalls.append(inter / len(b_spec))

    retained_mean = float(np.mean(recalls)) if recalls else None
    retained_std = float(np.std(recalls)) if len(recalls) > 1 else 0.0

    # Confidence pools (detection-level)
    conf_b_bird: List[float] = []
    conf_f: List[float] = []
    for _, row in bird.iterrows():
        for d in _parse_json_det(row["baseline_detections_json"]):
            conf_b_bird.append(float(d["confidence"]))
        for d in _parse_json_det(row["filtered_detections_json"]):
            conf_f.append(float(d["confidence"]))

    def _moments(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}
        a = np.array(xs, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
        }

    # Headline reductions (interpretable)
    n_seg = len(df)
    n_bird = len(bird)
    n_noise = len(noise)

    det_reduction_overall = None
    if baseline_all > 0:
        det_reduction_overall = 1.0 - (filtered_total / baseline_all)

    det_delta_on_bird_audio = None
    if baseline_on_bird > 0:
        det_delta_on_bird_audio = 1.0 - (filtered_total / baseline_on_bird)

    species_reduction_vs_all = None
    if len(sp_b_all) > 0:
        species_reduction_vs_all = 1.0 - (len(sp_f) / len(sp_b_all))

    species_reduction_bird_domain = None
    if len(sp_b_bird) > 0:
        species_reduction_bird_domain = 1.0 - (len(sp_f) / len(sp_b_bird))

    return {
        "segment_counts": {
            "total": n_seg,
            "bird": n_bird,
            "noise": n_noise,
            "uncertain": int((df["pipeline_bucket"] == "uncertain").sum()),
        },
        "detection_counts": {
            "baseline_sum_all_segments": baseline_all,
            "baseline_sum_bird_segments_only": baseline_on_bird,
            "baseline_sum_noise_segments_fp_proxy": baseline_on_noise,
            "filtered_sum_clean_bird_segments": filtered_total,
            "detection_reduction_rate_overall": det_reduction_overall,
            "detection_rate_on_bird_segments_baseline_vs_filtered": det_delta_on_bird_audio,
        },
        "species_richness": {
            "unique_species_baseline_all_segments": len(sp_b_all),
            "unique_species_baseline_bird_segments_only": len(sp_b_bird),
            "unique_species_filtered_clean_birds": len(sp_f),
            "species_reduction_rate_vs_baseline_all": species_reduction_vs_all,
            "species_reduction_rate_bird_domain_only": species_reduction_bird_domain,
            "jaccard_bird_baseline_vs_filtered": (
                len(sp_b_bird & sp_f) / len(sp_b_bird | sp_f) if (sp_b_bird | sp_f) else None
            ),
        },
        "false_positive_proxy": {
            "baseline_detections_on_noise_segments": baseline_on_noise,
            "filtered_detections_on_noise_segments": 0,
            "note": "Filtered arm does not run BirdNET on noise segments; FP proxy compares baseline-on-noise to zero.",
        },
        "confidence": {
            "baseline_on_bird_segments": _moments(conf_b_bird),
            "filtered_on_bird_segments": _moments(conf_f),
        },
        "retained_bird_recall": {
            "mean_species_recall_per_segment": retained_mean,
            "std_species_recall_per_segment": retained_std,
            "segments_with_non_empty_baseline": len(recalls),
        },
    }


def plot_figures(metrics: Dict[str, Any], df: pd.DataFrame, fig_dir: Path) -> List[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # 1) Detection counts
    d = metrics["detection_counts"]
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Baseline\n(all segments)", "Baseline\n(bird seg. only)", "Filtered\n(clean birds)"]
    vals = [
        d["baseline_sum_all_segments"],
        d["baseline_sum_bird_segments_only"],
        d["filtered_sum_clean_bird_segments"],
    ]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Detection count")
    ax.set_title("Detection counts (aligned segments)")
    fig.tight_layout()
    p = fig_dir / "detection_counts.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # 2) Species richness
    s = metrics["species_richness"]
    fig, ax = plt.subplots(figsize=(9, 6))
    labels2 = ["Unique sp.\n(baseline all)", "Unique sp.\n(baseline bird)", "Unique sp.\n(filtered)"]
    vals2 = [
        s["unique_species_baseline_all_segments"],
        s["unique_species_baseline_bird_segments_only"],
        s["unique_species_filtered_clean_birds"],
    ]
    ax.bar(np.arange(3), vals2, color=["#4C72B0", "#8172B3", "#55A868"])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels2, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Species richness")
    fig.tight_layout()
    p2 = fig_dir / "species_richness.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # 3) FP proxy on noise
    fp = metrics["false_positive_proxy"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([0], [fp["baseline_detections_on_noise_segments"]], width=0.5, color="#C44E52", label="Baseline on noise")
    ax.bar([1], [fp["filtered_detections_on_noise_segments"]], width=0.5, color="#55A868", label="Filtered (N/A)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(noise seg.)", "Filtered\n(no run)"])
    ax.set_ylabel("Detections")
    ax.set_title("False-positive proxy (BirdNET on MLP-noise segments)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = fig_dir / "false_positive_proxy.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    # 4) Confidence overlap
    bird = df[df["pipeline_bucket"] == "bird"]
    conf_b: List[float] = []
    conf_f: List[float] = []
    for _, row in bird.iterrows():
        for d in _parse_json_det(row["baseline_detections_json"]):
            conf_b.append(float(d["confidence"]))
        for d in _parse_json_det(row["filtered_detections_json"]):
            conf_f.append(float(d["confidence"]))
    fig, ax = plt.subplots(figsize=(9, 6))
    if conf_b:
        ax.hist(conf_b, bins=30, alpha=0.55, label="Baseline (bird seg.)", density=True, color="#4C72B0")
    if conf_f:
        ax.hist(conf_f, bins=30, alpha=0.55, label="Filtered", density=True, color="#55A868")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distribution (bird segments)")
    if conf_b or conf_f:
        ax.legend()
    fig.tight_layout()
    p4 = fig_dir / "confidence_distribution.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved.append(p4)

    # 5) Retained recall distribution
    recalls: List[float] = []
    for _, row in bird.iterrows():
        b_dets = _parse_json_det(row["baseline_detections_json"])
        f_dets = _parse_json_det(row["filtered_detections_json"])
        b_spec = {d["species_code"] for d in b_dets}
        f_spec = {d["species_code"] for d in f_dets}
        if not b_spec:
            continue
        recalls.append(len(b_spec & f_spec) / len(b_spec))
    fig, ax = plt.subplots(figsize=(9, 6))
    if recalls:
        ax.hist(recalls, bins=20, color="#8172B3", edgecolor="white")
        ax.axvline(
            float(np.mean(recalls)),
            color="k",
            linestyle="--",
            label=f"mean={np.mean(recalls):.3f}",
        )
        ax.legend()
    ax.set_xlabel("Per-segment species recall")
    ax.set_ylabel("Segments")
    ax.set_title("Retained bird recall (baseline vs filtered species sets)")
    fig.tight_layout()
    p5 = fig_dir / "retained_bird_recall.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved.append(p5)

    return saved


def run_compare(exp_path: Path | None) -> Dict[str, Any]:
    exp, pipeline = load_experiment_config(exp_path, None)
    df, align_path = align(exp, pipeline)

    metrics = compute_metrics(df)
    fp = experiment_fingerprint(exp)

    comp = resolve_under_root(exp["paths"]["comparison_dir"])
    fig_dir = comp / "figures"
    figures = plot_figures(metrics, df, fig_dir)

    summary: Dict[str, Any] = {
        "experiment_fingerprint": fp,
        "overlap_tau": exp.get("alignment", {}).get("overlap_tau"),
        "segment_alignment_path": str(align_path),
        "figures": [str(p) for p in figures],
        "metrics": metrics,
    }
    out_json = comp / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs filtered BirdNET")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_config.yaml",
    )
    args = parser.parse_args()
    run_compare(args.experiment)


if __name__ == "__main__":
    main()
