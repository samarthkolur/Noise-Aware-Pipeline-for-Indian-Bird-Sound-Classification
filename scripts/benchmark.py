#!/usr/bin/env python3
"""Phase 7: three-way benchmark (BirdNET baseline, MLP Only, MLP+AE Gate) on the
held-out test split, with statistical validation (design.md §9, §10, DD-012).

Read-only over data/manifest.csv + artifacts/: never triggers training or
re-evaluation of any other split. Running it twice produces identical results.

IMPORTANT SCOPE NOTE: this environment does not have access to the official
iBC53 corpus (design.md §18/§25). The bird class here is a real 53-species
Indian bird corpus; the noise class is synthetic (opt-in, DD-010). The numbers
this script produces are NOT a reproduction of the paper's Table I and must
not be reported as such — see design.md Technical Debt.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from pipeline.audio import load_audio
from pipeline.cache import EmbeddingCache
from pipeline.config import load_config
from pipeline.dataset import EmbeddingDataset
from pipeline.embedding import extract_max_confidence_batch
from pipeline.evaluate import compute_metrics, per_segment_correctness
from pipeline.model import BirdAutoencoder, FocalMLP
from pipeline.plots import (
    plot_ae_mse_histogram,
    plot_confusion_matrix,
    plot_embedding_projection,
    plot_pr_curve,
    plot_roc_curve,
)
from pipeline.router import route_batch
from pipeline.stats import paired_significance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BIRDNET_BASELINE_THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way benchmark on the test split.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.random_seed)

    cache = EmbeddingCache(cfg.resolve_path(cfg.paths.embeddings_cache_path))
    manifest_path = cfg.resolve_path(cfg.paths.manifest_path)
    artifacts_dir = cfg.resolve_path(cfg.paths.artifacts_dir)
    outputs_dir = cfg.resolve_path(cfg.paths.outputs_dir)
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "config.json") as f:
        saved = json.load(f)
    tau_ae = saved["tau_ae"]

    mlp = FocalMLP(cfg.embedding.embedding_dim, cfg.mlp)
    mlp.load_state_dict(torch.load(artifacts_dir / "mlp_best.pt", map_location="cpu"))
    mlp.eval()

    ae = BirdAutoencoder(cfg.embedding.embedding_dim, cfg.autoencoder)
    ae.load_state_dict(torch.load(artifacts_dir / "ae_best.pt", map_location="cpu"))
    ae.eval()

    test_ds = EmbeddingDataset(manifest_path, cache, split="test")
    test_x, test_y = test_ds.load_all()  # already L2-normalised by EmbeddingDataset
    logger.info(
        "Test split: %d segments (%d bird, %d noise)",
        len(test_ds),
        int(test_ds.bird_mask().sum()),
        int((~test_ds.bird_mask()).sum()),
    )

    # --- MLP Only ---
    with torch.no_grad():
        mlp_probs = mlp(test_x).numpy()
    mlp_preds = (mlp_probs >= 0.5).astype(int)
    mlp_metrics = compute_metrics(test_y.numpy(), mlp_preds, mlp_probs)

    # --- MLP + AE Gate ---
    with torch.no_grad():
        ae_mse = ae.reconstruction_error(test_x).numpy()
    ood_rejected = ae_mse > tau_ae
    gated_preds = np.where(ood_rejected, 0, mlp_preds)
    # OOD-rejected segments are forced to the noise-class probability floor for AUC purposes.
    gated_probs = np.where(ood_rejected, 0.0, mlp_probs)
    gated_metrics = compute_metrics(test_y.numpy(), gated_preds, gated_probs)

    # --- BirdNET Baseline ---
    logger.info("Running real BirdNET species-confidence inference for the baseline...")
    test_paths = [cfg.resolve_path(r["path"]) for r in test_ds.rows]
    test_audios = [load_audio(str(p), cfg.audio.target_sr)[0] for p in test_paths]
    baseline_conf = extract_max_confidence_batch(test_audios, cfg.audio.target_sr, cfg.embedding)
    baseline_preds = (baseline_conf >= BIRDNET_BASELINE_THRESHOLD).astype(int)
    baseline_metrics = compute_metrics(test_y.numpy(), baseline_preds, baseline_conf)

    # --- Statistical validation ---
    baseline_correct = per_segment_correctness(test_y.numpy(), baseline_preds)
    mlp_correct = per_segment_correctness(test_y.numpy(), mlp_preds)
    gated_correct = per_segment_correctness(test_y.numpy(), gated_preds)

    stats_baseline_to_mlp = paired_significance(baseline_correct, mlp_correct)
    stats_mlp_to_gated = paired_significance(mlp_correct, gated_correct)

    # --- Three-band router (on MLP+AE gate outputs) ---
    routing = route_batch(list(mlp_probs), list(ood_rejected), cfg.router)
    band_counts: dict[str, int] = {}
    for r in routing:
        band_counts[r.final_band] = band_counts.get(r.final_band, 0) + 1
    uncertain_fraction = band_counts.get("uncertain", 0) / len(routing) if routing else 0.0

    report = {
        "scope_note": (
            "Bird class = real 53-species Indian corpus (data/segmented). Noise class = "
            "SYNTHETIC (opt-in, DD-010) because no real iBC53 noise corpus is available in "
            "this environment. These numbers are NOT a paper Table I reproduction."
        ),
        "n_test": len(test_ds),
        "birdnet_baseline": baseline_metrics,
        "mlp_only": mlp_metrics,
        "mlp_ae_gate": gated_metrics,
        "tau_ae": tau_ae,
        "significance_baseline_to_mlp": stats_baseline_to_mlp,
        "significance_mlp_to_gated": stats_mlp_to_gated,
        "three_band_router": {"band_counts": band_counts, "uncertain_fraction": uncertain_fraction},
    }

    with open(outputs_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", outputs_dir / "evaluation_report.json")

    # --- Plots ---
    plot_roc_curve(test_y.numpy(), mlp_probs, plots_dir / "mlp_roc.png", "MLP Only ROC")
    plot_pr_curve(test_y.numpy(), mlp_probs, plots_dir / "mlp_pr.png", "MLP Only PR")
    plot_confusion_matrix(
        test_y.numpy(),
        mlp_preds,
        plots_dir / "mlp_confusion_matrix.png",
        "MLP Only Confusion Matrix",
    )
    plot_confusion_matrix(
        test_y.numpy(),
        gated_preds,
        plots_dir / "mlp_ae_gate_confusion_matrix.png",
        "MLP+AE Gate Confusion Matrix",
    )
    plot_confusion_matrix(
        test_y.numpy(),
        baseline_preds,
        plots_dir / "birdnet_baseline_confusion_matrix.png",
        "BirdNET Baseline Confusion Matrix",
    )
    plot_ae_mse_histogram(ae_mse, tau_ae, plots_dir / "ae_mse_histogram.png")
    if len(test_x) > 2:
        plot_embedding_projection(
            test_x.numpy(), test_y.numpy(), plots_dir / "embedding_pca.png", method="pca"
        )
    logger.info("Wrote plots to %s", plots_dir)

    logger.info("=== Summary ===")
    for name, m in [
        ("BirdNET Baseline", baseline_metrics),
        ("MLP Only", mlp_metrics),
        ("MLP + AE Gate", gated_metrics),
    ]:
        logger.info(
            "%-18s acc=%.4f f1=%.4f fnr=%.4f roc_auc=%.4f",
            name,
            m["accuracy"],
            m["f1"],
            m["fnr"],
            m.get("roc_auc", float("nan")),
        )


if __name__ == "__main__":
    main()
