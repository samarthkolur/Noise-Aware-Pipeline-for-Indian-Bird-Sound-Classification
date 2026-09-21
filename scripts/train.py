#!/usr/bin/env python3
"""Phase 5: train the Focal-loss MLP classifier and the bird-only autoencoder,
derive tau_AE from the validation split, and save artifacts (design.md §10)."""

from __future__ import annotations

import argparse
import json
import logging

import torch

from pipeline.cache import EmbeddingCache
from pipeline.config import load_config
from pipeline.dataset import EmbeddingDataset, make_weighted_sampler
from pipeline.train_loop import compute_tau_ae, train_autoencoder, train_focal_mlp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FocalMLP + BirdAutoencoder.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.random_seed)

    cache = EmbeddingCache(cfg.resolve_path(cfg.paths.embeddings_cache_path))
    manifest_path = cfg.resolve_path(cfg.paths.manifest_path)

    train_ds = EmbeddingDataset(manifest_path, cache, split="train")
    val_ds = EmbeddingDataset(manifest_path, cache, split="val")
    logger.info("Train: %d segments | Val: %d segments", len(train_ds), len(val_ds))

    train_x, train_y = train_ds.load_all()
    val_x, val_y = val_ds.load_all()
    sampler = make_weighted_sampler(train_ds.labels)

    logger.info("=== Training FocalMLP ===")
    mlp, mlp_history = train_focal_mlp(
        train_x, train_y, val_x, val_y, cfg.mlp, cfg.embedding.embedding_dim, sampler=sampler
    )

    logger.info("=== Training BirdAutoencoder (bird-class embeddings only) ===")
    train_bird_x = train_x[train_ds.bird_mask()]
    val_bird_x = val_x[val_ds.bird_mask()]
    ae, ae_history = train_autoencoder(
        train_bird_x,
        val_bird_x,
        cfg.autoencoder,
        cfg.embedding.embedding_dim,
        max_epochs=cfg.mlp.max_epochs,
    )

    tau_ae = compute_tau_ae(ae, val_bird_x, cfg.autoencoder.ae_percentile)
    logger.info("tau_AE (P%.0f val bird MSE) = %.6f", cfg.autoencoder.ae_percentile, tau_ae)

    artifacts_dir = cfg.resolve_path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.save(mlp.state_dict(), artifacts_dir / "mlp_best.pt")
    torch.save(ae.state_dict(), artifacts_dir / "ae_best.pt")

    config_json = {
        "random_seed": cfg.random_seed,
        "embedding_dim": cfg.embedding.embedding_dim,
        "tau_ae": tau_ae,
        "ae_percentile": cfg.autoencoder.ae_percentile,
        "mlp_best_epoch": mlp_history.best_epoch,
        "mlp_best_val_f1": max(mlp_history.val_metric) if mlp_history.val_metric else None,
        "ae_best_epoch": ae_history.best_epoch,
        "ae_best_val_mse": min(ae_history.val_metric) if ae_history.val_metric else None,
        "mlp_hyperparameters": cfg.mlp.model_dump(),
        "autoencoder_hyperparameters": cfg.autoencoder.model_dump(),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_train_bird": int(train_ds.bird_mask().sum()),
    }
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    logger.info("Saved mlp_best.pt, ae_best.pt, config.json to %s", artifacts_dir)


if __name__ == "__main__":
    main()
