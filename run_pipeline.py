#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry-point for the Bioacoustic Pipeline.

Usage:
    python run_pipeline.py --config config.yaml               # full pipeline
    python run_pipeline.py --config config.yaml --stage train  # single stage
"""

import argparse
import sys
from pathlib import Path

from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Stage registry ──────────────────────────────────────────

STAGES = ["preprocess", "embed", "train", "infer", "evaluate", "mine"]


def run_preprocess(cfg: dict) -> None:
    """Stage 1: Convert → segment → silence removal → save WAV + metadata."""
    from preprocessing.preprocessing import preprocess_dataset

    logger.info("═══ Stage: Preprocessing ═══")
    metas = preprocess_dataset(cfg)
    logger.info(f"Preprocessing complete ✓  ({len(metas)} segments saved)")


def run_embed(cfg: dict) -> None:
    """Stage 2: Extract embeddings → HDF5 + manifest CSV."""
    from embedding.embedding import extract_embeddings

    logger.info("═══ Stage: Embedding Extraction ═══")
    manifest = extract_embeddings(cfg)
    logger.info(f"Embedding extraction complete ✓  (manifest: {manifest})")


def run_train(cfg: dict) -> None:
    """Stage 3: Train MLP, then mandatory bird-only autoencoder (OOD gate)."""
    from training.trainer import Trainer
    from training.autoencoder_trainer import AutoencoderTrainer

    logger.info("═══ Stage: Training (MLP) ═══")
    trainer = Trainer(cfg)
    trainer.fit()
    logger.info("MLP training complete ✓")

    logger.info("═══ Stage: Autoencoder (OOD gate, bird-only) ═══")
    ae_trainer = AutoencoderTrainer(cfg)
    ae_trainer.fit()
    logger.info("Autoencoder training complete ✓")


def run_infer(cfg: dict) -> None:
    """Stage 4: Run inference (raw audio → segments → embeddings → routing)."""
    from inference.predictor import Predictor

    logger.info("═══ Stage: Inference ═══")
    predictor = Predictor(cfg)
    predictor.run()
    logger.info("Inference complete ✓")


def run_evaluate(cfg: dict) -> None:
    """Stage 5: Evaluate classifier on held-out test set."""
    from evaluate import evaluate

    logger.info("═══ Stage: Evaluation ═══")
    evaluate(cfg)
    logger.info("Evaluation complete ✓")


def run_mine(cfg: dict) -> None:
    """Stage 6: Hard-negative & false-negative mining."""
    from mining.mining import Miner

    logger.info("═══ Stage: Mining ═══")
    
    miner = Miner(cfg)

    # Mine false positives (noise predicted as bird)
    source_dir = Path(cfg["data"]["raw_dir"])
    fp_dir = Path(cfg["data"]["output_dir"]) / "false_positives"
    logger.info(f"Mining false positives from {source_dir} -> {fp_dir}")
    miner.mine_false_positives(source_dir, fp_dir)

    # Mine false negatives (bird classified as noise)
    fn_dir = Path(cfg["data"]["output_dir"]) / "false_negatives"
    logger.info(f"Mining false negatives from outputs/noise/ -> {fn_dir}")
    miner.mine_false_negatives(export_dir=fn_dir)

    # Report uncertain samples
    miner.mine_uncertain()
    
    logger.info("Mining complete ✓")


STAGE_RUNNERS = {
    "preprocess": run_preprocess,
    "embed": run_embed,
    "train": run_train,
    "infer": run_infer,
    "evaluate": run_evaluate,
    "mine": run_mine,
}

# ── CLI ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bioacoustic Pipeline — end-to-end runner",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        default=None,
        help="Run a single stage instead of the full pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    if args.stage:
        STAGE_RUNNERS[args.stage](cfg)
    else:
        for stage in STAGES:
            STAGE_RUNNERS[stage](cfg)

    logger.info("Pipeline finished 🎉")


if __name__ == "__main__":
    main()
