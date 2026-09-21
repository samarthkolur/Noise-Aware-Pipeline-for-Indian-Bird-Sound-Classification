#!/usr/bin/env python3
"""Phase 8: export the trained MLP + AE to ONNX for edge feasibility (design.md §10, DD-013)."""

from __future__ import annotations

import argparse
import logging

import torch

from pipeline.config import load_config
from pipeline.model import BirdAutoencoder, FocalMLP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLP + AE to ONNX.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = cfg.resolve_path(cfg.paths.artifacts_dir)
    embedding_dim = cfg.embedding.embedding_dim

    mlp = FocalMLP(embedding_dim, cfg.mlp)
    mlp.load_state_dict(torch.load(artifacts_dir / "mlp_best.pt", map_location="cpu"))
    mlp.eval()

    ae = BirdAutoencoder(embedding_dim, cfg.autoencoder)
    ae.load_state_dict(torch.load(artifacts_dir / "ae_best.pt", map_location="cpu"))
    ae.eval()

    dummy_input = torch.randn(1, embedding_dim)

    mlp_onnx_path = artifacts_dir / "mlp.onnx"
    torch.onnx.export(
        mlp,
        (dummy_input,),
        str(mlp_onnx_path),
        input_names=["embedding"],
        output_names=["prob"],
        dynamic_axes={"embedding": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported MLP to %s", mlp_onnx_path)

    ae_onnx_path = artifacts_dir / "ae.onnx"
    torch.onnx.export(
        ae,
        (dummy_input,),
        str(ae_onnx_path),
        input_names=["embedding"],
        output_names=["reconstruction"],
        dynamic_axes={"embedding": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported AE to %s", ae_onnx_path)


if __name__ == "__main__":
    main()
