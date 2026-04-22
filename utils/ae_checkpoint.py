"""Load trained autoencoder weights and τ_AE for mandatory OOD gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from models.autoencoder import EmbeddingAutoencoder


def load_tau_ae_and_meta(checkpoint_path: Path) -> tuple[float, dict[str, Any]]:
    """Load τ_AE from ae_threshold.json (preferred) or legacy autoencoder_meta.json."""
    threshold_path = checkpoint_path.parent / "ae_threshold.json"
    legacy_meta = checkpoint_path.with_name("autoencoder_meta.json")
    if threshold_path.exists():
        with open(threshold_path, encoding="utf-8") as f:
            data = json.load(f)
        tau = float(data["tau_ae"])
        return tau, data
    if legacy_meta.exists():
        with open(legacy_meta, encoding="utf-8") as f:
            data = json.load(f)
        tau = float(data.get("tau_ae", data["recon_threshold"]))
        return tau, data
    raise FileNotFoundError(
        f"Missing OOD gate threshold artifacts: expected {threshold_path} or {legacy_meta} "
        f"next to the autoencoder checkpoint {checkpoint_path}. Re-run the training stage."
    )


def build_autoencoder_from_cfg(cfg: dict, meta: dict[str, Any] | None = None) -> EmbeddingAutoencoder:
    emb_dim = int(cfg["embedding"]["embedding_dim"])
    ae_cfg = cfg.get("autoencoder", {})
    latent: int | None = None
    if meta and "latent_dim" in meta:
        latent = int(meta["latent_dim"])
    elif "latent_dim" in ae_cfg:
        latent = int(ae_cfg["latent_dim"])
    if latent is not None:
        return EmbeddingAutoencoder(input_dim=emb_dim, latent_dim=latent)
    return EmbeddingAutoencoder(input_dim=emb_dim)


def load_autoencoder_state(
    cfg: dict,
    device: torch.device,
) -> tuple[EmbeddingAutoencoder, float, dict[str, Any]]:
    """Load the trained autoencoder and τ_AE. Raises FileNotFoundError if files are missing."""
    ae_cfg = cfg.get("autoencoder", {})
    chkpt_path = Path(ae_cfg.get("checkpoint_path", "./checkpoints/autoencoder.pt"))
    if not chkpt_path.exists():
        raise FileNotFoundError(
            f"Autoencoder checkpoint required for OOD gating but not found at {chkpt_path}. "
            "Run the training stage to train the autoencoder on bird-only embeddings."
        )
    tau, meta = load_tau_ae_and_meta(chkpt_path)
    model = build_autoencoder_from_cfg(cfg, meta).to(device)
    chk = torch.load(chkpt_path, map_location=device, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    return model, tau, meta
