"""Compute per-sample predictions for BirdNET baseline, MLP, and AE+MLP."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from compute_baseline_metrics import load_baseline_predictions
from inference.prediction_api import predict_embeddings_ae_gate, predict_embeddings_mlp
from models.classifier import EmbeddingClassifier
from utils.ae_checkpoint import load_autoencoder_state


def baseline_predictions(
    keys: List[tuple],
    baseline_jsonl: Path,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """BirdNET baseline: pred bird iff max confidence >= threshold (missing key → 0)."""
    bl = load_baseline_predictions(str(baseline_jsonl))
    probs = np.array([float(bl.get(k, 0.0)) for k in keys], dtype=np.float64)
    pred = (probs >= threshold).astype(np.int64)
    return pred, probs


def mlp_predictions(
    cfg: dict,
    embs: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """MLP only: all embeddings forwarded; pred bird iff sigmoid(logit) >= threshold."""
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    model = EmbeddingClassifier(
        input_dim=int(cfg["embedding"]["embedding_dim"]),
        num_classes=1,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)
    chk = torch.load(chkpt_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    probs_out = np.zeros(len(embs), dtype=np.float64)
    bs = int(cfg.get("evaluation", {}).get("batch_size", 64))
    with torch.no_grad():
        for start in range(0, len(embs), bs):
            x = torch.from_numpy(embs[start : start + bs]).to(device, non_blocking=True)
            probs_out[start : start + bs] = (
                predict_embeddings_mlp(model, x, binary=True).cpu().numpy()
            )

    pred = (probs_out >= threshold).astype(np.int64)
    return pred, probs_out


def ae_mlp_predictions(
    cfg: dict,
    embs: np.ndarray,
    device: torch.device,
    mlp_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """AE OOD gate + MLP: OOD → noise (0); else pred bird iff prob >= mlp_threshold.

    Returns:
        pred, mlp_probs (full array; meaningless where OOD but filled), recon_errors, ood_mask
    """
    ae_model, tau, _meta = load_autoencoder_state(cfg, device)
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    model = EmbeddingClassifier(
        input_dim=int(cfg["embedding"]["embedding_dim"]),
        num_classes=1,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)
    chk = torch.load(chkpt_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    n = len(embs)
    recon_err = np.zeros(n, dtype=np.float64)
    mlp_probs = np.zeros(n, dtype=np.float64)
    pred = np.zeros(n, dtype=np.int64)
    ood = np.zeros(n, dtype=bool)

    bs = int(cfg.get("evaluation", {}).get("batch_size", 64))
    with torch.no_grad():
        for start in range(0, n, bs):
            x = torch.from_numpy(embs[start : start + bs]).to(device, non_blocking=True)
            is_ood, err = predict_embeddings_ae_gate(ae_model, x, threshold=tau)
            recon_err[start : start + bs] = err.cpu().numpy()
            ood[start : start + bs] = is_ood.cpu().numpy()
            pass_m = ~is_ood
            pred_b = torch.zeros(x.size(0), dtype=torch.long, device=device)
            prob_b = torch.zeros(x.size(0), device=device)
            if pass_m.any():
                pr = predict_embeddings_mlp(model, x[pass_m], binary=True)
                prob_b[pass_m] = pr
                pred_b[pass_m] = (pr >= mlp_threshold).long()
            pred[start : start + bs] = pred_b.cpu().numpy()
            mlp_probs[start : start + bs] = prob_b.cpu().numpy()

    return pred, mlp_probs, recon_err, ood.astype(np.bool_)
