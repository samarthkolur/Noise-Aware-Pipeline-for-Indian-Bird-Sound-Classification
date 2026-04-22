"""
Pure tensor/NN utilities for AE-gated embedding classification.

No file I/O and no config loading. Callers supply modules, thresholds, and tensors.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoencoder import EmbeddingAutoencoder


def predict_embeddings_ae_gate(
    autoencoder: nn.Module,
    embs: torch.Tensor,
    *,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autoencoder OOD gate from reconstruction error vs a scalar threshold.

    Args:
        autoencoder: Trained ``EmbeddingAutoencoder`` (or compatible forward).
        embs: Batch of embeddings ``(B, D)``.
        threshold: τ_AE; samples with error **strictly greater** than this are OOD.

    Returns:
        ``ood_mask`` — ``(B,)`` bool, True = reject before MLP (treat as noise).
        ``recon_err`` — ``(B,)`` per-sample MSE reconstruction error.
    """
    reconstructed, _ = autoencoder(embs)
    recon_err = EmbeddingAutoencoder.compute_reconstruction_error(embs, reconstructed)
    ood_mask = recon_err > threshold
    return ood_mask, recon_err


def predict_embeddings_mlp(
    classifier: nn.Module,
    embs: torch.Tensor,
    *,
    binary: bool,
) -> torch.Tensor:
    """Classifier forward on embeddings → class probabilities (no AE).

    Args:
        classifier: ``EmbeddingClassifier`` in eval mode.
        embs: ``(B, D)`` in-distribution batch (caller masks OOD rows).
        binary: If True, single-logit sigmoid → ``(B,)`` in ``(0, 1)``.
            If False, row-wise softmax → ``(B, C)``.

    Returns:
        Probability tensor on the same device/dtype policy as ``classifier`` output.
    """
    logits = classifier(embs)
    if binary:
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=1)


def route_probs_three_band(
    prob: float,
    *,
    high_threshold: float,
    low_threshold: float,
) -> Literal["bird", "noise", "uncertain"]:
    """Three-band routing for **binary** bird probability (matches ``Predictor``).

    Boundaries: ``bird`` if ``prob >= high_threshold``;
    ``noise`` if ``prob <= low_threshold``;
    else ``uncertain`` (strictly between low and high).
    """
    if prob >= high_threshold:
        return "bird"
    if prob <= low_threshold:
        return "noise"
    return "uncertain"


def gated_three_class_pred_tensor(
    probs: torch.Tensor,
    recon_err: torch.Tensor,
    tau: float,
    low_t: float,
    high_t: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized AE gate + three-band routing (binary probs).

    Args:
        probs: ``(N,)`` bird probabilities (ignored where AE rejects).
        recon_err: ``(N,)`` reconstruction MSE.
        tau: AE threshold τ_AE.
        low_t, high_t: MLP routing bands.

    Returns:
        ``pred`` — ``(N,)`` int64 labels ``0=noise, 1=bird, 2=uncertain``.
        ``ae_reject`` — ``(N,)`` bool, same condition as ``recon_err > tau``.
    """
    ae_reject = recon_err > tau
    pred = torch.zeros(probs.shape[0], dtype=torch.int64, device=probs.device)
    pred[ae_reject] = 0
    ok = ~ae_reject
    pred[ok & (probs >= high_t)] = 1
    pred[ok & (probs <= low_t)] = 0
    pred[ok & (probs > low_t) & (probs < high_t)] = 2
    return pred, ae_reject


def decision_binary_gated_single(
    autoencoder: nn.Module,
    classifier: nn.Module,
    emb_1b: torch.Tensor,
    *,
    threshold: float,
    high_threshold: float,
    low_threshold: float,
) -> Tuple[str, str, float, float, bool]:  # decision, species, prob, recon_err, ae_rejected
    """Single-segment binary pipeline: AE gate → optional MLP → three-band route.

    Args:
        emb_1b: Shape ``(1, D)``.

    Returns:
        ``(decision, species_label, prob, recon_error, ae_rejected)``.
        If AE rejects, ``prob`` is ``0.0`` and decision is ``noise``.
    """
    ood_mask, recon_err = predict_embeddings_ae_gate(
        autoencoder, emb_1b, threshold=threshold
    )
    r = float(recon_err.item())
    if bool(ood_mask.item()):
        return "noise", "noise", 0.0, r, True
    probs = predict_embeddings_mlp(classifier, emb_1b, binary=True)
    p = float(probs.item())
    decision = route_probs_three_band(
        p, high_threshold=high_threshold, low_threshold=low_threshold
    )
    species = "bird" if decision == "bird" else ("noise" if decision == "noise" else "uncertain")
    return decision, species, p, r, False
