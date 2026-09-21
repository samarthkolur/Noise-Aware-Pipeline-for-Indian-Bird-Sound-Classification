"""Training loops, early stopping, and checkpointing for FocalMLP + BirdAutoencoder
(design.md §6.3, §6.4, Phase 5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from pipeline.config import AutoencoderConfig, MLPConfig
from pipeline.losses import FocalLoss
from pipeline.model import BirdAutoencoder, FocalMLP

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    best_epoch: int = 0


def train_focal_mlp(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    config: MLPConfig,
    embedding_dim: int,
    sampler=None,
) -> tuple[FocalMLP, TrainHistory]:
    """Train FocalMLP with Adam + cosine annealing + early stopping on val F1."""
    model = FocalMLP(embedding_dim, config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    loss_fn = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)

    train_ds = TensorDataset(train_embeddings, train_labels)
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    history = TrainHistory()
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            if batch_x.shape[0] < 2:
                continue  # BatchNorm requires >= 2 samples per batch
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_embeddings)
            val_pred_labels = (val_preds >= 0.5).float()
            val_f1 = f1_score(val_labels.numpy(), val_pred_labels.numpy(), zero_division=0)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(mean_loss)
        history.val_metric.append(val_f1)
        logger.info(
            "epoch %d/%d train_loss=%.4f val_f1=%.4f",
            epoch + 1,
            config.max_epochs,
            mean_loss,
            val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (best val_f1=%.4f at epoch %d)",
                    epoch + 1,
                    best_val_f1,
                    history.best_epoch + 1,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_autoencoder(
    train_bird_embeddings: torch.Tensor,
    val_bird_embeddings: torch.Tensor,
    config: AutoencoderConfig,
    embedding_dim: int,
    max_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
) -> tuple[BirdAutoencoder, TrainHistory]:
    """Train the bird-only autoencoder; early-stop on val reconstruction MSE."""
    model = BirdAutoencoder(embedding_dim, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_ds = TensorDataset(train_bird_embeddings)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    history = TrainHistory()
    best_val_mse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_mse = float(model.reconstruction_error(val_bird_embeddings).mean().item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(mean_loss)
        history.val_metric.append(val_mse)
        logger.info(
            "AE epoch %d/%d train_loss=%.6f val_mse=%.6f", epoch + 1, max_epochs, mean_loss, val_mse
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "AE early stopping at epoch %d (best val_mse=%.6f)", epoch + 1, best_val_mse
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def compute_tau_ae(
    model: BirdAutoencoder, val_bird_embeddings: torch.Tensor, percentile: float
) -> float:
    """τ_AE = P{percentile} of reconstruction MSE over validation bird embeddings (DD-005)."""
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(val_bird_embeddings).numpy()
    return float(np.percentile(errors, percentile))
