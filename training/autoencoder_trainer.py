"""
autoencoder_trainer.py — Train EmbeddingAutoencoder on bird-only embeddings.

Reuses the same EmbeddingDataset / create_splits from dataset.dataset.
Filters to bird-only samples (label=1) so the autoencoder learns the bird
manifold; noise samples will naturally produce higher reconstruction error.

Saves:
    checkpoints/autoencoder.pt
    checkpoints/autoencoder_meta.json  (contains recon_threshold)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.dataset import EmbeddingDataset, create_splits
from models.autoencoder import EmbeddingAutoencoder
from utils.logger import get_logger

logger = get_logger(__name__)


class AutoencoderTrainer:
    """Train an EmbeddingAutoencoder on bird-only embeddings."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg)

        seed = cfg.get("project", {}).get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ae_cfg = cfg.get("autoencoder", {})
        self.epochs = int(ae_cfg.get("epochs", 30))
        self.batch_size = int(ae_cfg.get("batch_size", 64))
        self.lr = float(ae_cfg.get("learning_rate", 1e-3))

        chkpt_path_raw = ae_cfg.get("checkpoint_path", "./checkpoints/autoencoder.pt")
        self.checkpoint_path = Path(chkpt_path_raw)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.checkpoint_path.with_name("autoencoder_meta.json")

        # Load the SAME dataset used by the classifier
        binary = bool(cfg.get("model", {}).get("binary", True))
        embeddings_dir = Path(cfg["data"]["embeddings_dir"])
        manifest = embeddings_dir / "manifest.csv"
        if manifest.exists():
            dataset = EmbeddingDataset.from_manifest(manifest, binary=binary)
        else:
            dataset = EmbeddingDataset.from_directory(embeddings_dir, binary=binary)

        ds_cfg = cfg.get("dataset", {})
        splits = create_splits(
            dataset,
            val_frac=ds_cfg.get("val_split", 0.15),
            test_frac=ds_cfg.get("test_split", 0.10),
            stratify=ds_cfg.get("stratify", True),
            seed=seed,
        )

        # Filter to bird-only (label == 1)
        train_bird_idx = [i for i in splits.train_idx if dataset.labels[i] == 1]
        val_bird_idx = [i for i in splits.val_idx if dataset.labels[i] == 1]

        # Also keep val noise indices for threshold computation
        val_noise_idx = [i for i in splits.val_idx if dataset.labels[i] == 0]

        logger.info(
            f"Autoencoder dataset: {len(train_bird_idx)} train bird, "
            f"{len(val_bird_idx)} val bird, {len(val_noise_idx)} val noise"
        )

        self.train_loader = DataLoader(
            Subset(dataset, train_bird_idx),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.val_bird_loader = DataLoader(
            Subset(dataset, val_bird_idx),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.val_noise_loader = DataLoader(
            Subset(dataset, val_noise_idx),
            batch_size=self.batch_size,
            shuffle=False,
        ) if val_noise_idx else None

        emb_dim = int(cfg["embedding"]["embedding_dim"])
        self.model = EmbeddingAutoencoder(input_dim=emb_dim).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.loss_fn = nn.MSELoss()

    def fit(self) -> None:
        """Train autoencoder, save best checkpoint, compute recon threshold."""
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._eval_epoch(self.val_bird_loader)
            self.scheduler.step()

            logger.info(
                f"AE Epoch {epoch}/{self.epochs}  "
                f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_checkpoint(epoch, val_loss)

        # Load best weights for threshold computation
        self._load_best()

        # Compute reconstruction threshold (P95 of bird errors on val set)
        recon_threshold = self._compute_threshold()
        self._write_meta(recon_threshold, best_val_loss, best_epoch)

        logger.info(
            f"Autoencoder training complete ✓ "
            f"(best epoch={best_epoch}, val_mse={best_val_loss:.6f}, "
            f"recon_threshold={recon_threshold:.6f})"
        )

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for emb, _labels in self.train_loader:
            emb = emb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = self.model(emb)
            loss = self.loss_fn(reconstructed, emb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for emb, _labels in loader:
            emb = emb.to(self.device, non_blocking=True)
            reconstructed, _ = self.model(emb)
            loss = self.loss_fn(reconstructed, emb)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _compute_threshold(self) -> float:
        """P95 percentile of bird reconstruction errors on validation set.

        Any embedding with error above this is likely not a bird.
        """
        self.model.eval()
        errors: List[float] = []

        for emb, _labels in self.val_bird_loader:
            emb = emb.to(self.device, non_blocking=True)
            reconstructed, _ = self.model(emb)
            per_sample = EmbeddingAutoencoder.compute_reconstruction_error(emb, reconstructed)
            errors.extend(per_sample.cpu().tolist())

        if not errors:
            logger.warning("No bird validation samples for threshold — using 0.01")
            return 0.01

        threshold = float(np.percentile(errors, 95))

        # Log noise stats if available
        if self.val_noise_loader is not None:
            noise_errors: List[float] = []
            for emb, _labels in self.val_noise_loader:
                emb = emb.to(self.device, non_blocking=True)
                reconstructed, _ = self.model(emb)
                per_sample = EmbeddingAutoencoder.compute_reconstruction_error(emb, reconstructed)
                noise_errors.extend(per_sample.cpu().tolist())
            if noise_errors:
                logger.info(
                    f"AE threshold={threshold:.6f} | "
                    f"bird_err_mean={np.mean(errors):.6f} | "
                    f"noise_err_mean={np.mean(noise_errors):.6f}"
                )

        return threshold

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_mse": val_loss,
            },
            self.checkpoint_path,
        )

    def _load_best(self) -> None:
        chk = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(chk["model_state_dict"])

    def _write_meta(self, threshold: float, best_mse: float, best_epoch: int) -> None:
        meta = {
            "recon_threshold": threshold,
            "best_val_mse": best_mse,
            "best_epoch": best_epoch,
            "embedding_dim": int(self.cfg["embedding"]["embedding_dim"]),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Wrote {self.meta_path}")

    @staticmethod
    def _resolve_device(cfg: dict) -> torch.device:
        s = cfg.get("project", {}).get("device", "auto")
        if s == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(s)
