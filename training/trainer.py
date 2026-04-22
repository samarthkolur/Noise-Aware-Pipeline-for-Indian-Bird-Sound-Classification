"""
trainer.py — Train EmbeddingClassifier on BirdNET (or other) embeddings.

Loads HDF5 embeddings via dataset.build_dataloaders, optimizes BCE (binary) or
CrossEntropy (multiclass), tracks accuracy / precision / recall / F1 on the
validation set, saves the best checkpoint by validation F1, and runs early
stopping. Writes ``best_model.pt`` and ``best_model_meta.json`` for inference.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import (
    LabelEncoder,
    build_dataloaders,
    compute_class_weights,
)
from models.classifier import EmbeddingClassifier
from utils.metrics import compute_metrics, find_optimal_threshold
from utils.logger import get_logger
from utils.run_metadata import dump_run_metadata

logger = get_logger(__name__)


# ── Losses ─────────────────────────────────────────────────


class BinaryFocalLoss(nn.Module):
    """Focal loss for single-logit binary classification."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits (B,), targets (B,) float 0/1
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = a_t * focal
        return focal.mean()


class MulticlassFocalLoss(nn.Module):
    """Focal loss for softmax multiclass (gamma on CE)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = -(1 - p_t).pow(self.gamma) * log_p_t
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def _build_loss(
    cfg: dict,
    binary: bool,
    num_classes: int,
    train_labels: np.ndarray,
    device: torch.device,
) -> Optional[nn.Module]:
    """Return a loss module, or ``None`` for binary BCE (handled with pos_weight in Trainer)."""
    tcfg = cfg.get("training", {})
    name = tcfg.get("loss", "cross_entropy")

    if binary:
        if name == "focal":
            fl = tcfg.get("focal_loss", {})
            alpha = fl.get("alpha")
            gamma = float(fl.get("gamma", 2.0))
            return BinaryFocalLoss(gamma=gamma, alpha=alpha).to(device)
        return None  # plain BCEWithLogits via functional + pos_weight

    weight_np = compute_class_weights(train_labels, num_classes)
    weight = torch.from_numpy(weight_np).float().to(device)

    if name == "focal":
        fl = tcfg.get("focal_loss", {})
        gamma = float(fl.get("gamma", 2.0))
        return MulticlassFocalLoss(gamma=gamma, weight=weight).to(device)

    if name == "label_smoothing":
        ls = float(tcfg.get("label_smoothing", 0.1))
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=ls).to(device)

    return nn.CrossEntropyLoss(weight=weight).to(device)


def _binary_logits_from_output(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim > 1:
        return logits.squeeze(-1)
    return logits


def _bce_targets(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    return labels.float().to(device)


# ── Trainer ────────────────────────────────────────────────


class Trainer:
    """Train ``EmbeddingClassifier`` on precomputed embeddings."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg)
        self.binary = bool(cfg.get("model", {}).get("binary", True))

        seed = int(cfg.get("project", {}).get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_loader, val_loader, test_loader, encoder = build_dataloaders(
            cfg, binary=self.binary
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_encoder: LabelEncoder = encoder

        # Full training label array for loss weights / pos_weight
        self._train_labels = self._gather_train_labels(cfg)

        num_classes = 1 if self.binary else self.label_encoder.num_classes
        self.model = EmbeddingClassifier(
            input_dim=int(cfg["embedding"]["embedding_dim"]),
            num_classes=num_classes,
            hidden_dims=list(cfg["model"].get("hidden_dims", [512, 256])),
            dropout=float(cfg["model"].get("dropout", 0.3)),
        ).to(self.device)

        self.loss_fn = _build_loss(
            cfg,
            self.binary,
            self.label_encoder.num_classes,
            self._train_labels,
            self.device,
        )

        n0 = max(1, int((self._train_labels == 0).sum()))
        n1 = max(1, int((self._train_labels == 1).sum()))
        self._bce_pos_weight = torch.tensor([n0 / n1], dtype=torch.float32)

        tcfg = cfg.get("training", {})
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(tcfg.get("learning_rate", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )

        epochs = int(tcfg.get("epochs", 50))
        self.epochs = epochs
        self.scheduler = self._build_scheduler(epochs)

        es = tcfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(es.get("enabled", True))
        self.patience = int(es.get("patience", 7))
        self.min_delta = float(es.get("min_delta", 0.001))

        self.checkpoint_dir = Path(tcfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _gather_train_labels(self, cfg: dict) -> np.ndarray:
        """Labels for training indices only (matches stratified split)."""
        from dataset.dataset import EmbeddingDataset, create_splits
        from torch.utils.data import Subset

        embeddings_dir = Path(cfg["data"]["embeddings_dir"])
        manifest = embeddings_dir / "manifest.csv"
        if manifest.exists():
            ds = EmbeddingDataset.from_manifest(manifest, binary=self.binary)
        else:
            ds = EmbeddingDataset.from_directory(embeddings_dir, binary=self.binary)

        splits = create_splits(
            ds,
            val_frac=cfg.get("dataset", {}).get("val_split", 0.15),
            test_frac=cfg.get("dataset", {}).get("test_split", 0.10),
            stratify=cfg.get("dataset", {}).get("stratify", True),
            seed=cfg.get("project", {}).get("seed", 42),
        )
        return ds.labels[splits.train_idx]

    def _build_scheduler(self, epochs: int) -> Optional[Any]:
        name = self.cfg.get("training", {}).get("scheduler", "cosine")
        if name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
        if name == "step":
            return StepLR(self.optimizer, step_size=max(1, epochs // 3), gamma=0.1)
        if name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3
            )
        return None

    @staticmethod
    def _resolve_device(cfg: dict) -> torch.device:
        s = cfg.get("project", {}).get("device", "auto")
        if s == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(s)
        logger.info(f"Using device: {device}")
        return device

    def _forward_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.binary:
            lg = _binary_logits_from_output(logits)
            y = _bce_targets(labels, lg.device)
            if isinstance(self.loss_fn, BinaryFocalLoss):
                return self.loss_fn(lg, y)
            pw = self._bce_pos_weight.to(device=lg.device, dtype=lg.dtype, non_blocking=True)
            return F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw)
        assert self.loss_fn is not None
        return self.loss_fn(logits, labels.to(self.device, non_blocking=True).long())

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for emb, labels in self.train_loader:
            emb = emb.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(emb)
            loss = self._forward_loss(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Return mean loss and metrics (acc, prec, rec, f1) at threshold 0.5 (binary)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for emb, labels in loader:
            emb = emb.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(emb)
            loss = self._forward_loss(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        cat_l = torch.cat(all_logits)
        cat_y = torch.cat(all_labels)
        avg_loss = total_loss / max(n_batches, 1)
        m = compute_metrics(cat_l, cat_y, binary=self.binary, threshold=0.5)
        return avg_loss, m

    def _label_map_for_meta(self) -> Dict[str, int]:
        if self.binary:
            enc = self.label_encoder
            return {enc.decode(0): 0, enc.decode(1): 1}
        return dict(self.label_encoder.name2id)

    def fit(self) -> None:
        """Full training with best checkpoint by validation F1."""
        best_f1 = -1.0
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._evaluate_loader(self.val_loader)

            logger.info(
                f"Epoch {epoch}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_metrics['acc']:.4f}  val_prec={val_metrics['prec']:.4f}  "
                f"val_rec={val_metrics['rec']:.4f}  val_f1={val_metrics['f1']:.4f}"
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["f1"])
                else:
                    self.scheduler.step()

            improved = val_metrics["f1"] > best_f1 + self.min_delta
            if improved:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                epochs_no_improve = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                epochs_no_improve += 1

            if self.early_stopping_enabled and epochs_no_improve >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (no +{self.min_delta} val F1 for "
                    f"{self.patience} epochs). Best: epoch {best_epoch}, F1={best_f1:.4f}"
                )
                break

        optimal_threshold = 0.5
        if self.binary and best_epoch >= 1:
            self._load_checkpoint_weights()
            val_logits, val_labels = self._collect_logits(self.val_loader)
            thr_res = find_optimal_threshold(
                val_logits, val_labels, metric="f1", steps=50
            )
            optimal_threshold = float(thr_res["best_threshold"])
            logger.info(
                f"Validation F1-optimal threshold: {optimal_threshold:.4f} "
                f"(F1={thr_res['best_value']:.4f})"
            )

        self._write_meta(optimal_threshold, best_f1, best_epoch)
        logger.info(f"Best model: epoch {best_epoch}, val_f1={best_f1:.4f}")

        run_meta_path = dump_run_metadata(self.cfg, trainer="EmbeddingClassifierTrainer")
        logger.info(f"Run metadata written → {run_meta_path}")

    @torch.no_grad()
    def _collect_logits(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        outs, ys = [], []
        for emb, labels in loader:
            emb = emb.to(self.device, non_blocking=True)
            logits = self.model(emb)
            if self.binary and logits.ndim > 1:
                logits = logits.squeeze(-1)
            outs.append(logits.cpu())
            ys.append(labels)
        return torch.cat(outs), torch.cat(ys)

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            path,
        )
        logger.info(f"Saved checkpoint → {path} (val_f1={val_metrics['f1']:.4f})")

    def _load_checkpoint_weights(self) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        chk = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(chk["model_state_dict"])

    def _write_meta(
        self,
        optimal_threshold: float,
        best_val_f1: float,
        best_epoch: int,
    ) -> None:
        meta = {
            "binary": self.binary,
            "label_map": self._label_map_for_meta(),
            "optimal_threshold": optimal_threshold,
            "best_val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch),
            "embedding_dim": int(self.cfg["embedding"]["embedding_dim"]),
        }
        out = self.checkpoint_dir / "best_model_meta.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Wrote {out}")
