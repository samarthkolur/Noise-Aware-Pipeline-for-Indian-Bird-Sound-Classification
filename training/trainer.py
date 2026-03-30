"""
trainer.py — Training & validation loop for embedding classifiers.

Supports:
  • Binary (BCE) and Multiclass (CrossEntropy) modes.
  • Inverse-frequency class weighting to handle imbalance.
  • F1-based checkpointing (instead of val_loss) for recall-sensitive tasks.
  • Optimal threshold search after training.
  • Confusion matrix logging.
  • Early stopping and best-model checkpointing.
  • AdamW optimizer with cosine LR scheduling.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset import build_dataloaders
from training.metrics import compute_metrics, compute_confusion_matrix, find_optimal_threshold
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """End-to-end training manager for embedding classifiers."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg)

        t_cfg = cfg.get("training", {})
        self.epochs = t_cfg.get("epochs", 50)
        self.lr = t_cfg.get("learning_rate", 1e-3)
        self.weight_decay = t_cfg.get("weight_decay", 1e-4)

        self.binary = cfg.get("model", {}).get("binary", False)

        self.checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        es_cfg = t_cfg.get("early_stopping", {})
        self.es_enabled = es_cfg.get("enabled", True)
        self.es_patience = es_cfg.get("patience", 7)
        self.es_min_delta = es_cfg.get("min_delta", 0.001)

    # ── Public API ──────────────────────────────────────────

    def fit(self) -> None:
        """Run the full training loop: data → build model → train/val → save."""
        # 1. Load Data
        train_loader, val_loader, _, label_encoder = build_dataloaders(
            self.cfg, binary=self.binary
        )
        self.label_encoder = label_encoder

        # 2. Build Model
        from models.classifier import EmbeddingClassifier
        self.model = EmbeddingClassifier(
            input_dim=self.cfg["embedding"]["embedding_dim"],
            num_classes=1 if self.binary else label_encoder.num_classes,
            hidden_dims=self.cfg["model"].get("hidden_dims", [512, 256]),
            dropout=self.cfg["model"].get("dropout", 0.3),
        ).to(self.device)

        # 3. Optimiser & Scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        # 4. Loss Function
        train_labels = train_loader.dataset.dataset.labels[
            train_loader.dataset.indices
        ]
        class_w = self._get_class_weights(train_labels, label_encoder.num_classes)
        criterion = self._build_loss(class_w)

        # Log class distribution
        if self.binary:
            n_bird = int((train_labels == 1).sum())
            n_noise = int((train_labels == 0).sum())
            logger.info(f"Train set: {n_bird} bird, {n_noise} noise "
                        f"(ratio {n_bird / max(n_noise, 1):.2f})")

        # 5. Loop — checkpoint on best F1 (recall-sensitive)
        best_f1 = -1.0
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training for {self.epochs} epochs "
                    f"({label_encoder.num_classes} classes, binary={self.binary})")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_metrics, val_logits, val_labels = self._validate(
                val_loader, criterion
            )
            scheduler.step()

            m = val_metrics
            logger.info(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {m['acc']:.3f} | Prec: {m['prec']:.3f} | "
                f"Rec: {m['rec']:.3f} | F1: {m['f1']:.3f}"
            )

            # Checkpoint on best F1 (not val_loss) for recall-sensitive tasks
            current_f1 = m["f1"]
            if current_f1 > best_f1 + self.es_min_delta:
                best_f1 = current_f1
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_metrics)
            else:
                patience_counter += 1

            if self.es_enabled and patience_counter >= self.es_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # 6. Post-training: optimal threshold + confusion matrix
        logger.info("Training complete ✓")
        self._post_training_analysis(val_logits, val_labels)

    # ── Private ─────────────────────────────────────────────

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for embs, labels in loader:
            embs = embs.to(self.device)
            labels = labels.to(self.device)

            if self.binary:
                labels = labels.float()

            logits = self.model(embs)
            if self.binary and logits.ndim > 1:
                logits = logits.squeeze(-1)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * embs.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _validate(
        self, loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, dict, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        for embs, labels in loader:
            embs = embs.to(self.device)
            labels = labels.to(self.device)

            if self.binary:
                labels_f = labels.float()
            else:
                labels_f = labels

            logits = self.model(embs)
            if self.binary and logits.ndim > 1:
                logits = logits.squeeze(-1)

            loss = criterion(logits, labels_f)
            total_loss += loss.item() * embs.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / len(loader.dataset)
        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)

        metrics = compute_metrics(cat_logits, cat_labels, binary=self.binary)
        return avg_loss, metrics, cat_logits, cat_labels

    def _post_training_analysis(
        self, val_logits: torch.Tensor, val_labels: torch.Tensor
    ) -> None:
        """After training: find optimal threshold and log confusion matrix."""
        if not self.binary:
            cm_str = compute_confusion_matrix(val_logits, val_labels, binary=False)
            logger.info(f"\n{cm_str}")
            return

        # Find optimal threshold
        result = find_optimal_threshold(
            val_logits, val_labels, metric="f1", steps=50
        )
        opt_thresh = result["best_threshold"]
        opt_f1 = result["best_value"]

        logger.info(f"Optimal threshold: {opt_thresh:.3f} (F1={opt_f1:.4f})")

        # Log confusion matrix at optimal threshold
        cm_str = compute_confusion_matrix(
            val_logits, val_labels, binary=True, threshold=opt_thresh
        )
        logger.info(f"\n{cm_str}")

        # Also log at default 0.5 for comparison
        cm_default = compute_confusion_matrix(
            val_logits, val_labels, binary=True, threshold=0.5
        )
        logger.info(f"\nAt default threshold=0.50:\n{cm_default}")

        # Save optimal threshold and curve into checkpoint metadata
        meta_path = self.checkpoint_dir / "best_model_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["optimal_threshold"] = opt_thresh
            meta["threshold_curve"] = result["curve"]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Saved optimal threshold ({opt_thresh:.3f}) to {meta_path}")

    def _get_class_weights(
        self, labels_np, num_classes: int
    ) -> Optional[torch.Tensor]:
        if self.binary:
            pos = (labels_np == 1).sum()
            neg = (labels_np == 0).sum()
            pos_weight = float(neg) / max(float(pos), 1.0)
            return torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
        else:
            from dataset.dataset import compute_class_weights
            w = compute_class_weights(labels_np, num_classes)
            return torch.from_numpy(w).to(self.device)

    def _build_loss(self, class_weights: torch.Tensor) -> nn.Module:
        if self.binary:
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            return nn.CrossEntropyLoss(weight=class_weights)

    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: dict) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        meta_path = self.checkpoint_dir / "best_model_meta.json"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
            "binary": self.binary,
            "num_classes": self.label_encoder.num_classes,
        }, str(path))

        with open(meta_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "val_loss": val_loss,
                "metrics": metrics,
                "binary": self.binary,
                "label_map": self.label_encoder.name2id,
            }, f, indent=2)

    @staticmethod
    def _resolve_device(cfg: dict) -> torch.device:
        device_str = cfg.get("project", {}).get("device", "auto")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
