"""
MLP Binary Classifier — BirdNET Confidence-Score Embeddings

Reads 1024-D PCA embeddings from data/embeddings/ (produced by
extract_embeddings.py) and trains a 3-layer MLP to classify each
3-second segment as bird (1) or noise (0).

Architecture
------------
Linear(input_dim → 512) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(512 → 256)        → BatchNorm1d → ReLU → Dropout(0.3)
Linear(256 → 1)          ← raw logit, NO sigmoid here

Training
--------
- Loss   : BCEWithLogitsLoss  (numerically stable; works directly on logits)
- Optim  : Adam lr=1e-3
- Epochs : up to 30, early stopping patience=5 on val_loss
- pos_weight balanced for bird/noise class imbalance (~10:1)

Threshold selection
-------------------
After training, logits are converted to probabilities via sigmoid and
evaluated at thresholds [0.3, 0.5, 0.7]. The threshold that maximises
val F1 is saved as the operational default.

Outputs
-------
models/saved/mlp_classifier.pt           state dict + metadata
models/saved/mlp_config.json             input_dim, best_threshold
results/mlp_metrics.json                 per-threshold + overall metrics
results/training_logs/mlp_train_log.csv  epoch-level training history

Usage
-----
    python models/mlp_classifier.py
    python models/mlp_classifier.py --epochs 25 --lr 5e-4 --patience 7
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST   = Path("data/embeddings/manifest.csv")
SAVED_DIR  = Path("models/saved")
RESULTS    = Path("results")
TRAIN_LOGS = RESULTS / "training_logs"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EmbeddingDataset(Dataset):
    """Reads pre-computed .npy embeddings for one split from the manifest."""

    def __init__(self, manifest: pd.DataFrame, split: str):
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        # Keep only rows whose file actually exists
        mask  = rows["path"].apply(lambda p: Path(p).exists())
        rows  = rows[mask].reset_index(drop=True)
        if (~mask).any():
            missing = (~mask).sum()
            print(f"[MLP]  WARN: {missing} missing .npy files skipped in '{split}' split.")

        self.paths  = rows["path"].tolist()
        self.labels = torch.tensor(
            rows["binary_label"].astype("float32").values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        emb = np.load(self.paths[idx]).astype("float32")
        return torch.from_numpy(emb), self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BirdMLP(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits — shape (B,). Apply sigmoid externally."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def _threshold_metrics(logits: np.ndarray, labels: np.ndarray,
                        threshold: float) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    epochs: int   = 30,
    lr: float     = 1e-3,
    batch_size: int = 256,
    patience: int = 5,
) -> dict:

    # -- Guard ----------------------------------------------------------------
    if not MANIFEST.exists():
        print(f"[MLP] Manifest not found: {MANIFEST}")
        print("[MLP] Run  python models/extract_embeddings.py  first.")
        sys.exit(1)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    TRAIN_LOGS.mkdir(parents=True, exist_ok=True)

    # -- Data -----------------------------------------------------------------
    manifest  = pd.read_csv(MANIFEST)

    # Verify binary_label column only contains {0, 1}
    assert set(manifest["binary_label"].unique()).issubset({0, 1}), \
        "binary_label column must contain only 0 (noise) and 1 (bird)."

    train_ds = EmbeddingDataset(manifest, "train")
    val_ds   = EmbeddingDataset(manifest, "val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[MLP] ERROR: empty train or val dataset — check manifest paths.")
        sys.exit(1)

    # Auto-detect input_dim from first embedding
    sample_emb, _ = train_ds[0]
    input_dim = sample_emb.shape[0]
    print(f"[MLP] Detected input_dim : {input_dim}")
    print(f"[MLP] Train samples      : {len(train_ds)}")
    print(f"[MLP] Val samples        : {len(val_ds)}")

    # Label counts (from manifest, not dataset, in case some files were skipped)
    tr_rows    = manifest[manifest["split"] == "train"]
    n_bird     = int((tr_rows["binary_label"] == 1).sum())
    n_noise    = int((tr_rows["binary_label"] == 0).sum())
    print(f"[MLP] Train class counts : bird={n_bird}, noise={n_noise}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

    # -- Model ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLP] Device             : {device}")

    model = BirdMLP(input_dim=input_dim).to(device)

    # BCEWithLogitsLoss with pos_weight to handle ~10:1 class imbalance.
    # pos_weight = n_noise / n_bird  → equal total contribution from each class.
    pos_weight = torch.tensor(
        [n_noise / max(n_bird, 1)], dtype=torch.float32
    ).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # -- Training loop --------------------------------------------------------
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    log_rows       = []

    for epoch in range(1, epochs + 1):

        # --- train ---
        model.train()
        running_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
        train_loss = running_loss / len(train_ds)

        # --- validate ---
        model.eval()
        val_running = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_running += criterion(logits, y).item() * len(y)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        val_loss  = val_running / len(val_ds)
        scheduler.step(val_loss)

        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels).astype(int)
        val_f1    = _threshold_metrics(logits_np, labels_np, 0.5)["f1"]

        print(
            f"[MLP] Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_f1@0.5={val_f1:.4f}"
        )
        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_f1":     round(val_f1, 4),
        })

        # --- early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[MLP] Early stop at epoch {epoch} "
                    f"(no val_loss improvement for {patience} epochs)."
                )
                break

    # -- Restore best weights -------------------------------------------------
    model.load_state_dict(best_state)

    # -- Threshold sweep on val set -------------------------------------------
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in val_dl:
            all_logits.append(model(X.to(device)).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)

    thresholds     = [0.3, 0.5, 0.7]
    thr_metrics    = [_threshold_metrics(logits_np, labels_np, t) for t in thresholds]
    best_thr_row   = max(thr_metrics, key=lambda r: r["f1"])
    best_threshold = best_thr_row["threshold"]

    print("\n[MLP] Threshold sweep (val):")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    for row in thr_metrics:
        marker = " <-- best" if row["threshold"] == best_threshold else ""
        print(
            f"  {row['threshold']:>10.1f}  "
            f"{row['precision']:>10.4f}  "
            f"{row['recall']:>8.4f}  "
            f"{row['f1']:>8.4f}{marker}"
        )

    # -- Save model -----------------------------------------------------------
    ckpt_path = SAVED_DIR / "mlp_classifier.pt"
    torch.save(
        {
            "state_dict":      model.state_dict(),
            "input_dim":       input_dim,
            "best_threshold":  best_threshold,
        },
        str(ckpt_path),
    )
    print(f"\n[MLP] Model saved       : {ckpt_path}")

    # -- Save config ----------------------------------------------------------
    config = {"input_dim": input_dim, "best_threshold": best_threshold}
    config_path = SAVED_DIR / "mlp_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[MLP] Config saved      : {config_path}")

    # -- Save metrics ---------------------------------------------------------
    metrics_out = {
        "best_threshold":      best_threshold,
        "per_threshold":       thr_metrics,
        "best_val_loss":       round(best_val_loss, 6),
        "n_train":             len(train_ds),
        "n_val":               len(val_ds),
        "train_class_balance": {"bird": n_bird, "noise": n_noise},
    }
    metrics_path = RESULTS / "mlp_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[MLP] Metrics saved     : {metrics_path}")

    # -- Save training log ----------------------------------------------------
    log_path = TRAIN_LOGS / "mlp_train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[MLP] Training log saved: {log_path}")

    return metrics_out


# ---------------------------------------------------------------------------
# Inference helper (used by inference_pipeline.py)
# ---------------------------------------------------------------------------
def load_model(device: torch.device | None = None) -> tuple[nn.Module, float]:
    """
    Returns (model, best_threshold).
    model is in eval mode on the given device.
    """
    ckpt_path = SAVED_DIR / "mlp_classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {ckpt_path}")

    if device is None:
        device = torch.device("cpu")

    ckpt      = torch.load(str(ckpt_path), map_location=device)
    input_dim = ckpt.get("input_dim", 1024)
    threshold = ckpt.get("best_threshold", 0.5)

    model = BirdMLP(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train MLP binary classifier on BirdNET PCA embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Maximum training epochs.")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int,   default=256,
                        help="Mini-batch size.")
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Early stopping patience (epochs).")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
