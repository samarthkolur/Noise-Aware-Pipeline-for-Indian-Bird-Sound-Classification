"""
Autoencoder — Out-of-Distribution Detection on BirdNET Embeddings

Trained EXCLUSIVELY on bird embeddings (binary_label=1, train split).
Noise segments are never seen during training.

At inference, segments whose reconstruction MSE exceeds a learned
threshold are flagged as out-of-distribution (noise).  The threshold
is chosen by sweeping percentiles of the training-bird MSE distribution
and selecting the one that maximises F1 on the validation set.

Architecture
------------
Encoder: Linear(input_dim -> 512) -> ReLU -> Linear(512 -> 256) -> ReLU
Decoder: Linear(256 -> 512)       -> ReLU -> Linear(512 -> input_dim)

Loss / Optimiser
----------------
MSELoss, Adam lr=1e-3, ReduceLROnPlateau scheduler

Training
--------
- Only bird samples from the training split.
- 10% of train-bird samples held out as an internal AE val set for
  early stopping (kept strictly separate from the official val split).
- Up to 30 epochs; early stopping patience=5 on internal val MSE.

Threshold selection
-------------------
After training:
1. Compute per-sample reconstruction MSE on ALL training-bird samples
   (the full set, not just the internal AE-val subset).
2. Derive candidate thresholds at percentiles [80, 85, 90, 95, 99].
3. For each threshold, classify the OFFICIAL val split (bird + noise)
   and compute F1 (positive = bird, negative = noise).
4. Select the threshold with the highest val F1.

Outputs
-------
models/saved/autoencoder.pt        state dict + metadata
models/saved/ae_threshold.json     best threshold + percentile + val F1
results/ae_metrics.json            per-percentile breakdown + summary
results/training_logs/ae_train_log.csv  epoch-level train/val MSE

Usage
-----
    python models/autoencoder.py
    python models/autoencoder.py --epochs 25 --patience 7
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST   = Path("data/embeddings/manifest.csv")
SAVED_DIR  = Path("models/saved")
RESULTS    = Path("results")
TRAIN_LOGS = RESULTS / "training_logs"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BirdAE(nn.Module):
    """Symmetric bottleneck autoencoder: input_dim -> 512 -> 256 -> 512 -> input_dim."""

    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),       nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),       nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Helper — load embeddings for one group from manifest
# ---------------------------------------------------------------------------
def _load_embeddings(manifest: pd.DataFrame,
                     split: str,
                     binary_label: int | None = None) -> np.ndarray:
    """
    Load all .npy embeddings that match split (and optionally binary_label).
    Returns float32 array of shape (N, input_dim).  Skips missing files.
    """
    rows = manifest[manifest["split"] == split]
    if binary_label is not None:
        rows = rows[rows["binary_label"] == binary_label]
    rows = rows.reset_index(drop=True)

    embs = []
    missing = 0
    for _, row in rows.iterrows():
        p = Path(row["path"])
        if p.exists():
            embs.append(np.load(str(p)).astype("float32"))
        else:
            missing += 1

    if missing:
        print(f"[AE]  WARN: {missing} missing .npy file(s) skipped.")
    if not embs:
        raise RuntimeError(
            f"[AE] No embeddings loaded for split={split!r}, "
            f"binary_label={binary_label!r}."
        )
    return np.stack(embs, axis=0)


# ---------------------------------------------------------------------------
# Per-sample reconstruction MSE
# ---------------------------------------------------------------------------
def _recon_mse(model: BirdAE, X: np.ndarray,
               batch_size: int = 512,
               device: torch.device = torch.device("cpu")) -> np.ndarray:
    """Returns 1-D array of per-sample MSE values."""
    model.eval()
    mses = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon = model(batch)
            mse   = ((recon - batch) ** 2).mean(dim=1).cpu().numpy()
            mses.append(mse)
    return np.concatenate(mses)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    epochs:     int   = 30,
    lr:         float = 1e-3,
    batch_size: int   = 256,
    patience:   int   = 5,
) -> dict:

    # -- Guards ---------------------------------------------------------------
    if not MANIFEST.exists():
        print(f"[AE] Manifest not found: {MANIFEST}")
        print("[AE] Run  python models/extract_embeddings.py  first.")
        sys.exit(1)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    TRAIN_LOGS.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)

    # -- Load data ------------------------------------------------------------
    print("[AE] Loading train-bird embeddings (label=1, train split only) …")
    X_train_bird = _load_embeddings(manifest, "train", binary_label=1)
    input_dim    = X_train_bird.shape[1]
    print(f"[AE] Train-bird samples  : {len(X_train_bird)}")
    print(f"[AE] Input dim           : {input_dim}")

    print("[AE] Loading val embeddings (bird + noise) …")
    X_val_bird  = _load_embeddings(manifest, "val", binary_label=1)
    X_val_noise = _load_embeddings(manifest, "val", binary_label=0)
    print(f"[AE] Val bird            : {len(X_val_bird)}")
    print(f"[AE] Val noise           : {len(X_val_noise)}")

    # -- Internal AE val split (10% of train-bird) ----------------------------
    rng      = np.random.default_rng(seed=42)
    idx      = rng.permutation(len(X_train_bird))
    n_ae_val = max(1, int(0.10 * len(X_train_bird)))
    ae_val_idx   = idx[:n_ae_val]
    ae_train_idx = idx[n_ae_val:]

    X_ae_train = X_train_bird[ae_train_idx]
    X_ae_val   = X_train_bird[ae_val_idx]
    print(f"[AE] AE train / AE-val   : {len(X_ae_train)} / {len(X_ae_val)} "
          f"(internal 90/10 split of train-bird)")

    # -- DataLoaders ----------------------------------------------------------
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_ae_train)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )

    # -- Model ----------------------------------------------------------------
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AE] Device              : {device}")
    model     = BirdAE(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # -- Training loop --------------------------------------------------------
    best_val_mse = float("inf")
    best_state   = None
    no_improve   = 0
    log_rows     = []

    for epoch in range(1, epochs + 1):

        # train
        model.train()
        running = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        train_mse = running / len(X_ae_train)

        # internal val (bird only — just for early stopping)
        val_mse = float(_recon_mse(model, X_ae_val, device=device).mean())
        scheduler.step(val_mse)

        print(
            f"[AE] Epoch {epoch:3d}/{epochs}  "
            f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}"
        )
        log_rows.append({
            "epoch":     epoch,
            "train_mse": round(train_mse, 8),
            "val_mse":   round(val_mse, 8),
        })

        if val_mse < best_val_mse - 1e-9:
            best_val_mse = val_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[AE] Early stop at epoch {epoch} "
                    f"(no val_mse improvement for {patience} epochs)."
                )
                break

    model.load_state_dict(best_state)

    # -- Save model -----------------------------------------------------------
    ckpt_path = SAVED_DIR / "autoencoder.pt"
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": input_dim},
        str(ckpt_path),
    )
    print(f"\n[AE] Model saved         : {ckpt_path}")

    # -- Save training log ----------------------------------------------------
    log_path = TRAIN_LOGS / "ae_train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[AE] Training log saved  : {log_path}")

    # -- Reconstruction MSE on the full sets ----------------------------------
    print("\n[AE] Computing reconstruction MSE on full sets …")
    mse_train_bird  = _recon_mse(model, X_train_bird, device=device)
    mse_val_bird    = _recon_mse(model, X_val_bird,   device=device)
    mse_val_noise   = _recon_mse(model, X_val_noise,  device=device)

    print(f"[AE] MSE train-bird  : mean={mse_train_bird.mean():.6f}  "
          f"p95={np.percentile(mse_train_bird, 95):.6f}")
    print(f"[AE] MSE val-bird    : mean={mse_val_bird.mean():.6f}")
    print(f"[AE] MSE val-noise   : mean={mse_val_noise.mean():.6f}")

    # -- Threshold sweep ------------------------------------------------------
    # Candidate thresholds come from the training-bird MSE distribution.
    # Classification rule: mse <= threshold -> bird (1), else noise (0).
    X_val_all   = np.concatenate([X_val_bird, X_val_noise], axis=0)
    mse_val_all = np.concatenate([mse_val_bird, mse_val_noise])
    y_val_all   = np.array(
        [1] * len(X_val_bird) + [0] * len(X_val_noise), dtype=int
    )

    percentiles  = [80, 85, 90, 95, 99]
    sweep_rows   = []
    best_f1      = -1.0
    best_row     = None

    print(f"\n[AE] Threshold sweep on val F1:")
    print(f"  {'Pct':>4}  {'Threshold':>12}  {'Precision':>10}  "
          f"{'Recall':>8}  {'F1':>8}  {'Note'}")

    for pct in percentiles:
        thresh = float(np.percentile(mse_train_bird, pct))
        preds  = (mse_val_all <= thresh).astype(int)  # <=thresh -> bird
        prec   = float(precision_score(y_val_all, preds, zero_division=0))
        rec    = float(recall_score(y_val_all,    preds, zero_division=0))
        f1     = float(f1_score(y_val_all,        preds, zero_division=0))

        note = ""
        if f1 > best_f1:
            best_f1  = f1
            best_row = {"percentile": pct, "threshold": thresh,
                        "precision": prec, "recall": rec, "f1": f1}
            note = "<-- best"

        sweep_rows.append({
            "percentile": pct,
            "threshold":  round(thresh, 8),
            "precision":  round(prec, 4),
            "recall":     round(rec, 4),
            "f1":         round(f1, 4),
        })
        print(
            f"  {pct:>4}  {thresh:>12.6f}  {prec:>10.4f}  "
            f"{rec:>8.4f}  {f1:>8.4f}  {note}"
        )

    best_threshold = best_row["threshold"]
    best_pct       = best_row["percentile"]
    print(
        f"\n[AE] Best threshold      : {best_threshold:.6f}  "
        f"(p{best_pct}, val_F1={best_f1:.4f})"
    )

    # -- Save threshold -------------------------------------------------------
    thr_path = SAVED_DIR / "ae_threshold.json"
    with open(thr_path, "w") as f:
        json.dump(
            {
                "threshold":  best_threshold,
                "percentile": best_pct,
                "val_f1":     round(best_f1, 4),
            },
            f, indent=2,
        )
    print(f"[AE] Threshold saved     : {thr_path}")

    # -- Save ae_metrics.json -------------------------------------------------
    metrics = {
        "best_threshold":  best_threshold,
        "best_percentile": best_pct,
        "best_val_f1":     round(best_f1, 4),
        "mse_stats": {
            "train_bird_mean":  round(float(mse_train_bird.mean()), 8),
            "train_bird_std":   round(float(mse_train_bird.std()),  8),
            "val_bird_mean":    round(float(mse_val_bird.mean()),   8),
            "val_noise_mean":   round(float(mse_val_noise.mean()),  8),
            "separation_ratio": round(
                float(mse_val_noise.mean()) / max(float(mse_val_bird.mean()), 1e-12),
                4,
            ),
        },
        "percentile_sweep": sweep_rows,
        "n_train_bird": int(len(X_train_bird)),
        "n_val_bird":   int(len(X_val_bird)),
        "n_val_noise":  int(len(X_val_noise)),
    }
    met_path = RESULTS / "ae_metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[AE] Metrics saved       : {met_path}")

    return metrics


# ---------------------------------------------------------------------------
# Inference helper (used by inference_pipeline.py)
# ---------------------------------------------------------------------------
def load_model(device: torch.device | None = None) -> tuple:
    """Returns (BirdAE model in eval mode, threshold float)."""
    ckpt_path = SAVED_DIR / "autoencoder.pt"
    thr_path  = SAVED_DIR / "ae_threshold.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {ckpt_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"AE threshold not found: {thr_path}")

    if device is None:
        device = torch.device("cpu")

    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model = BirdAE(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(thr_path) as f:
        threshold = float(json.load(f)["threshold"])

    return model, threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train autoencoder OOD detector on bird embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
