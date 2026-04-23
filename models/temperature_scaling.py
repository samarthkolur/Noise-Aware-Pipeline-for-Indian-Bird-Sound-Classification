"""
Temperature Scaling — Post-hoc MLP Calibration

Adjusts the MLP's predicted probabilities with a single learned scalar T:

    p_calibrated = sigmoid(logit / T)

T is tuned by minimising Negative Log-Likelihood (NLL) on the
validation set only.  The MLP weights are NEVER updated.

Steps
-----
1. If results/mlp_val_logits.npy / mlp_val_labels.npy are missing,
   run the frozen MLP on the val split to produce them.
2. Optimise T in (0.1, 10.0) via scipy.optimize.minimize_scalar.
3. Compute ECE before and after scaling (15 equal-width bins).
4. Save results and plot calibration curves (before vs after).

Outputs
-------
results/mlp_val_logits.npy           raw val logits  (generated if absent)
results/mlp_val_labels.npy           val binary labels (generated if absent)
models/saved/temperature.json        {"temperature": T}
results/calibration_improvement.json ECE before/after + optimal T
results/calibration_curve.png        reliability diagram (before vs after)

Usage
-----
    python models/temperature_scaling.py
    python models/temperature_scaling.py --force   # regenerate logits too
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `models.*` imports work whether the
# script is run directly or as part of the pipeline.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SAVED_DIR   = Path("models/saved")
RESULTS     = Path("results")
MANIFEST    = Path("data/embeddings/manifest.csv")

LOGITS_FILE = RESULTS / "mlp_val_logits.npy"
LABELS_FILE = RESULTS / "mlp_val_labels.npy"
TEMP_FILE   = SAVED_DIR / "temperature.json"
CAL_FILE    = RESULTS / "calibration_improvement.json"
CURVE_FILE  = RESULTS / "calibration_curve.png"


# ---------------------------------------------------------------------------
# Step 1 — generate val logits from the frozen MLP (if not already saved)
# ---------------------------------------------------------------------------
def _generate_val_logits(force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the trained MLP on the val split in eval mode (no gradient updates).
    Saves mlp_val_logits.npy and mlp_val_labels.npy, then returns them.
    """
    if LOGITS_FILE.exists() and LABELS_FILE.exists() and not force:
        print("[TEMP] Loading cached val logits/labels …")
        return np.load(LOGITS_FILE), np.load(LABELS_FILE)

    print("[TEMP] Generating val logits from frozen MLP …")

    # Guard
    ckpt_path = SAVED_DIR / "mlp_classifier.pt"
    if not ckpt_path.exists():
        print(f"[TEMP] ERROR: {ckpt_path} not found. Run mlp_classifier.py first.")
        sys.exit(1)
    if not MANIFEST.exists():
        print(f"[TEMP] ERROR: {MANIFEST} not found. Run extract_embeddings.py first.")
        sys.exit(1)

    import torch
    from torch.utils.data import DataLoader
    from models.mlp_classifier import BirdMLP, EmbeddingDataset

    manifest = pd.read_csv(MANIFEST)
    val_ds   = EmbeddingDataset(manifest, "val")
    val_dl   = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    model  = BirdMLP(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()   # <- inference only, weights frozen

    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in val_dl:
            all_logits.append(model(X).numpy())
            all_labels.append(y.numpy())

    logits = np.concatenate(all_logits).astype("float32")
    labels = np.concatenate(all_labels).astype("int32")

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(str(LOGITS_FILE), logits)
    np.save(str(LABELS_FILE), labels)
    print(f"[TEMP] Saved: {LOGITS_FILE}  ({logits.shape})")
    print(f"[TEMP] Saved: {LABELS_FILE}  ({labels.shape})")
    return logits, labels


# ---------------------------------------------------------------------------
# ECE — Expected Calibration Error (equal-width bins)
# ---------------------------------------------------------------------------
def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Bins are equal-width over [0, 1].  Empty bins are skipped.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    n         = len(probs)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc  = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += mask.sum() * abs(acc - conf)

    return float(ece / n)


# ---------------------------------------------------------------------------
# NLL objective used by scipy optimiser
# ---------------------------------------------------------------------------
def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of sigmoid(logit / T) against binary labels."""
    T     = max(float(T), 1e-8)
    probs = 1.0 / (1.0 + np.exp(-logits / T))
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return -float(np.mean(
        labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)
    ))


# ---------------------------------------------------------------------------
# Reliability-diagram data  (for plotting)
# ---------------------------------------------------------------------------
def _reliability_data(probs: np.ndarray, labels: np.ndarray,
                       n_bins: int = 15) -> tuple[list, list, list]:
    """Returns (bin_centres, mean_accuracy, bin_counts)."""
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        centres.append(float((lo + hi) / 2))
        accs.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    return centres, accs, counts


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------
def _plot_calibration(
    probs_before: np.ndarray,
    probs_after:  np.ndarray,
    labels:       np.ndarray,
    ece_before:   float,
    ece_after:    float,
    T:            float,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n_bins = 15

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, ece, title_suffix in [
        (axes[0], probs_before, ece_before, f"Before  (T=1.00, ECE={ece_before:.4f})"),
        (axes[1], probs_after,  ece_after,  f"After   (T={T:.4f}, ECE={ece_after:.4f})"),
    ]:
        centres, accs, counts = _reliability_data(probs, labels, n_bins)

        # Bar width proportional to bin width
        width = 1.0 / n_bins

        # Calibration bars
        ax.bar(centres, accs, width=width * 0.9, alpha=0.7,
               color="steelblue", label="Fraction positive")

        # Gap to perfect calibration (red)
        for c, a in zip(centres, accs):
            lo, hi = (a, c) if a < c else (c, a)
            ax.bar(c, hi - lo, width=width * 0.9, bottom=lo,
                   alpha=0.4, color="tomato")

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability Diagram — {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate bin counts along x axis
        for c, cnt in zip(centres, counts):
            ax.text(c, -0.06, str(cnt), ha="center", va="top",
                    fontsize=6, rotation=90, color="grey")

    fig.tight_layout()
    fig.savefig(str(CURVE_FILE), dpi=150)
    plt.close(fig)
    print(f"[TEMP] Calibration curve saved : {CURVE_FILE}")


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------
def calibrate(force: bool = False) -> dict:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # -- Step 1: obtain raw val logits ----------------------------------------
    logits, labels = _generate_val_logits(force=force)
    labels = labels.astype(int)

    print(f"[TEMP] Val set : {len(logits)} samples  "
          f"(bird={labels.sum()}, noise={(labels == 0).sum()})")

    # -- Step 2: probabilities before scaling (T = 1) -------------------------
    probs_before = 1.0 / (1.0 + np.exp(-logits))
    ece_before   = _ece(probs_before, labels)
    print(f"[TEMP] ECE before scaling : {ece_before:.6f}  (T = 1.00)")

    # -- Step 3: optimise T on val NLL ----------------------------------------
    result = minimize_scalar(
        _nll,
        bounds=(0.1, 10.0),
        method="bounded",
        args=(logits, labels),
        options={"xatol": 1e-6, "maxiter": 500},
    )
    T_opt = float(result.x)
    print(f"[TEMP] Optimal T          : {T_opt:.6f}  "
          f"(NLL = {result.fun:.6f})")

    # -- Step 4: probabilities after scaling ----------------------------------
    probs_after = 1.0 / (1.0 + np.exp(-logits / T_opt))
    ece_after   = _ece(probs_after, labels)
    print(f"[TEMP] ECE after  scaling : {ece_after:.6f}  "
          f"(delta = {ece_before - ece_after:+.6f})")

    # -- Step 5a: save temperature --------------------------------------------
    with open(TEMP_FILE, "w") as f:
        json.dump({"temperature": T_opt}, f, indent=2)
    print(f"[TEMP] Temperature saved  : {TEMP_FILE}")

    # -- Step 5b: save calibration improvement report -------------------------
    report = {
        "temperature":       round(T_opt, 8),
        "ece_before":        round(ece_before, 8),
        "ece_after":         round(ece_after,  8),
        "ece_reduction":     round(ece_before - ece_after, 8),
        "nll_before":        round(_nll(1.0, logits, labels), 8),
        "nll_after":         round(float(result.fun), 8),
        "n_val_samples":     int(len(logits)),
        "optimiser":         "scipy.minimize_scalar (bounded, Brent)",
        "T_search_bounds":   [0.1, 10.0],
    }
    with open(CAL_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[TEMP] Calibration report : {CAL_FILE}")

    # -- Step 6: plot calibration curves --------------------------------------
    _plot_calibration(
        probs_before, probs_after, labels,
        ece_before, ece_after, T_opt
    )

    # -- Summary --------------------------------------------------------------
    print("\n[TEMP] Summary:")
    print(f"  Optimal temperature T : {T_opt:.4f}")
    print(f"  ECE before            : {ece_before:.4f}")
    print(f"  ECE after             : {ece_after:.4f}")
    print(f"  ECE reduction         : {ece_before - ece_after:+.4f}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Temperature scaling calibration for the trained MLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate mlp_val_logits.npy even if it already exists.",
    )
    args = parser.parse_args()
    calibrate(force=args.force)


if __name__ == "__main__":
    main()
