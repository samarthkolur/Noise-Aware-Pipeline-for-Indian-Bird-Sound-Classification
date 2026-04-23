"""
Three-Stage Inference Pipeline

Stage 1 — Autoencoder OOD gate
    Compute reconstruction MSE for the embedding.
    If MSE > ae_threshold  ->  label = "noise", skip Stage 2.

Stage 2 — MLP + Temperature Scaling
    logit  = MLP(embedding)
    prob   = sigmoid(logit / T)
    label  = "bird" if prob >= mlp_threshold else "noise"

Stage 3 — Species lookup
    If label == "bird", pull the top BirdNET species + confidence from
    results/processed_predictions.json (matched by filename stem).
    If label == "noise", top_prediction = None.

Outputs
-------
results/mlp_predictions.json      MLP + temperature only (no AE gate)
results/mlp_ae_predictions.json   AE gate + MLP + temperature

Record format (compatible with evaluate_metrics.py)
----------------------------------------------------
{
  "file":            str,
  "ground_truth":    str,
  "is_noise":        bool,   <- ground-truth flag
  "top_prediction":  str|null,
  "top_confidence":  float|null,
  "correct":         bool
}

Usage
-----
    python models/inference_pipeline.py
    python models/inference_pipeline.py --split val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIFEST        = Path("data/embeddings/manifest.csv")
SAVED_DIR       = Path("models/saved")
RESULTS         = Path("results")
PROCESSED_PREDS = RESULTS / "processed_predictions.json"
MLP_OUT         = RESULTS / "mlp_predictions.json"
MLP_AE_OUT      = RESULTS / "mlp_ae_predictions.json"


# ---------------------------------------------------------------------------
# Species lookup by filename stem
# ---------------------------------------------------------------------------
def _build_species_lookup(preds_path: Path) -> dict:
    """stem -> {"top_prediction": str|None, "top_confidence": float|None}"""
    with open(preds_path) as f:
        records = json.load(f)
    return {Path(r["file"]).stem: r for r in records}


# ---------------------------------------------------------------------------
# Per-sample reconstruction MSE (single embedding, no batch dim)
# ---------------------------------------------------------------------------
def _recon_mse_single(ae_model, x_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        recon = ae_model(x_tensor.unsqueeze(0))
    return float(((recon - x_tensor.unsqueeze(0)) ** 2).mean())


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------
def run_inference(split: str = "val") -> None:

    # -- Guards ---------------------------------------------------------------
    required = {
        MANIFEST:                    "Run extract_embeddings.py first.",
        SAVED_DIR / "mlp_classifier.pt": "Run mlp_classifier.py first.",
        SAVED_DIR / "autoencoder.pt":    "Run autoencoder.py first.",
        SAVED_DIR / "ae_threshold.json": "Run autoencoder.py first.",
        PROCESSED_PREDS:             "Run run_processed_birdnet.py first.",
    }
    for path, hint in required.items():
        if not path.exists():
            print(f"[INFER] ERROR: {path} not found. {hint}")
            sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # -- Load models ----------------------------------------------------------
    from models.mlp_classifier import BirdMLP
    from models.autoencoder    import BirdAE

    device = torch.device("cpu")

    mlp_ckpt      = torch.load(str(SAVED_DIR / "mlp_classifier.pt"), map_location=device)
    input_dim     = mlp_ckpt["input_dim"]
    mlp_threshold = float(mlp_ckpt.get("best_threshold", 0.5))

    mlp = BirdMLP(input_dim=input_dim).to(device)
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.eval()

    ae_ckpt = torch.load(str(SAVED_DIR / "autoencoder.pt"), map_location=device)
    ae = BirdAE(input_dim=ae_ckpt["input_dim"]).to(device)
    ae.load_state_dict(ae_ckpt["state_dict"])
    ae.eval()

    with open(SAVED_DIR / "ae_threshold.json") as f:
        ae_threshold = float(json.load(f)["threshold"])

    T = 1.0
    temp_file = SAVED_DIR / "temperature.json"
    if temp_file.exists():
        with open(temp_file) as f:
            T = float(json.load(f)["temperature"])

    print(f"[INFER] MLP threshold : {mlp_threshold}")
    print(f"[INFER] AE  threshold : {ae_threshold:.8f}")
    print(f"[INFER] Temperature T : {T:.6f}")

    # -- Species lookup -------------------------------------------------------
    species_lookup = _build_species_lookup(PROCESSED_PREDS)
    print(f"[INFER] Species lookup: {len(species_lookup)} entries")

    # -- Load manifest rows ---------------------------------------------------
    manifest = pd.read_csv(MANIFEST)
    val_rows = manifest[manifest["split"] == split].reset_index(drop=True)
    print(f"[INFER] {split} samples  : {len(val_rows)}")

    # -- Inference loop -------------------------------------------------------
    mlp_records    = []
    mlp_ae_records = []
    n_ae_blocked   = 0
    n_missing      = 0

    for _, row in val_rows.iterrows():
        emb_path = Path(row["path"])
        if not emb_path.exists():
            n_missing += 1
            continue

        x_np     = np.load(str(emb_path)).astype("float32")
        x_tensor = torch.from_numpy(x_np).to(device)

        ground_truth = row["label"].lower()
        is_noise_gt  = bool(row["binary_label"] == 0)
        stem         = Path(row["source_file"]).stem

        birdnet_rec  = species_lookup.get(stem, {})
        birdnet_spp  = birdnet_rec.get("top_prediction")   # None if BirdNET had no detection
        birdnet_conf = birdnet_rec.get("top_confidence")

        # ---- Stage 2: MLP + Temperature ------------------------------------
        with torch.no_grad():
            logit = float(mlp(x_tensor.unsqueeze(0)).squeeze())
        prob = 1.0 / (1.0 + np.exp(-logit / T))

        mlp_is_bird = prob >= mlp_threshold

        if mlp_is_bird:
            # Use BirdNET species when available; otherwise sentinel so
            # evaluate_metrics.py counts this as a bird detection (not noise).
            mlp_top_pred = birdnet_spp if birdnet_spp is not None else "__bird__"
            mlp_top_conf = round(float(prob), 6)
        else:
            mlp_top_pred = None
            mlp_top_conf = None

        # correct: True only when binary label AND species both match.
        # Species is considered correct if BirdNET fired and matched ground_truth.
        mlp_species_correct = (birdnet_spp is not None and birdnet_spp == ground_truth)
        mlp_correct = (
            (not mlp_is_bird and is_noise_gt)
            or (mlp_is_bird and not is_noise_gt and mlp_species_correct)
        )

        mlp_records.append({
            "file":           row["source_file"],
            "ground_truth":   ground_truth,
            "is_noise":       is_noise_gt,
            "top_prediction": mlp_top_pred,
            "top_confidence": mlp_top_conf,
            "correct":        bool(mlp_correct),
        })

        # ---- Stage 1: AE gate ----------------------------------------------
        mse = _recon_mse_single(ae, x_tensor)

        if mse > ae_threshold:
            n_ae_blocked += 1
            ae_is_bird   = False
            ae_top_pred  = None
            ae_top_conf  = None
        else:
            ae_is_bird   = mlp_is_bird
            ae_top_pred  = mlp_top_pred
            ae_top_conf  = mlp_top_conf

        ae_species_correct = (birdnet_spp is not None and birdnet_spp == ground_truth)
        ae_correct = (
            (not ae_is_bird and is_noise_gt)
            or (ae_is_bird and not is_noise_gt and ae_species_correct)
        )

        mlp_ae_records.append({
            "file":           row["source_file"],
            "ground_truth":   ground_truth,
            "is_noise":       is_noise_gt,
            "top_prediction": ae_top_pred,
            "top_confidence": ae_top_conf,
            "correct":        bool(ae_correct),
        })

    if n_missing:
        print(f"[INFER] WARN: {n_missing} embedding files missing, skipped.")
    print(f"[INFER] AE blocked     : {n_ae_blocked} / {len(mlp_ae_records)}")

    # -- Save outputs ---------------------------------------------------------
    with open(MLP_OUT, "w") as f:
        json.dump(mlp_records, f, indent=2)
    print(f"[INFER] MLP only       : {MLP_OUT}  ({len(mlp_records)} records)")

    with open(MLP_AE_OUT, "w") as f:
        json.dump(mlp_ae_records, f, indent=2)
    print(f"[INFER] MLP + AE       : {MLP_AE_OUT}  ({len(mlp_ae_records)} records)")

    # -- Quick detection F1 summary ------------------------------------------
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
    except ImportError:
        return

    def _det_f1(records: list, name: str) -> None:
        y_true = ["noise" if r["is_noise"] else "bird" for r in records]
        y_pred = ["noise" if r["top_prediction"] is None else "bird" for r in records]
        p = precision_score(y_true, y_pred, pos_label="bird", zero_division=0)
        r = recall_score(y_true,    y_pred, pos_label="bird", zero_division=0)
        f = f1_score(y_true,        y_pred, pos_label="bird", zero_division=0)
        print(f"[INFER] {name:14s}  precision={p:.4f}  recall={r:.4f}  F1={f:.4f}")

    print()
    _det_f1(mlp_records,    "MLP only")
    _det_f1(mlp_ae_records, "MLP + AE")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Three-stage inference: AE gate -> MLP+T -> species lookup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="val",
                        help="Manifest split to run inference on.")
    args = parser.parse_args()
    run_inference(split=args.split)


if __name__ == "__main__":
    main()
