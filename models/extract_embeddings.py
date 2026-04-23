"""
BirdNET Embedding Extraction — Per-Class Confidence Scores + PCA

Pipeline
--------
1. For every audio segment in splits/train.csv and splits/val.csv, call
   `analyzer.predict(chunk)` (BirdNET's public API) to get the raw
   per-class confidence scores — a 6522-D vector, one value per
   BirdNET V2.4 species class.  No TFLite internal layers are accessed.

2. Fit sklearn PCA (n_components=1024) ONLY on the training split.

3. Transform both splits with the fitted PCA → 1024-D embeddings.

4. Save each embedding as a .npy file and write manifest.csv.

Outputs
-------
data/embeddings/train/<label>/<filestem>.npy
data/embeddings/val/<label>/<filestem>.npy
data/embeddings/manifest.csv        — path, label, binary_label, split, source_file
models/saved/pca_transform.pkl      — fitted PCA (for inference-time use)
models/saved/birdnet_labels.json    — ordered label list (index → class name)

Binary label convention: 1 = bird, 0 = noise.

Usage
-----
    python models/extract_embeddings.py
    python models/extract_embeddings.py --n-components 512
    python models/extract_embeddings.py --force          # re-extract raw scores
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

EMB_ROOT  = Path("data/embeddings")
SAVED_DIR = Path("models/saved")
MANIFEST  = EMB_ROOT / "manifest.csv"


# ---------------------------------------------------------------------------
# Audio loading and raw score extraction
# ---------------------------------------------------------------------------
def _load_analyzer():
    try:
        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer
        return Analyzer, Recording
    except ImportError as exc:
        print(f"[EMBED] ERROR: birdnetlib not installed — {exc}")
        sys.exit(1)


def _raw_scores(analyzer, Recording, file_path: str) -> np.ndarray | None:
    """
    Returns a 1-D numpy array of shape (n_classes,) containing the raw
    per-class confidence scores produced by BirdNET for this audio file.

    Strategy
    --------
    - Use Recording.read_audio_data() to split the WAV into 3-second chunks
      (same pre-processing BirdNET uses internally).
    - Call analyzer.predict(chunk)[0] for each chunk to get the raw
      classifier output vector — no TFLite tensor index manipulation.
    - For multi-chunk files, take the element-wise maximum across chunks so
      the strongest signal for each class is preserved.
    - Returns None on any loading or inference error.
    """
    try:
        recording = Recording(analyzer, file_path, min_conf=0.1)
        recording.read_audio_data()
        chunks = recording.chunks
        if not chunks:
            return None

        chunk_preds = []
        for chunk in chunks:
            pred = analyzer.predict(chunk)[0]   # shape: (n_classes,)
            chunk_preds.append(pred)

        # Element-wise max: keep the highest confidence for each class
        scores = np.max(np.stack(chunk_preds, axis=0), axis=0)
        return scores.astype("float32")

    except Exception as exc:
        tqdm.write(f"[EMBED] WARN {os.path.basename(file_path)}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Build raw feature matrix for a split
# ---------------------------------------------------------------------------
def _build_raw_matrix(
    analyzer, Recording,
    df: pd.DataFrame,
    split_name: str,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Returns (X, file_paths, labels, binary_labels) for all processable rows.
    X shape: (N, n_classes).
    """
    rows_X, paths, labels, binaries = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{split_name}] raw scores"):
        fp    = row["file_path"].replace("\\", "/")
        label = row["label"]

        if not os.path.isfile(fp):
            tqdm.write(f"[EMBED] WARN file missing: {fp}")
            continue

        scores = _raw_scores(analyzer, Recording, fp)
        if scores is None:
            continue

        rows_X.append(scores)
        paths.append(fp)
        labels.append(label)
        binaries.append(0 if label.lower() == "noise" else 1)

    if not rows_X:
        return np.empty((0, len(analyzer.labels))), [], [], []

    return np.stack(rows_X, axis=0), paths, labels, binaries


# ---------------------------------------------------------------------------
# Save embeddings and build manifest rows
# ---------------------------------------------------------------------------
def _save_embeddings(
    X_pca: np.ndarray,
    file_paths: list[str],
    labels: list[str],
    binaries: list[int],
    split_name: str,
) -> list[dict]:
    rows = []
    for emb, fp, label, binary in zip(X_pca, file_paths, labels, binaries):
        stem     = Path(fp).stem
        out_path = EMB_ROOT / split_name / label / f"{stem}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), emb.astype("float32"))
        rows.append({
            "path":         str(out_path),
            "label":        label,
            "binary_label": binary,
            "split":        split_name,
            "source_file":  fp,
        })
    return rows


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
def extract_embeddings(
    split_csvs: list[str] | None = None,
    n_components: int = 1024,
    force: bool = False,
) -> pd.DataFrame:
    if split_csvs is None:
        split_csvs = ["splits/train.csv", "splits/val.csv"]

    # Identify train and val CSV paths
    train_csv = next((p for p in split_csvs if "train" in Path(p).stem), None)
    val_csv   = next((p for p in split_csvs if "val"   in Path(p).stem), None)

    if train_csv is None or val_csv is None:
        print("[EMBED] ERROR: supply both a train and a val CSV via --splits.")
        sys.exit(1)

    EMB_ROOT.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if manifest already exists and skip if not forced
    if MANIFEST.exists() and not force:
        print(f"[EMBED] Manifest already exists: {MANIFEST}. Use --force to re-extract.")
        return pd.read_csv(MANIFEST)

    Analyzer, Recording = _load_analyzer()
    print("[EMBED] Loading BirdNET analyzer …")
    analyzer = Analyzer()
    n_classes = len(analyzer.labels)
    print(f"[EMBED] BirdNET has {n_classes} classes.")

    # Save ordered label list for downstream use
    labels_path = SAVED_DIR / "birdnet_labels.json"
    with open(labels_path, "w") as f:
        json.dump(analyzer.labels, f, indent=2)
    print(f"[EMBED] Label list saved: {labels_path}")

    # ------------------------------------------------------------------
    # Step 1: Extract raw confidence scores from TRAIN split
    # ------------------------------------------------------------------
    train_df = pd.read_csv(train_csv)
    print(f"\n[EMBED] Extracting raw scores — train ({len(train_df)} segments) …")
    X_train_raw, train_paths, train_labels, train_bin = _build_raw_matrix(
        analyzer, Recording, train_df, "train"
    )
    print(f"[EMBED] Train raw matrix: {X_train_raw.shape}")

    if X_train_raw.shape[0] == 0:
        print("[EMBED] ERROR: no training samples extracted. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Fit PCA on training data only
    # ------------------------------------------------------------------
    from sklearn.decomposition import PCA

    actual_components = min(n_components, X_train_raw.shape[0] - 1, X_train_raw.shape[1])
    if actual_components < n_components:
        print(f"[EMBED] NOTE: n_components capped at {actual_components} "
              f"(requested {n_components}).")

    print(f"\n[EMBED] Fitting PCA({actual_components}) on training data …")
    pca = PCA(n_components=actual_components, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train_raw)

    explained = pca.explained_variance_ratio_.sum()
    print(f"[EMBED] PCA done. Explained variance: {explained:.2%} "
          f"with {actual_components} components.")

    pca_path = SAVED_DIR / "pca_transform.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"[EMBED] PCA model saved: {pca_path}")

    # ------------------------------------------------------------------
    # Step 3: Extract raw scores from VAL split
    # ------------------------------------------------------------------
    val_df = pd.read_csv(val_csv)
    print(f"\n[EMBED] Extracting raw scores — val ({len(val_df)} segments) …")
    X_val_raw, val_paths, val_labels, val_bin = _build_raw_matrix(
        analyzer, Recording, val_df, "val"
    )
    print(f"[EMBED] Val raw matrix: {X_val_raw.shape}")

    # ------------------------------------------------------------------
    # Step 4: Transform val with fitted PCA (NO re-fitting)
    # ------------------------------------------------------------------
    X_val_pca = pca.transform(X_val_raw)

    # ------------------------------------------------------------------
    # Step 5: Save embeddings + write manifest
    # ------------------------------------------------------------------
    print("\n[EMBED] Saving embeddings …")
    manifest_rows = []
    manifest_rows.extend(
        _save_embeddings(X_train_pca, train_paths, train_labels, train_bin, "train")
    )
    manifest_rows.extend(
        _save_embeddings(X_val_pca, val_paths, val_labels, val_bin, "val")
    )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST, index=False)

    print(f"\n[EMBED] Manifest: {MANIFEST} ({len(manifest_df)} entries)")
    for split in manifest_df["split"].unique():
        sub = manifest_df[manifest_df["split"] == split]
        n_bird  = (sub["binary_label"] == 1).sum()
        n_noise = (sub["binary_label"] == 0).sum()
        print(f"  {split}: {n_bird} bird + {n_noise} noise = {len(sub)}")

    return manifest_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract BirdNET confidence-score embeddings with PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["splits/train.csv", "splits/val.csv"],
        help="Train and val CSV paths.",
    )
    parser.add_argument(
        "--n-components", type=int, default=1024,
        help="PCA output dimensionality.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if manifest.csv already exists.",
    )
    args = parser.parse_args()
    extract_embeddings(args.splits, n_components=args.n_components, force=args.force)


if __name__ == "__main__":
    main()
