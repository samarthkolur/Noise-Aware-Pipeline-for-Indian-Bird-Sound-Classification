#!/usr/bin/env python3
"""
compute_baseline_metrics.py
Compare BirdNET baseline vs Noise-Aware Pipeline.

Default mode is a FAIR comparison on the same held-out TEST SPLIT
(seed=42, test_frac=0.10, stratified). Optional --full-dataset mode
evaluates both systems on every row in manifest.csv.

Baseline predictions:  comparison/baseline_normalized.jsonl (confidence @0.5)
Pipeline metrics:      recomputed from trained classifier on TEST SPLIT ONLY
                       at threshold=0.5 (same as baseline)
Ground truth:          data/embeddings/manifest.csv (noise/ → 0, species → 1)

IMPORTANT: Test-split mode is the fair generalization comparison.
Full-dataset mode is useful for manifest-wide operational comparison but
includes training rows for the pipeline.
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None
import numpy as np

from utils.metrics import confusion_binary, metrics_from_confusion


# ── Key building helpers (must match load_ground_truth logic) ──────────────


def _norm_key_part(s: str) -> str:
    """Canonical string form for key tuple elements (case- and whitespace-insensitive)."""
    return str(s).strip().lower()


def _jsonl_segment_index(d: dict, *, seg_from_time: int, name: str) -> int:
    """Prefer explicit ``segment_index`` (filtered JSONL), else ``_segNNNN`` in filename, else time."""
    raw = d.get("segment_index", None)
    if raw is not None and str(raw).strip() != "" and str(raw).strip().lower() != "none":
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            pass
    if "_seg" in name:
        suf = name.split("_seg", 1)[1]
        seg_part = suf.split(".", 1)[0]
        try:
            return int(seg_part)
        except ValueError:
            pass
    return int(seg_from_time)


def _build_key_from_manifest_row(row: dict) -> Tuple[str, str, int]:
    """Build canonical (a, b, seg_idx) aligned with ``_build_key_from_baseline_jsonl``.

    Tuple entries are lowercased path tokens so JSONL paths on Windows/Linux still join.

    Noise segments use decoded (orig_species, orig_filenum, seg_idx) from
    ``.../noise/Orig__File_segNNNN.wav`` (``seg_idx`` comes from manifest column).
    """
    species = row["species"]
    src_file = str(row["source_file"]).replace("\\", "/")
    seg_idx = int(row["segment_index"])
    label_noise = str(species).strip().lower() == "noise"

    p = Path(src_file)
    name = p.name
    parent = _norm_key_part(p.parent.name)
    file_stem_raw = name.split("_seg")[0]

    if label_noise and "__" in file_stem_raw:
        orig_species, orig_filenum = file_stem_raw.rsplit("__", 1)
        return (_norm_key_part(orig_species), _norm_key_part(orig_filenum), seg_idx)
    return (parent, _norm_key_part(file_stem_raw), seg_idx)


def _build_key_from_baseline_jsonl(d: dict) -> Tuple[str, str, int]:
    """Same canonical triple as ``_build_key_from_manifest_row`` for the same segment."""
    src = str(d["source_file"]).replace("\\", "/")
    p = Path(src)
    name = p.name
    stem_full = p.stem
    parent = _norm_key_part(p.parent.name)
    start_sec = float(d.get("start_sec", 0.0))
    seg_from_time = int(start_sec // 3)

    # Routed noise clip: .../noise/Orig__File_segNNNN.wav
    if parent == "noise" and "__" in stem_full:
        left, _, right = stem_full.rpartition("__")
        left_n = _norm_key_part(left)
        if "_seg" in right:
            filenum, _, seg_s = right.rpartition("_seg")
            try:
                seg_fname = int(seg_s)
            except ValueError:
                seg_fname = seg_from_time
            raw = d.get("segment_index", None)
            if raw is not None and str(raw).strip() != "" and str(raw).strip().lower() != "none":
                try:
                    seg = int(float(raw))
                except (TypeError, ValueError):
                    seg = seg_fname
            else:
                seg = seg_fname
            return (left_n, _norm_key_part(filenum), seg)
        seg = _jsonl_segment_index(d, seg_from_time=seg_from_time, name=name)
        return (left_n, _norm_key_part(right), seg)

    if "_seg" in name:
        file_stem = _norm_key_part(name.split("_seg")[0])
        seg = _jsonl_segment_index(d, seg_from_time=seg_from_time, name=name)
        return (parent, file_stem, seg)

    seg = _jsonl_segment_index(d, seg_from_time=seg_from_time, name=name)
    return (parent, _norm_key_part(stem_full), seg)


def _get_ordered_manifest_rows(manifest_path: str) -> List[dict]:
    """Return manifest rows in the SAME order as EmbeddingDataset.from_manifest().

    EmbeddingDataset groups rows by hdf5_path (sorted alphabetically), then
    iterates each group in manifest-insertion order.  Replicating this mapping
    lets us go from dataset index i → manifest row without touching PyTorch.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by hdf5_path, preserving insertion order within each group
    h5_groups: Dict[str, List[dict]] = {}
    for row in rows:
        h5_groups.setdefault(row["hdf5_path"], []).append(row)

    ordered: List[dict] = []
    for h5_path in sorted(h5_groups.keys()):
        ordered.extend(h5_groups[h5_path])

    assert len(ordered) == len(rows), "Row count mismatch after grouping"
    return ordered


# ── Ground truth loaders ───────────────────────────────────────────────────

def load_ground_truth(manifest_path: str) -> dict:
    """Load ALL binary labels from manifest (used to verify counts only)."""
    gt = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        key = _build_key_from_manifest_row(row)
        label = 0 if row["species"].lower() == "noise" else 1
        gt[key] = label
    return gt


def build_test_split_ground_truth(manifest_path: str, cfg: dict) -> Tuple[dict, List]:
    """Return (gt_dict, test_manifest_rows) for the test split only.

    Uses the same create_splits(seed, val_frac, test_frac, stratify) call as
    trainer.py and evaluate.py so the split is identical.
    """
    from dataset.dataset import EmbeddingDataset, create_splits

    manifest_path = Path(manifest_path)
    dataset = EmbeddingDataset.from_manifest(manifest_path, binary=True)

    ds_cfg = cfg.get("dataset", {})
    splits = create_splits(
        dataset,
        val_frac=ds_cfg.get("val_split", 0.15),
        test_frac=ds_cfg.get("test_split", 0.10),
        stratify=ds_cfg.get("stratify", True),
        seed=cfg.get("project", {}).get("seed", 42),
    )

    ordered_rows = _get_ordered_manifest_rows(str(manifest_path))
    assert len(ordered_rows) == len(dataset), (
        f"Manifest rows ({len(ordered_rows)}) != dataset length ({len(dataset)}). "
        "Run embedding extraction again to regenerate manifest."
    )

    test_gt: dict = {}
    test_rows: List[dict] = []
    for i in splits.test_idx:
        row = ordered_rows[i]
        key = _build_key_from_manifest_row(row)
        label = 0 if row["species"].lower() == "noise" else 1
        test_gt[key] = label
        test_rows.append(row)

    return test_gt, test_rows


# ── Baseline predictions loader ────────────────────────────────────────────

def load_baseline_predictions(jsonl_path: str) -> dict:
    """Load BirdNET detections, keyed by (folder, file_stem, seg_idx).

    Each JSON object should include ``confidence`` and a path field:
    ``source_file`` (preferred, written by ``normalize_birdnet_export``) or
    ``input_path`` (CSV ``File`` column) as fallback.
    """
    predictions: dict = {}
    n_lines = 0
    n_bad_json = 0
    n_skipped = 0
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue
            src = d.get("source_file") or d.get("input_path")
            if not src:
                n_skipped += 1
                continue
            d = {**d, "source_file": str(src)}
            try:
                key = _build_key_from_baseline_jsonl(d)
                conf = float(d.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                n_skipped += 1
                continue
            predictions[key] = max(predictions.get(key, 0.0), conf)
    if predictions:
        load_baseline_predictions._last_empty_stats = None  # type: ignore[attr-defined]
    else:
        load_baseline_predictions._last_empty_stats = (  # type: ignore[attr-defined]
            n_lines,
            n_bad_json,
            n_skipped,
        )
    return predictions


# ── Confusion matrix / metrics (canonical: ``utils.metrics``) ───────────────


def compute_confusion(y_true, y_pred):
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    tp, tn, fp, fn = confusion_binary(yt, yp)
    return tp, tn, fp, fn


# ── Pipeline metrics on test split ────────────────────────────────────────

def compute_pipeline_metrics(
    config_path: str = "config.yaml",
    threshold: float = 0.5,
    full_dataset: bool = False,
) -> Tuple[dict, int, int, int]:
    """Evaluate trained MLP on test split (default) or full manifest.

    In test-split mode this uses held-out test indices (same split as training).
    In full-dataset mode this uses all manifest rows.
    """
    import torch
    from dataset.dataset import EmbeddingDataset, create_splits
    from models.classifier import EmbeddingClassifier
    from torch.utils.data import DataLoader, Subset

    cfg_mod = __import__("utils.config", fromlist=["load_config"])
    cfg = cfg_mod.load_config(config_path)
    device = torch.device("cpu")

    manifest = Path(cfg["data"]["embeddings_dir"]) / "manifest.csv"
    ds = EmbeddingDataset.from_manifest(manifest, binary=True)

    ds_cfg = cfg.get("dataset", {})
    splits = create_splits(
        ds,
        val_frac=ds_cfg.get("val_split", 0.15),
        test_frac=ds_cfg.get("test_split", 0.10),
        stratify=ds_cfg.get("stratify", True),
        seed=cfg.get("project", {}).get("seed", 42),
    )

    if full_dataset:
        loader = DataLoader(ds, batch_size=64, shuffle=False)
    else:
        test_subset = Subset(ds, splits.test_idx)
        loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    model = EmbeddingClassifier(
        input_dim=cfg["embedding"]["embedding_dim"],
        num_classes=1,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
    ).to(device)
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    chk = torch.load(
        chkpt_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for embs, labels in loader:
            logits = model(embs.to(device))
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    probs = torch.sigmoid(logits_t).numpy()
    y = labels_t.numpy()

    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    result = metrics_from_confusion(tp, tn, fp, fn)
    n_noise = int((y == 0).sum())
    n_bird  = int((y == 1).sum())
    n_total = int(len(y))
    return result, n_noise, n_bird, n_total


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_metrics_comparison(baseline: dict, pipeline: dict, out_path: str,
                            n_test: int = 0):
    if plt is None:
        print("  [skip plots] matplotlib not installed")
        return
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    b_vals = [baseline[k] for k in ("accuracy", "precision", "recall", "f1")]
    p_vals = [pipeline[k] for k in ("accuracy", "precision", "recall", "f1")]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline (Raw BirdNET @0.5)", color="#4C72B0")
    bars_p = ax.bar(x + w / 2, p_vals, w, label="Noise-Aware Pipeline @0.5",   color="#55A868")

    ax.set_ylabel("Score", fontsize=12)
    title = "Performance: Raw BirdNET vs Noise-Aware Pipeline"
    if n_test:
        title += f"\n(Same test split, N={n_test})"
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)

    for bars in (bars_b, bars_p):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_error_comparison(baseline: dict, pipeline: dict, out_path: str,
                          n_test: int = 0):
    if plt is None:
        print("  [skip plots] matplotlib not installed")
        return
    labels = ["False Positive Rate\n(Noise → Bird)", "False Negative Rate\n(Bird Missed)"]
    b_vals = [baseline.get("fpr", 0), baseline.get("fnr", 0)]
    p_vals = [pipeline.get("fpr", 0), pipeline.get("fnr", 0)]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline (BirdNET)", color="#C44E52")
    bars_p = ax.bar(x + w / 2, p_vals, w, label="Noise-Aware Pipeline", color="#8172B3")

    ax.set_ylabel("Rate", fontsize=12)
    title = "Error Rate Comparison"
    if n_test:
        title += f" (N={n_test})"
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(b_vals), max(p_vals)) * 1.25 + 0.05)

    for bars in (bars_b, bars_p):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BirdNET baseline vs Pipeline — fair test-split comparison"
    )
    parser.add_argument("--manifest",   default="data/embeddings/manifest.csv")
    parser.add_argument("--baseline",   default="comparison/baseline_normalized.jsonl")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Confidence threshold for BirdNET AND MLP (default: 0.5)")
    parser.add_argument("--config",     type=str,   default="config.yaml")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Compare on all manifest rows instead of only the held-out test split.",
    )
    parser.add_argument(
        "--debug-keys",
        action="store_true",
        help="Print sample manifest vs baseline keys (Step 2) for alignment diagnosis.",
    )
    parser.add_argument(
        "--min-key-match-rate",
        type=float,
        default=0.7,
        help="Abort if matched_keys/test_keys is below this rate (default 0.7). Set to 0 to disable.",
    )
    args = parser.parse_args()

    print("=" * 65)
    if args.full_dataset:
        print("  BASELINE vs PIPELINE — FULL-MANIFEST COMPARISON")
    else:
        print("  BASELINE vs PIPELINE — FAIR TEST-SPLIT COMPARISON")
    print("=" * 65)
    if args.full_dataset:
        print("  Both systems evaluated on the SAME full manifest")
    else:
        print("  Both systems evaluated on the SAME held-out test split")
        print("  (seed=42, test_frac=0.10, stratified — same split as trainer.py)")
    print(f"  Both thresholded at {args.threshold:.2f}  (apples-to-apples)")

    # ── 1. Load config ──────────────────────────────────────────────
    from utils.config import load_config
    cfg = load_config(args.config)

    # ── 2. Build ground truth from manifest ──────────────────────────
    if args.full_dataset:
        print(f"\n--- Step 1: Build Full-Manifest Ground Truth ---")
        eval_gt = load_ground_truth(args.manifest)
    else:
        print(f"\n--- Step 1: Build Test-Split Ground Truth ---")
        eval_gt, _ = build_test_split_ground_truth(args.manifest, cfg)

    gt_labels = Counter(eval_gt.values())
    n_bird_eval  = gt_labels[1]
    n_noise_eval = gt_labels[0]
    n_total_eval = len(eval_gt)
    print(f"  Eval set size:      {n_total_eval}")
    print(f"  Bird  (y==1):       {n_bird_eval}")
    print(f"  Noise (y==0):       {n_noise_eval}")
    if n_noise_eval == 0:
        print("  STOP: No noise samples in test split — FPR cannot be computed.")
        return

    # ── 3. Load baseline predictions ─────────────────────────────────
    print(f"\n--- Step 2: Load Baseline Predictions ---")
    bl_path = Path(args.baseline)
    if not bl_path.is_file():
        raise SystemExit(
            f"Baseline JSONL not found: {bl_path.resolve()}. "
            "Generate it with:\n"
            "  python birdnet_integration/normalize_birdnet_export.py "
            "--input-dir <dir_with_*.BirdNET.results.csv> --run baseline"
        )
    bl = load_baseline_predictions(str(bl_path))
    if not bl:
        stats = getattr(load_baseline_predictions, "_last_empty_stats", None)
        extra = ""
        if stats is not None:
            nl, nb, ns = stats
            extra = f" ({nl} non-empty lines, {nb} bad JSON, {ns} skipped.)"
        raise SystemExit(
            f"No predictions loaded from {bl_path.resolve()} (0 keys){extra} "
            "File may be empty or records lack source_file/input_path + confidence. "
            f"Inspect: head -n 2 {bl_path}\n"
            "Regenerate: python birdnet_integration/normalize_birdnet_export.py "
            "--input-dir <dir_with_*.BirdNET.results.csv> --run baseline"
        )

    test_keys   = set(eval_gt.keys())
    bl_keys     = set(bl.keys())
    matched     = test_keys & bl_keys
    noise_match = sum(1 for k in matched if eval_gt[k] == 0)
    print(f"  Test keys:          {len(test_keys)}")
    print(f"  Baseline keys:      {len(bl_keys)}")
    print(f"  Matched:            {len(matched)}")
    print(f"  GT-only (no detect):{len(test_keys - bl_keys)}  → treated as conf=0 → pred=0")
    print(f"  Noise segs matched: {noise_match}/{n_noise_eval}")

    if args.debug_keys:
        sample_manifest_keys = sorted(test_keys)[:3]
        sample_bl_keys = sorted(bl_keys)[:3]
        print("SAMPLE MANIFEST KEYS:", sample_manifest_keys)
        print("SAMPLE BASELINE KEYS:", sample_bl_keys)

    match_rate = len(matched) / max(len(test_keys), 1)
    min_rate = float(args.min_key_match_rate)
    if min_rate > 0 and match_rate < min_rate:
        raise SystemExit(
            f"Key match rate too low: {match_rate:.2%} ({len(matched)}/{len(test_keys)}). "
            "Check _build_key_from_manifest_row vs _build_key_from_baseline_jsonl alignment "
            f"(re-run with --debug-keys). Minimum required: {min_rate:.0%}."
        )

    # ── 4. Baseline confusion on test split ──────────────────────────
    y_true, y_pred_base = [], []
    for key, label in eval_gt.items():
        conf = bl.get(key, 0.0)
        pred = 1 if conf >= args.threshold else 0
        y_true.append(label)
        y_pred_base.append(pred)

    pred_dist = Counter(y_pred_base)
    print(f"\n--- Step 3: Baseline Predictions on Test Split ---")
    print(f"  y_pred distribution: {{0: {pred_dist[0]}, 1: {pred_dist[1]}}}")
    if 1 not in pred_dist:
        print("  WARNING: baseline predicts 0 for every test sample — "
              "check that baseline JSONL keys match manifest structure.")

    tp, tn, fp, fn = compute_confusion(y_true, y_pred_base)
    base = metrics_from_confusion(tp, tn, fp, fn)

    print(f"\n--- Baseline Confusion (threshold={args.threshold}) ---")
    print(f"  [[TN={tn:4d}, FP={fp:4d}],")
    print(f"   [FN={fn:4d}, TP={tp:4d}]]   total={tp+tn+fp+fn}")
    print(f"  Accuracy:  {base['accuracy']:.4f}")
    print(f"  Precision: {base['precision']:.4f}")
    print(f"  Recall:    {base['recall']:.4f}")
    print(f"  F1:        {base['f1']:.4f}")
    print(f"  FPR:       {base['fpr']:.4f}  = {fp}/({fp}+{tn})")
    print(f"  FNR:       {base['fnr']:.4f}  = {fn}/({fn}+{tp})")

    # ── 5. Pipeline on test split ─────────────────────────────────────
    if args.full_dataset:
        print(f"\n--- Step 4: Pipeline Metrics on Full Manifest (threshold={args.threshold}) ---")
    else:
        print(f"\n--- Step 4: Pipeline Metrics on Test Split (threshold={args.threshold}) ---")
    pipe, pipe_n_noise, pipe_n_bird, pipe_n_total = compute_pipeline_metrics(
        config_path=args.config,
        threshold=args.threshold,
        full_dataset=args.full_dataset,
    )
    pipe_tp, pipe_tn = pipe["tp"], pipe["tn"]
    pipe_fp, pipe_fn = pipe["fp"], pipe["fn"]

    print(f"  Test set: {pipe_n_bird} bird + {pipe_n_noise} noise (total={pipe_n_total})")
    print(f"  [[TN={pipe_tn:4d}, FP={pipe_fp:4d}],")
    print(f"   [FN={pipe_fn:4d}, TP={pipe_tp:4d}]]")
    print(f"  Accuracy:  {pipe['accuracy']:.4f}")
    print(f"  Precision: {pipe['precision']:.4f}")
    print(f"  Recall:    {pipe['recall']:.4f}")
    print(f"  F1:        {pipe['f1']:.4f}")
    print(f"  FPR:       {pipe['fpr']:.4f}  = {pipe_fp}/({pipe_fp}+{pipe_tn})")
    print(f"  FNR:       {pipe['fnr']:.4f}  = {pipe_fn}/({pipe_fn}+{pipe_tp})")

    # ── 6. Sanity check N totals ──────────────────────────────────────
    print(f"\n--- Step 5: Sanity Check ---")
    base_total = tp + tn + fp + fn
    print(f"  Baseline  total N: {base_total}")
    print(f"  Pipeline  total N: {pipe_n_total}")
    if base_total == pipe_n_total:
        print(f"  ✓ Same N — comparison is valid (apples to apples)")
    else:
        print(f"  ✗ Different N ({base_total} vs {pipe_n_total}) — "
              f"key mismatch; check manifest ↔ JSONL alignment")

    if base["fpr"] > pipe["fpr"]:
        print(f"  ✓ FPR:  Baseline {base['fpr']:.4f} > Pipeline {pipe['fpr']:.4f} "
              f"(pipeline rejects noise better)")
    else:
        print(f"  ⚠ FPR: Baseline {base['fpr']:.4f} ≤ Pipeline {pipe['fpr']:.4f}")

    if base["f1"] < pipe["f1"]:
        print(f"  ✓ F1:   Baseline {base['f1']:.4f} < Pipeline {pipe['f1']:.4f} "
              f"(pipeline better overall)")

    # ── 7. Save JSON outputs ──────────────────────────────────────────
    Path("comparison").mkdir(exist_ok=True)
    Path("results/comparison_graphs").mkdir(parents=True, exist_ok=True)

    base_out = {k: base[k] for k in
                ("accuracy", "precision", "recall", "f1", "fpr", "fnr",
                 "tp", "tn", "fp", "fn")}
    base_out["eval_set"] = "full_manifest" if args.full_dataset else "test_split"
    base_out["threshold"] = args.threshold
    base_out["n_total"] = base_total
    base_out["n_bird"] = n_bird_eval
    base_out["n_noise"] = n_noise_eval

    pipe_out = {k: pipe[k] for k in
                ("accuracy", "precision", "recall", "f1", "fpr", "fnr",
                 "tp", "tn", "fp", "fn")}
    pipe_out["eval_set"] = "full_manifest" if args.full_dataset else "test_split"
    pipe_out["threshold"] = args.threshold
    pipe_out["n_total"] = pipe_n_total
    pipe_out["n_bird"] = pipe_n_bird
    pipe_out["n_noise"] = pipe_n_noise

    with open("comparison/baseline_metrics.json", "w") as f:
        json.dump(base_out, f, indent=2)
    with open("comparison/pipeline_metrics.json", "w") as f:
        json.dump(pipe_out, f, indent=2)

    mode_note = (
        f"Both evaluated on held-out test split (seed=42, test_frac=0.10, stratified). "
        if not args.full_dataset
        else "Both evaluated on full manifest rows. "
    )
    comp = {
        "note": (
            f"{mode_note}N={base_total}. Threshold={args.threshold} for both."
        ),
        "Baseline (BirdNET)":       base_out,
        "Pipeline (Noise-Aware)":   pipe_out,
    }
    with open("comparison/comparison_table.json", "w") as f:
        json.dump(comp, f, indent=2)

    print(f"\n--- Saved ---")
    print(f"  comparison/baseline_metrics.json")
    print(f"  comparison/pipeline_metrics.json")
    print(f"  comparison/comparison_table.json")

    # ── 8. Plots ──────────────────────────────────────────────────────
    print(f"\n--- Step 6: Plots ---")
    plot_metrics_comparison(
        base_out, pipe_out,
        "results/comparison_graphs/metrics_comparison.png",
        n_test=base_total,
    )
    plot_error_comparison(
        base_out, pipe_out,
        "results/comparison_graphs/error_comparison.png",
        n_test=base_total,
    )

    print(f"\n{'=' * 65}")
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
