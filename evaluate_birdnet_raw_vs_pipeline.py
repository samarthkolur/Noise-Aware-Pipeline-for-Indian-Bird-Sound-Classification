#!/usr/bin/env python3
"""
STRICT evaluation: real BirdNET (birdnetlib) inference on:

  (A) data/processed/**  — GT: parent folder ``noise`` → 0, species folders → 1
  (B) outputs/clean_birds/** + outputs/noise/** — GT: clean_birds → 1, noise → 0

Each WAV is analyzed with Analyzer() + Recording(...).analyze() — no JSONL cache,
no placeholder metrics.

Prerequisites:
  pip install birdnetlib ai-edge-litert
  Preprocess populated ``data/processed/`` (including noise WAVs for GT=0)
  Infer populated ``outputs/clean_birds/`` and ``outputs/noise/``

Usage (from repo root):
  python evaluate_birdnet_raw_vs_pipeline.py --config config.yaml --threshold 0.5

Outputs:
  results/real_birdnet_eval/metrics_comparison.png
  results/real_birdnet_eval/error_comparison.png
  results/real_birdnet_eval/real_birdnet_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("tensorflow").setLevel(logging.ERROR)

_MIN_DET_CONF = 0.01

_analyzer = None
_Recording = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_config(path: str) -> dict:
    from utils.config import load_config

    return load_config(path)


def init_worker() -> None:
    global _analyzer, _Recording
    from birdnetlib.analyzer import Analyzer
    from birdnetlib.main import Recording

    _analyzer = Analyzer()
    _Recording = Recording
    print(
        f"[worker] BirdNET loaded: {type(_analyzer).__module__}.{type(_analyzer).__name__}",
        flush=True,
    )


def birdnet_infer_one(path_str: str) -> Dict[str, Any]:
    assert _analyzer is not None and _Recording is not None
    rec = _Recording(_analyzer, str(path_str), min_conf=_MIN_DET_CONF)
    rec.analyze()

    dets = list(rec.detections or [])
    max_conf = 0.0
    best_species = "(no detection)"

    for d in dets:
        try:
            c = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            c = 0.0
        if c > max_conf:
            max_conf = c
            best_species = (
                d.get("common_name")
                or d.get("scientific_name")
                or d.get("species")
                or "unknown"
            )

    return {
        "path": str(Path(path_str).resolve()),
        "max_confidence": float(max_conf),
        "best_species": str(best_species),
        "n_detections": int(len(dets)),
    }


def process_task(task: Tuple[str, int, float]) -> Dict[str, Any]:
    path_str, y_true, thr = task
    try:
        out = birdnet_infer_one(path_str)
        pred = 1 if out["max_confidence"] >= thr else 0
        out.update(
            {
                "y_true": int(y_true),
                "y_pred": int(pred),
                "threshold": float(thr),
                "error": None,
            }
        )
        return out
    except Exception as e:
        return {
            "path": path_str,
            "y_true": int(y_true),
            "y_pred": -1,
            "max_confidence": 0.0,
            "best_species": "",
            "n_detections": 0,
            "threshold": float(thr),
            "error": str(e),
        }


def collect_processed(processed_dir: Path) -> List[Tuple[Path, int]]:
    rows: List[Tuple[Path, int]] = []
    if not processed_dir.is_dir():
        return rows
    for d in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        y = 0 if d.name.lower() == "noise" else 1
        for wav in sorted(d.glob("*.wav")):
            rows.append((wav, y))
    return rows


def collect_outputs(out_root: Path) -> List[Tuple[Path, int]]:
    rows: List[Tuple[Path, int]] = []
    clean = out_root / "clean_birds"
    noise = out_root / "noise"
    if clean.is_dir():
        for wav in sorted(clean.rglob("*.wav")):
            rows.append((wav, 1))
    if noise.is_dir():
        for wav in sorted(noise.rglob("*.wav")):
            rows.append((wav, 0))
    return rows


def confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    return tn, fp, fn, tp


def metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    tot = tn + fp + fn + tp
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def run_pool(name: str, tasks: List[Tuple[str, int, float]], workers: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    good = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as ex:
        futs = {ex.submit(process_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            out.append(r)
            done += 1
            if r.get("error"):
                print(f"[{name}] ERROR {r['path']}: {r['error']}", flush=True)
            else:
                good += 1
                if good <= 3:
                    print(
                        f"[{name}] SAMPLE | file={Path(r['path']).name} | "
                        f"species={r['best_species']} | conf={r['max_confidence']:.4f} | "
                        f"y_true={r['y_true']} | y_pred={r['y_pred']}",
                        flush=True,
                    )
            if done % 500 == 0:
                print(f"[{name}] {done}/{len(tasks)}", flush=True)
    return out


def plot_pair(ma: Dict[str, float], mb: Dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, [ma[k] for k in keys], w, label="Raw processed + BirdNET", color="#4C72B0")
    ax.bar(x + w / 2, [mb[k] for k in keys], w, label="Pipeline outputs + BirdNET", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Metrics (BirdNET binary: pred=1 if max conf ≥ threshold)")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar([0 - w / 2, 1 - w / 2], [ma["fpr"], ma["fnr"]], w, label="Raw + BirdNET", color="#C44E52")
    ax2.bar([0 + w / 2, 1 + w / 2], [mb["fpr"], mb["fnr"]], w, label="Pipeline + BirdNET", color="#8172B3")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["FPR\n(noise→bird)", "FNR\n(bird missed)"])
    ax2.legend()
    ax2.set_ylabel("Rate")
    fig2.tight_layout()
    fig2.savefig(out_dir / "error_comparison.png", dpi=150)
    plt.close(fig2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real BirdNET on raw processed WAVs vs pipeline-routed WAVs."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary: pred=bird if max(detection confidence) >= threshold",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Max files per experiment (debug)")
    args = ap.parse_args()

    root = _repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    cfg = _load_config(str(cfg_path))

    processed_dir = (root / cfg["data"]["processed_dir"]).resolve()
    output_dir = (root / cfg["data"]["output_dir"]).resolve()

    print("=" * 72)
    print("PROOF: BirdNET via birdnetlib.analyzer.Analyzer() (per worker)")
    print("PROOF: birdnetlib.main.Recording(analyzer, wav, min_conf=0.01).analyze()")
    print(f"Binary threshold on max confidence: {args.threshold}")
    print("=" * 72)

    try:
        import birdnetlib  # noqa: F401
    except ImportError:
        print("FAIL: pip install birdnetlib ai-edge-litert", flush=True)
        return 1

    raw_rows = collect_processed(processed_dir)
    if args.limit:
        raw_rows = raw_rows[: args.limit]

    n_noise = sum(1 for _, y in raw_rows if y == 0)
    n_bird = sum(1 for _, y in raw_rows if y == 1)
    print("\n[A] RAW processed segments", flush=True)
    print(f"  dir={processed_dir}", flush=True)
    print(f"  total={len(raw_rows)} bird_gt={n_bird} noise_gt={n_noise}", flush=True)
    if n_noise == 0:
        print("FAIL: no noise WAVs under processed/noise/ — cannot compute FPR meaningfully.", flush=True)
        return 2

    tasks_a = [(str(p), y, args.threshold) for p, y in raw_rows]
    res_a = run_pool("RAW", tasks_a, args.workers)

    yta, ypa = [], []
    for r in res_a:
        if r.get("error") or r["y_pred"] < 0:
            continue
        yta.append(r["y_true"])
        ypa.append(r["y_pred"])
    print(f"  unique y_pred (A): {sorted(set(ypa))}", flush=True)

    tna, fpa, fna, tpa = confusion(yta, ypa)
    ma = metrics(tna, fpa, fna, tpa)
    print(f"  TN={tna} FP={fpa} FN={fna} TP={tpa}", flush=True)
    print(f"  FPR={ma['fpr']:.4f}  FNR={ma['fnr']:.4f}", flush=True)

    pipe_rows = collect_outputs(output_dir)
    if args.limit:
        pipe_rows = pipe_rows[: args.limit]

    n_noise_b = sum(1 for _, y in pipe_rows if y == 0)
    n_bird_b = sum(1 for _, y in pipe_rows if y == 1)
    print("\n[B] PIPELINE inference outputs", flush=True)
    print(f"  dir={output_dir}", flush=True)
    print(f"  total={len(pipe_rows)} bird_gt={n_bird_b} noise_gt={n_noise_b}", flush=True)
    if not pipe_rows:
        print("FAIL: no WAVs under clean_birds/ or noise/. Run: python run_pipeline.py --stage infer", flush=True)
        return 3

    tasks_b = [(str(p), y, args.threshold) for p, y in pipe_rows]
    res_b = run_pool("PIPE", tasks_b, args.workers)

    ytb, ypb = [], []
    for r in res_b:
        if r.get("error") or r["y_pred"] < 0:
            continue
        ytb.append(r["y_true"])
        ypb.append(r["y_pred"])
    print(f"  unique y_pred (B): {sorted(set(ypb))}", flush=True)

    tnb, fpb, fnb, tpb = confusion(ytb, ypb)
    mb = metrics(tnb, fpb, fnb, tpb)
    print(f"  TN={tnb} FP={fpb} FN={fnb} TP={tpb}", flush=True)
    print(f"  FPR={mb['fpr']:.4f}  FNR={mb['fnr']:.4f}", flush=True)

    fp_red: Optional[float] = (100.0 * (fpa - fpb) / fpa) if fpa > 0 else None

    print("\n--- Sanity (FP counts are not directly comparable across different sample sets) ---", flush=True)
    print(f"  FP (A, raw noise segments): {fpa}", flush=True)
    print(f"  FP (B, pipeline noise folder): {fpb}", flush=True)
    if fpa > 0 and fpb < fpa:
        print("  Note: fewer FP detections on B's noise set vs A's noise set (count-wise).", flush=True)
    elif fpa > 0:
        print(
            "  Note: B uses a different file set (MLP-filtered); FP_b can exceed FP_a.",
            flush=True,
        )

    out_dir = root / "results" / "real_birdnet_eval"
    plot_pair(ma, mb, out_dir)

    payload = {
        "threshold": args.threshold,
        "experiment_a_raw_processed": {
            "dir": str(processed_dir),
            "n_files": len(raw_rows),
            "confusion": {"tn": tna, "fp": fpa, "fn": fna, "tp": tpa},
            "metrics": ma,
            "results": res_a,
        },
        "experiment_b_pipeline_outputs": {
            "dir": str(output_dir),
            "n_files": len(pipe_rows),
            "confusion": {"tn": tnb, "fp": fpb, "fn": fnb, "tp": tpb},
            "metrics": mb,
            "results": res_b,
        },
        "fp_reduction_pct_count_proxy": fp_red,
    }
    json_path = out_dir / "real_birdnet_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print("METRICS (BirdNET binary thresholding)")
    print("=" * 72)
    print(f"{'Metric':<14} {'Raw + BirdNET':>16} {'Pipeline + BirdNET':>22}")
    for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"):
        print(f"{k:<14} {ma[k]:>16.4f} {mb[k]:>22.4f}")
    if fp_red is not None:
        print(f"\nFP count reduction proxy (A→B): {fp_red:.2f}%")

    print(f"\nSaved: {json_path}", flush=True)
    print(f"Saved: {out_dir / 'metrics_comparison.png'}", flush=True)
    print(f"Saved: {out_dir / 'error_comparison.png'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
