"""Top-K error mining, heuristics, and audio copies."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .audio_heuristics import tag_segment


def resolve_audio_path(row: dict, project_root: Path) -> Path:
    """Resolve manifest source_file to a local WAV path."""
    raw = row.get("source_file", "")
    p = Path(raw)
    if p.is_file():
        return p
    cand = project_root / raw
    if cand.is_file():
        return cand
    return p


def top_k_indices(
    mask: np.ndarray,
    scores: np.ndarray,
    k: int,
    descending: bool = True,
) -> np.ndarray:
    """Indices where mask True, ranked by scores."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([], dtype=np.int64)
    s = scores[idx]
    order = np.argsort(-s if descending else s)
    return idx[order[:k]]


def mine_and_copy(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_mlp: np.ndarray,
    pred_ae: np.ndarray,
    baseline_probs: np.ndarray,
    mlp_probs: np.ndarray,
    recon_err: np.ndarray,
    ood_mask: np.ndarray,
    rows: List[dict],
    project_root: Path,
    out_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Select top-K FP/FN/OOD/recovered; copy WAVs; run heuristics."""
    out_dir = Path(out_dir)
    subdirs = ("false_positives", "false_negatives", "recovered_by_pipeline", "rejected_by_autoencoder")
    for s in subdirs:
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    # FP (noise → bird) for AE+MLP system at 0.5
    fp_mask = (y_true == 0) & (pred_ae == 1)
    fp_scores = mlp_probs  # high conf false bird
    fp_idx = top_k_indices(fp_mask, fp_scores, top_k, descending=True)

    # FN (bird → noise): rank OOD birds by high recon err; others by low MLP prob
    fn_mask = (y_true == 1) & (pred_ae == 0)
    fn_rank_scores = np.where(ood_mask & (y_true == 1), recon_err, 1.0 - mlp_probs)
    fn_idx = top_k_indices(fn_mask, fn_rank_scores, top_k, descending=True)

    # Recovered: baseline wrong, AE+MLP correct
    rec_mask = (pred_base != y_true) & (pred_ae == y_true)
    rec_scores = np.abs(baseline_probs - 0.5)
    rec_idx = top_k_indices(rec_mask, rec_scores, top_k, descending=True)

    # OOD rejected (any true label)
    ood_idx = top_k_indices(ood_mask, recon_err, top_k, descending=True)

    def pack_indices(name: str, indices: np.ndarray) -> List[Dict[str, Any]]:
        items = []
        for j in indices:
            row = rows[int(j)]
            p = resolve_audio_path(row, project_root)
            meta = {
                "index": int(j),
                "source_file": row.get("source_file"),
                "species": row.get("species"),
                "y_true": int(y_true[j]),
                "resolved_path": str(p) if p else None,
            }
            if name == "false_positives":
                meta["mlp_prob"] = float(mlp_probs[j])
            elif name == "false_negatives":
                meta["ood"] = bool(ood_mask[j])
                meta["recon_error"] = float(recon_err[j])
                meta["mlp_prob"] = float(mlp_probs[j])
            elif name == "recovered_by_pipeline":
                meta["baseline_prob"] = float(baseline_probs[j])
            elif name == "rejected_by_autoencoder":
                meta["recon_error"] = float(recon_err[j])
            if p.is_file():
                dest = out_dir / name / f"{name}_{j:05d}_{Path(p).name}"
                try:
                    shutil.copy2(p, dest)
                    meta["copied_to"] = str(dest)
                except OSError as e:
                    meta["copy_error"] = str(e)
            else:
                meta["copy_error"] = "file not found"
            meta["heuristics"] = tag_segment(p) if p.is_file() else {}
            items.append(meta)
        return items

    report = {
        "false_positives_topk": pack_indices("false_positives", fp_idx),
        "false_negatives_topk": pack_indices("false_negatives", fn_idx),
        "recovered_by_pipeline_topk": pack_indices("recovered_by_pipeline", rec_idx),
        "rejected_by_autoencoder_topk": pack_indices("rejected_by_autoencoder", ood_idx),
        "top_k": top_k,
    }
    return report
