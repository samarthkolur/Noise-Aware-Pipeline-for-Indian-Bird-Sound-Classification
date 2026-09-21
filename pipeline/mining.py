"""Hard-example mining: export likely false positives / false negatives for
expert review (design.md §6.5, §20), plus human-in-the-loop corrections
captured from the dashboard's noise-analysis view (DD-022).

  Likely FP: noise-class segments where MLP assigns p > 0.5
  Likely FN: bird-class segments where MLP assigns p < 0.5 OR AE-rejected
"""

from __future__ import annotations

import csv
import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.audio import write_wav


@dataclass
class MiningResult:
    likely_fp: list[str]  # segment_ids
    likely_fn: list[str]


def find_hard_examples(
    segment_ids: list[str],
    true_labels: list[str],  # "bird" | "noise"
    mlp_probs: list[float],
    ood_rejected: list[bool],
) -> MiningResult:
    """Identify likely FP/FN segment_ids given ground truth and pipeline outputs."""
    likely_fp = []
    likely_fn = []
    for seg_id, label, prob, ood in zip(
        segment_ids, true_labels, mlp_probs, ood_rejected, strict=True
    ):
        if label == "noise" and prob > 0.5:
            likely_fp.append(seg_id)
        elif label == "bird" and (prob < 0.5 or ood):
            likely_fn.append(seg_id)
    return MiningResult(likely_fp=likely_fp, likely_fn=likely_fn)


def export_hard_examples(
    result: MiningResult,
    segment_paths: dict[str, str],
    review_dir: Path,
) -> None:
    """Copy likely-FP/FN audio into outputs/review/likely_fp/ and likely_fn/,
    and write a manifest CSV of what's inside each."""
    fp_dir = review_dir / "likely_fp"
    fn_dir = review_dir / "likely_fn"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    for seg_id in result.likely_fp:
        src = segment_paths.get(seg_id)
        if src and Path(src).exists():
            shutil.copy2(src, fp_dir / Path(src).name)

    for seg_id in result.likely_fn:
        src = segment_paths.get(seg_id)
        if src and Path(src).exists():
            shutil.copy2(src, fn_dir / Path(src).name)

    with open(review_dir / "hard_examples_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "category"])
        for seg_id in result.likely_fp:
            writer.writerow([seg_id, "likely_fp"])
        for seg_id in result.likely_fn:
            writer.writerow([seg_id, "likely_fn"])


def export_user_correction(
    audio: np.ndarray,
    sr: int,
    source_path: str,
    segment_index: int,
    start_sec: float,
    end_sec: float,
    pipeline_band: str,
    corrected_label: str,
    review_dir: Path,
) -> Path:
    """Persist a human-flagged correction (dashboard DD-022) for future
    retraining: writes the segment's audio to
    outputs/review/user_corrections/ and appends a row to
    outputs/review/user_corrections.csv.

    This is the human-in-the-loop counterpart to find_hard_examples/
    export_hard_examples above: those flag likely errors algorithmically from
    known ground truth; this captures a person overriding the pipeline's
    decision on unlabeled real-world audio (design.md §6.5 hard-example
    mining feeds active-learning cycles, §17 Future Work).
    """
    corrections_dir = review_dir / "user_corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)

    seg_name = f"{Path(source_path).stem}_seg{segment_index}.wav"
    seg_path = corrections_dir / seg_name
    write_wav(str(seg_path), audio, sr)

    csv_path = review_dir / "user_corrections.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "source_path",
                    "segment_index",
                    "start_sec",
                    "end_sec",
                    "pipeline_band",
                    "corrected_label",
                    "audio_path",
                ]
            )
        writer.writerow(
            [
                datetime.datetime.now().isoformat(),
                source_path,
                segment_index,
                f"{start_sec:.3f}",
                f"{end_sec:.3f}",
                pipeline_band,
                corrected_label,
                str(seg_path),
            ]
        )
    return seg_path
