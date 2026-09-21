#!/usr/bin/env python3
"""Phase 2: build data/manifest.csv and data/split.csv (design.md §10 Phase 2).

Source corpus: this environment does not have request-gated access to the
official iBC53 release (design.md §18/§25). It does have a real, raw (variable
-length, unsegmented) corpus of 53 Indian bird species under
data/iBC53/<species>/*.wav (config.paths.raw_data_dir). This script performs
the actual Phase 2 work the design specifies: resample -> 3 s non-overlapping
segmentation (pipeline.audio.segment_audio) -> RMS silence gate -> manifest,
writing the resulting clips to data/segments/ (config.paths.segments_dir).

Note: data/segmented/ (no 's' before '/') is a *different*, legacy directory
produced by a now-deleted pipeline that used overlapping windows (~1 s hop on
a 3 s window) — confirmed by comparing its segment counts against the raw
corpus's duration. It is no longer read by this script because overlapping
segments from the same recording could land in different train/val/test
splits, violating the independence the stratified split (DD-008) assumes.

No real noise-class recordings are available in this environment. The noise
class is filled with synthetic segments (pipeline/synthetic.py), which is
strictly opt-in (--allow-synthetic) and never runs by default (DD-010). This
is documented as a reproduction gap in design.md, not silently substituted.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

from pipeline.audio import is_silent, load_audio, rms_db, segment_audio, write_wav
from pipeline.config import PROJECT_ROOT, load_config
from pipeline.synthetic import generate_synthetic_noise_corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_manifest(
    raw_data_dir: Path,
    segments_dir: Path,
    max_segments_per_species: int | None,
    silence_db_threshold: float,
    target_sr: int,
    segment_length_s: float,
) -> list[dict]:
    """Resample + non-overlapping-segment every raw recording under raw_data_dir,
    RMS-gate each resulting clip, write survivors to segments_dir, and return
    their bird-class manifest rows.

    Raw files are processed in a fixed random order per species and stop early
    once max_segments_per_species survivors have been written, so a modest
    sample doesn't require decoding the full ~6.6 GB raw corpus.
    """
    rows = []
    species_dirs = sorted(p for p in raw_data_dir.iterdir() if p.is_dir())
    logger.info("Found %d species directories under %s", len(species_dirs), raw_data_dir)

    rng = random.Random(42)
    for species_dir in species_dirs:
        species_slug = species_dir.name.replace(" ", "_")
        out_dir = segments_dir / species_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(species_dir.glob("*.wav"))
        rng.shuffle(raw_files)

        cap = max_segments_per_species
        species_rows: list[dict] = []
        for raw_path in raw_files:
            if cap is not None and len(species_rows) >= cap:
                break
            audio, sr = load_audio(str(raw_path), target_sr)
            segments = segment_audio(audio, sr, segment_length_s)

            for i, seg in enumerate(segments):
                if cap is not None and len(species_rows) >= cap:
                    break
                level_db = rms_db(seg)
                silent = is_silent(seg, silence_db_threshold)
                seg_id = f"{species_slug}__{raw_path.stem}_seg{i}"
                seg_path = out_dir / f"{raw_path.stem}_seg{i}.wav"
                if not silent:
                    write_wav(str(seg_path), seg, sr)
                species_rows.append(
                    {
                        "segment_id": seg_id,
                        "path": str(seg_path.relative_to(PROJECT_ROOT)),
                        "species_label": species_dir.name,
                        "class": "bird",
                        "rms_db": f"{level_db:.4f}",
                        "silence_rejected": str(silent),
                    }
                )
        rows.extend(species_rows)
    return rows


def build_synthetic_noise_rows(
    n_segments: int,
    target_sr: int,
    segment_length_s: float,
    noise_types: list[str],
    output_dir: Path,
    seed: int,
) -> list[dict]:
    """Generate and write synthetic noise WAVs, returning their manifest rows."""
    from pipeline.audio import write_wav

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = generate_synthetic_noise_corpus(
        n_segments=n_segments,
        sr=target_sr,
        segment_length_s=segment_length_s,
        noise_types=noise_types,
        allow_synthetic_noise=True,
        seed=seed,
    )
    rows = []
    for i, seg in enumerate(segments):
        seg_id = f"synthetic_noise_{i:05d}"
        wav_path = output_dir / f"{seg_id}.wav"
        write_wav(str(wav_path), seg, target_sr)
        rows.append(
            {
                "segment_id": seg_id,
                "path": str(wav_path.relative_to(PROJECT_ROOT)),
                "species_label": "noise",
                "class": "noise",
                "rms_db": f"{rms_db(seg):.4f}",
                "silence_rejected": "False",
            }
        )
    return rows


def stratified_split(
    rows: list[dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> None:
    """Assign a 'split' column in-place, stratified by class (DD-008)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)

    rng = random.Random(seed)
    for _cls, cls_rows in by_class.items():
        rng.shuffle(cls_rows)
        n = len(cls_rows)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for i, row in enumerate(cls_rows):
            if i < n_train:
                row["split"] = "train"
            elif i < n_train + n_val:
                row["split"] = "val"
            else:
                row["split"] = "test"


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare manifest + split for the pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--max-segments-per-species",
        type=int,
        default=20,
        help="Cap non-overlapping 3 s segments produced per species (the raw corpus is "
        "~6.6 GB; the default keeps a full end-to-end run tractable in development). "
        "Use --max-segments-per-species=-1 to process the entire raw corpus.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Generate synthetic noise-class segments (DD-010, opt-in).",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=200,
        help="Number of synthetic noise segments to generate if --allow-synthetic is set.",
    )
    parser.add_argument("--force-resplit", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_segments_per_species = (
        None if args.max_segments_per_species == -1 else args.max_segments_per_species
    )

    raw_data_dir = cfg.resolve_path(cfg.paths.raw_data_dir)
    segments_dir = cfg.resolve_path(cfg.paths.segments_dir)
    manifest_path = cfg.resolve_path(cfg.paths.manifest_path)
    split_path = cfg.resolve_path(cfg.paths.split_path)

    if split_path.exists() and not args.force_resplit:
        logger.info(
            "%s already exists; leaving it untouched (DD-008). Use --force-resplit to overwrite.",
            split_path,
        )
        return

    rows = build_manifest(
        raw_data_dir,
        segments_dir,
        max_segments_per_species,
        cfg.audio.silence_db_threshold,
        cfg.audio.target_sr,
        cfg.audio.segment_length_s,
    )
    logger.info("Collected %d real bird-class segments", len(rows))

    if args.allow_synthetic:
        synthetic_dir = cfg.resolve_path(cfg.paths.data_dir) / "synthetic_noise"
        noise_rows = build_synthetic_noise_rows(
            n_segments=args.n_synthetic,
            target_sr=cfg.audio.target_sr,
            segment_length_s=cfg.audio.segment_length_s,
            noise_types=cfg.synthetic.noise_types,
            output_dir=synthetic_dir,
            seed=cfg.random_seed,
        )
        rows.extend(noise_rows)
        logger.info("Added %d synthetic noise-class segments", len(noise_rows))
    else:
        logger.warning(
            "No noise-class segments included (pass --allow-synthetic to generate them). "
            "The manifest will contain bird-class segments only."
        )

    rows = [r for r in rows if r["silence_rejected"] != "True"]

    stratified_split(
        rows, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio, cfg.random_seed
    )

    fieldnames = [
        "segment_id",
        "path",
        "species_label",
        "class",
        "rms_db",
        "silence_rejected",
        "split",
    ]
    write_csv(rows, manifest_path, fieldnames)
    write_csv(rows, split_path, ["segment_id", "class", "split"])

    logger.info("Wrote %d rows to %s", len(rows), manifest_path)
    logger.info("Wrote split assignments to %s", split_path)


if __name__ == "__main__":
    main()
