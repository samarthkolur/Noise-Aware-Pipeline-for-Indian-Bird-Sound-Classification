#!/usr/bin/env python3
"""
preprocess_cli.py — CLI for the preprocessing module.

Usage:
    # Process entire dataset (reads paths from config.yaml)
    python -m preprocessing.preprocess_cli --config config.yaml

    # Process a single audio file
    python -m preprocessing.preprocess_cli --config config.yaml \\
        --file path/to/audio.wav --species "Pnoepyga pusilla"

    # Disable silence removal
    python -m preprocessing.preprocess_cli --config config.yaml --keep-silence

    # Custom RMS threshold
    python -m preprocessing.preprocess_cli --config config.yaml --rms-threshold -35
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running as `python -m preprocessing.preprocess_cli` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.preprocessing import Preprocessor, preprocess_dataset
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bioacoustic Preprocessing — convert, segment, and filter audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full dataset
  python -m preprocessing.preprocess_cli --config config.yaml

  # Single file
  python -m preprocessing.preprocess_cli --config config.yaml \\
      --file recordings/Pnoepyga_pusilla/XC12345.wav \\
      --species "Pnoepyga pusilla"

  # Keep silent segments
  python -m preprocessing.preprocess_cli --config config.yaml --keep-silence
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a single audio file instead of the full dataset",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="unknown",
        help="Species label when processing a single file (default: unknown)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--keep-silence",
        action="store_true",
        help="Disable RMS-based silence removal",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=None,
        help="Override RMS threshold in dB (default from config, typically -40)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to save a JSON summary of all processed segments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ── Apply CLI overrides ────────────────────────────────
    if args.keep_silence:
        cfg.setdefault("silence_removal", {})["enabled"] = False

    if args.rms_threshold is not None:
        cfg.setdefault("silence_removal", {})["rms_threshold_db"] = args.rms_threshold

    if args.output_dir:
        cfg["data"]["processed_dir"] = args.output_dir

    output_dir = Path(cfg["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run ────────────────────────────────────────────────
    if args.file:
        # Single file mode
        audio_path = Path(args.file)
        if not audio_path.exists():
            logger.error(f"File not found: {audio_path}")
            sys.exit(1)

        preprocessor = Preprocessor(cfg)
        metas = preprocessor.process_file(audio_path, output_dir, args.species)
        logger.info(f"Processed 1 file → {len(metas)} segments")
    else:
        # Full dataset mode
        metas = preprocess_dataset(cfg)

    # ── Summary report ─────────────────────────────────────
    _print_summary(metas)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump([asdict(m) for m in metas], f, indent=2)
        logger.info(f"Summary saved → {summary_path}")


def _print_summary(metas) -> None:
    """Print a concise summary table to the console."""
    if not metas:
        logger.info("No segments produced.")
        return

    species_counts: dict[str, int] = {}
    total_duration = 0.0
    for m in metas:
        species_counts[m.species] = species_counts.get(m.species, 0) + 1
        total_duration += m.duration_sec

    logger.info("─" * 50)
    logger.info("Preprocessing Summary")
    logger.info("─" * 50)
    logger.info(f"  Total segments : {len(metas)}")
    logger.info(f"  Total duration : {total_duration:.1f}s ({total_duration/60:.1f}m)")
    logger.info(f"  Species        : {len(species_counts)}")
    logger.info("─" * 50)

    for species in sorted(species_counts):
        logger.info(f"  {species:40s} {species_counts[species]:>5d} segments")

    logger.info("─" * 50)


if __name__ == "__main__":
    main()
