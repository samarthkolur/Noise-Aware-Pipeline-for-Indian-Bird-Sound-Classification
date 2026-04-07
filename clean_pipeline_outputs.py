#!/usr/bin/env python3
"""
Remove cached pipeline artifacts so the next run does not mix old data.

Deletes (relative to this repo root):
  - data/processed   — segmented WAVs and segments_manifest.csv
  - data/embeddings    — HDF5 embeddings and manifest.csv
  - checkpoints/     — trained MLP weights and best_model_meta.json

Does not delete: raw audio (e.g. iBC53/), config, or code.

Usage:
  python clean_pipeline_outputs.py           # delete
  python clean_pipeline_outputs.py --dry-run # only print what would be removed
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete processed data, embeddings, checkpoints.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths only; do not delete anything.",
    )
    args = parser.parse_args()

    root = _repo_root()
    targets = [
        root / "data" / "processed",
        root / "data" / "embeddings",
        root / "checkpoints",
    ]

    for d in targets:
        if not d.exists():
            print(f"[skip] not found: {d}")
            continue
        if args.dry_run:
            print(f"[would remove] {d}")
            continue
        shutil.rmtree(d)
        print(f"[removed] {d}")

    if args.dry_run:
        print("Dry run only — no directories were deleted.")
    else:
        print("Done. You can run preprocess → embed → train from a clean state.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
