#!/usr/bin/env python3
"""Phase 6: run the full inference pipeline on a folder of WAV files
(design.md §10, §6.1). Routes each segment to outputs/{bird,uncertain,noise}/
and writes outputs/manifest_results.csv."""

from __future__ import annotations

import argparse
import logging

from pipeline.config import load_config
from pipeline.inference import InferencePipeline, route_output_file, write_manifest_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full inference pipeline on a folder of WAV files."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True, help="Directory of WAV files to process.")
    parser.add_argument(
        "--no-copy", action="store_true", help="Skip physically copying segments into outputs/."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = InferencePipeline.from_artifacts(cfg)

    records = pipeline.process_directory(args.input)
    logger.info("Processed %d segments from %s", len(records), args.input)

    outputs_dir = cfg.resolve_path(cfg.paths.outputs_dir)
    if not args.no_copy:
        for record in records:
            if record.final_band is not None:
                route_output_file(record, outputs_dir, record.path)

    write_manifest_results(records, outputs_dir / "manifest_results.csv")

    bands: dict[str | None, int] = {}
    for r in records:
        bands[r.final_band] = bands.get(r.final_band, 0) + 1
    logger.info("Routing summary: %s", bands)


if __name__ == "__main__":
    main()
