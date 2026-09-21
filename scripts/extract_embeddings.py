#!/usr/bin/env python3
"""Phase 4: run BirdNET V2.4 on every manifest segment, populate the HDF5 cache.

Idempotent (DD-003): re-running this script only extracts embeddings for
segment_ids missing from the cache.
"""

from __future__ import annotations

import argparse
import csv
import logging

from pipeline.audio import load_audio
from pipeline.cache import EmbeddingCache
from pipeline.config import load_config
from pipeline.embedding import extract_embeddings_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BirdNET embeddings into the HDF5 cache.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap number of segments processed (dev/testing)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = cfg.resolve_path(cfg.paths.manifest_path)
    cache = EmbeddingCache(cfg.resolve_path(cfg.paths.embeddings_cache_path))

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    segment_ids = [r["segment_id"] for r in rows]
    missing_ids = cache.missing(segment_ids)
    logger.info(
        "%d/%d segments already cached; %d to extract",
        len(rows) - len(missing_ids),
        len(rows),
        len(missing_ids),
    )

    if not missing_ids:
        logger.info("Cache already complete for this manifest; nothing to do.")
        return

    missing_rows = {r["segment_id"]: r for r in rows if r["segment_id"] in missing_ids}
    missing_id_list = list(missing_rows.keys())

    for i in range(0, len(missing_id_list), args.batch_size):
        batch_ids = missing_id_list[i : i + args.batch_size]
        audios = []
        for seg_id in batch_ids:
            audio, _ = load_audio(
                str(cfg.resolve_path(missing_rows[seg_id]["path"])), cfg.audio.target_sr
            )
            audios.append(audio)

        embeddings = extract_embeddings_batch(audios, cfg.audio.target_sr, cfg.embedding)
        n_written = cache.write_batch(batch_ids, embeddings)
        logger.info(
            "Extracted %d/%d (%d newly cached)",
            min(i + args.batch_size, len(missing_id_list)),
            len(missing_id_list),
            n_written,
        )

    logger.info("Done. Cache now has %d entries at %s", len(cache.keys()), cache.cache_path)


if __name__ == "__main__":
    main()
