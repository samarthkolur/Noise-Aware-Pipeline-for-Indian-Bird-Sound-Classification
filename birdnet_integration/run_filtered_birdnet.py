#!/usr/bin/env python3
"""Run BirdNET-Analyzer on outputs/clean_birds (Pipeline B filtered). Optional: skips if BirdNET unavailable."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from birdnet_integration.integration_config import (
    build_analyze_argv,
    load_experiment_config,
    preflight_birdnet_cli,
    resolve_under_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdNET on clean_birds segment WAVs")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_config.yaml",
    )
    parser.add_argument("--pipeline-config", type=Path, default=None)
    args = parser.parse_args()

    try:
        exp, pipeline = load_experiment_config(args.experiment, args.pipeline_config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = resolve_under_root(pipeline["data"]["output_dir"])
    clean_birds = output_dir / "clean_birds"
    if not clean_birds.is_dir():
        print(f"ERROR: clean_birds not found: {clean_birds}", file=sys.stderr)
        sys.exit(1)

    wavs = list(clean_birds.glob("*.wav"))
    if not wavs:
        print(f"ERROR: no WAV files in {clean_birds}", file=sys.stderr)
        sys.exit(1)

    out = resolve_under_root(exp["paths"]["filtered_birdnet_out"])
    out.mkdir(parents=True, exist_ok=True)

    py = preflight_birdnet_cli(exp)
    if py is None:
        print(
            "[BirdNET] Skipped filtered run (BirdNET Analyzer not available).",
            flush=True,
        )
        return

    argv = [str(py)] + build_analyze_argv(exp, clean_birds, out)
    print("[BirdNET] Running:", " ".join(argv), flush=True)

    try:
        proc = subprocess.run(argv)
    except Exception as e:  # noqa: BLE001
        print(
            f"[BirdNET] ERROR during filtered run ({type(e).__name__}): {e}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[BirdNET] Continuing without BirdNET filtered (optional integration).",
            flush=True,
        )
        return

    if proc.returncode != 0:
        print(
            f"[BirdNET] analyze exited with code {proc.returncode}. "
            "Optional integration — not failing the process.",
            flush=True,
        )


if __name__ == "__main__":
    main()
