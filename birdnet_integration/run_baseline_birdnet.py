#!/usr/bin/env python3
"""Run BirdNET-Analyzer on raw audio (Pipeline A baseline). Optional: skips if BirdNET unavailable."""

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
    parser = argparse.ArgumentParser(description="BirdNET baseline on data.raw_dir")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_config.yaml",
        help="Experiment YAML (BirdNET args + paths)",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="Override pipeline config.yaml path",
    )
    args = parser.parse_args()

    try:
        exp, pipeline = load_experiment_config(args.experiment, args.pipeline_config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    raw_dir = resolve_under_root(pipeline["data"]["raw_dir"])
    if not raw_dir.is_dir():
        print(f"ERROR: raw_dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    out = resolve_under_root(exp["paths"]["baseline_birdnet_out"])
    out.mkdir(parents=True, exist_ok=True)

    py = preflight_birdnet_cli(exp)
    if py is None:
        print(
            "[BirdNET] Skipped baseline run (BirdNET Analyzer not available).",
            flush=True,
        )
        return

    argv = [str(py)] + build_analyze_argv(exp, raw_dir, out)
    print("[BirdNET] Running:", " ".join(argv), flush=True)

    try:
        proc = subprocess.run(argv, cwd=str(raw_dir.parent))
    except Exception as e:  # noqa: BLE001
        print(
            f"[BirdNET] ERROR during baseline run ({type(e).__name__}): {e}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[BirdNET] Continuing without BirdNET baseline (optional integration).",
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
