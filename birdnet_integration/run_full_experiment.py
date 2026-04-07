#!/usr/bin/env python3
"""Run baseline BirdNET → filtered BirdNET → normalize → compare (full chain).

``compare_baseline_filtered`` rebuilds ``segment_alignment`` internally; use
``python -m birdnet_integration.align_segments`` only if you want the table without plots.

Run from project root (same directory as config.yaml)::

    python -m birdnet_integration.run_full_experiment

Prerequisites: trained MLP + inference so ``outputs/clean_birds/`` exists. In the **same** venv as
this command, install pipeline + integration deps: ``pip install -r requirements.txt`` or at least
``pip install -r requirements-birdnet-integration.txt`` (pandas, matplotlib, pyarrow for
normalize/compare). BirdNET CLI may use a separate Python via ``birdnet_python`` in
``experiment_config.yaml``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def _run_soft(cmd: list[str], label: str) -> None:
    """Run a step without aborting the chain (optional BirdNET follow-up steps)."""
    print("\n>>>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(
            f"[BirdNET] {label} exited with code {proc.returncode}. "
            "Optional integration — continuing.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full BirdNET baseline vs filtered experiment")
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-filtered", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    args = parser.parse_args()

    exp = ["--experiment", str(args.experiment)] if args.experiment else []
    py = sys.executable

    if not args.skip_baseline:
        _run([py, "-m", "birdnet_integration.run_baseline_birdnet", *exp])
    if not args.skip_filtered:
        _run([py, "-m", "birdnet_integration.run_filtered_birdnet", *exp])
    if not args.skip_normalize:
        _run_soft(
            [
                py,
                "-m",
                "birdnet_integration.normalize_birdnet_export",
                *exp,
                "--input-dir",
                "outputs/baseline_birdnet",
                "--run",
                "baseline",
            ],
            "normalize_birdnet_export (baseline)",
        )
        _run_soft(
            [
                py,
                "-m",
                "birdnet_integration.normalize_birdnet_export",
                *exp,
                "--input-dir",
                "outputs/filtered_birdnet",
                "--run",
                "filtered",
            ],
            "normalize_birdnet_export (filtered)",
        )
    _run_soft([py, "-m", "birdnet_integration.compare_baseline_filtered", *exp], "compare_baseline_filtered")


if __name__ == "__main__":
    main()
