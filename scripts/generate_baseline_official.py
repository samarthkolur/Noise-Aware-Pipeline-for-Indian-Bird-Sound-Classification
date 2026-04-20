#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from _compat import run_root_script


def _normalize_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("-h", "--help", action="store_true")
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.help:
        print(
            "usage: python scripts/generate_baseline_official.py [--config CONFIG]\n\n"
            "Runs the repository's BirdNET baseline generation workflow.\n"
            "The current implementation ignores --config and uses repo paths."
        )
        raise SystemExit(0)
    return remaining


if __name__ == "__main__":
    run_root_script("compute_real_baseline.py", _normalize_args(sys.argv[1:]))
