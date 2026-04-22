#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from _compat import run_root_script


def _normalize_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate the trained classifier",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluate on all manifest rows instead of only the held-out test split.",
    )
    parser.parse_args(argv)
    return argv


if __name__ == "__main__":
    run_root_script("evaluate.py", _normalize_args(sys.argv[1:]))
