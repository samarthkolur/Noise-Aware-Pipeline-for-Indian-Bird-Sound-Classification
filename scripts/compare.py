#!/usr/bin/env python3
from __future__ import annotations

import sys

from _compat import run_root_script


def _normalize_args(argv: list[str]) -> list[str]:
    # Keep compatibility with older README examples.
    normalized: list[str] = []
    for arg in argv:
        if arg == "--full":
            normalized.append("--full-dataset")
        else:
            normalized.append(arg)
    return normalized


if __name__ == "__main__":
    run_root_script("compute_baseline_metrics.py", _normalize_args(sys.argv[1:]))
