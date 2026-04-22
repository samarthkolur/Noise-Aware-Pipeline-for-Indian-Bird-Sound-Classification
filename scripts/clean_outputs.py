#!/usr/bin/env python3
from __future__ import annotations

import sys

from _compat import run_root_script


if __name__ == "__main__":
    run_root_script("clean_pipeline_outputs.py", sys.argv[1:])
