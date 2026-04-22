from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ROOT


def run_root_script(script_name: str, argv: list[str] | None = None) -> None:
    root = project_root()
    if argv is not None:
        sys.argv = [str(root / script_name), *argv]
    runpy.run_path(str(root / script_name), run_name="__main__")
