"""Training run provenance: git SHA, pip freeze, resolved config."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def dump_run_metadata(cfg: dict, trainer: str) -> Path:
    """Write ``results/run_metadata.json`` (path from ``evaluation.results_dir``)."""
    results_dir = Path(cfg.get("evaluation", {}).get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "run_metadata.json"

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        sha = ""

    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        freeze = ""

    payload: Dict[str, Any] = {
        "trainer": trainer,
        "git_sha": sha,
        "pip_freeze": freeze,
        "config": cfg,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path
