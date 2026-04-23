"""Threshold-loading helpers for reporting and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_optimal_threshold(cfg: dict, default: float = 0.5) -> float:
    """Load the saved validation-optimal classifier threshold, if available."""
    chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
    meta_path = chkpt_dir / "best_model_meta.json"

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            raw = meta.get("optimal_threshold")
            if raw is not None:
                return float(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    return float(default)


def resolve_threshold_arg(raw: Any, cfg: dict, default: float = 0.5) -> float:
    """Resolve a CLI threshold argument to a float.

    Accepts numeric values directly or the string ``"auto"`` (plus a few
    synonyms), which pulls the saved classifier threshold from checkpoint
    metadata.
    """
    if raw is None:
        return float(default)

    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return float(default)

    if text.lower() in {"auto", "best", "optimal"}:
        return load_optimal_threshold(cfg, default=default)

    return float(text)


def threshold_mode_label(raw: Any) -> str:
    """Return ``auto`` when the threshold came from checkpoint metadata."""
    if raw is None:
        return "fixed"
    return "auto" if str(raw).strip().lower() in {"auto", "best", "optimal"} else "fixed"
