"""
config.py — YAML configuration loading with defaults merging.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULTS: Dict[str, Any] = {
    "project": {"name": "bioacoustic_pipeline", "seed": 42, "device": "auto"},
    "audio": {
        "sample_rate": 48000,  # BirdNET V2.4 requires 48 kHz
        "segment_duration_s": 3.0,
        "overlap": 0.0,        # Non-overlapping segments (matches preprocessing logic)
        "mono": True,
    },
}


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file and merge with defaults.

    Args:
        path: Path to the YAML file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    # Deep merge: defaults ← user overrides
    merged = _deep_merge(_DEFAULTS, user_cfg)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
