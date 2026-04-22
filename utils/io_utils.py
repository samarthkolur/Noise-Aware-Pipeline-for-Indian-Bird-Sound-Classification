"""
io_utils.py — Path helpers and safe file I/O utilities.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path (for chaining).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_save(data: Any, path: Path, fmt: str = "json") -> None:
    """Atomically save data to a file (write to temp, then rename).

    Args:
        data: Data to save (dict for JSON, str for text).
        path: Target file path.
        fmt: Format — 'json' or 'text'.
    """
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        if fmt == "json":
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(tmp_path, "w") as f:
                f.write(str(data))

        shutil.move(str(tmp_path), str(path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_json(path: Path) -> Dict:
    """Load a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def list_audio_files(directory: Path, extensions: tuple = (".wav", ".mp3", ".flac")) -> list:
    """Recursively list audio files under a directory.

    Args:
        directory: Root directory to search.
        extensions: Tuple of valid audio extensions.

    Returns:
        Sorted list of Path objects.
    """
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)
