"""Load pipeline + experiment YAML; build BirdNET-Analyzer argv and fingerprint."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from utils.config import load_config as load_pipeline_config


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_experiment_config(
    experiment_path: Optional[Path] = None,
    pipeline_config_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (experiment_cfg, pipeline_cfg)."""
    root = _project_root()
    exp_path = experiment_path or (Path(__file__).resolve().parent / "experiment_config.yaml")
    with open(exp_path, "r", encoding="utf-8") as f:
        exp = yaml.safe_load(f) or {}

    pipe_rel = exp.get("pipeline_config", "config.yaml")
    pipe_path = pipeline_config_path or (root / pipe_rel)
    if not pipe_path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {pipe_path}")
    pipeline = load_pipeline_config(str(pipe_path))
    return exp, pipeline


def resolve_under_root(p: str | Path) -> Path:
    root = _project_root()
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def birdnet_python_executable_with_source(exp: Dict[str, Any]) -> Tuple[Path, str]:
    """Resolve BirdNET subprocess interpreter with explicit resolution source.

    Resolution order (first match wins):
        1. Environment variable ``BIRDNET_PYTHON``
        2. ``birdnet_python`` in ``experiment_config.yaml`` (if set and non-empty)
        3. ``sys.executable`` (current Python running the wrapper scripts)

    Returns:
        (absolute_path, source_tag) where source_tag is ``BIRDNET_PYTHON``,
        ``experiment_config``, or ``sys.executable``.
    """
    env = os.environ.get("BIRDNET_PYTHON", "").strip()
    if env:
        return Path(env).expanduser().resolve(), "BIRDNET_PYTHON"

    raw = exp.get("birdnet_python")
    if raw is not None:
        s = str(raw).strip()
        if s and s.lower() != "null":
            p = Path(s)
            if not p.is_absolute():
                return resolve_under_root(p), "experiment_config"
            return p.resolve(), "experiment_config"

    return Path(sys.executable).resolve(), "sys.executable"


def birdnet_python_executable(exp: Dict[str, Any]) -> Path:
    """Interpreter used for ``python -m birdnet_analyzer.analyze`` (same order as *with_source*)."""
    path, _ = birdnet_python_executable_with_source(exp)
    return path


def log_birdnet_python_choice(path: Path, source: str) -> None:
    """Emit logs for which interpreter BirdNET integration will use."""
    print(f"[BirdNET] Using Python: {path}", flush=True)
    print(f"[BirdNET] Resolution order: {source}", flush=True)


def verify_birdnet_import(python_exe: Path) -> Tuple[bool, str]:
    """Return (ok, stderr snippet) after ``import birdnet_analyzer`` in *python_exe*."""
    try:
        r = subprocess.run(
            [str(python_exe), "-c", "import birdnet_analyzer"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        err = (r.stderr or r.stdout or "").strip()
        return r.returncode == 0, err[:500]
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:500]


def preflight_birdnet_cli(exp: Dict[str, Any]) -> Optional[Path]:
    """Resolve interpreter, log choice, verify ``birdnet_analyzer`` imports in subprocess.

    If import fails, logs a warning and returns ``None`` (do not raise).
    """
    py, source = birdnet_python_executable_with_source(exp)
    log_birdnet_python_choice(py, source)
    ok, err = verify_birdnet_import(py)
    if ok:
        return py

    print(
        "[BirdNET] BirdNET not available. Skipping BirdNET baseline/filtered runs.",
        file=sys.stderr,
        flush=True,
    )
    if err:
        print(f"[BirdNET] Import check detail: {err[:400]}", file=sys.stderr, flush=True)
    return None


def build_analyze_argv(
    exp: Dict[str, Any],
    input_path: Path,
    output_path: Path,
) -> List[str]:
    """Build argv for `python -m birdnet_analyzer.analyze` (positional + options)."""
    bn = exp.get("birdnet", {})
    argv: List[str] = [
        "-m",
        "birdnet_analyzer.analyze",
        str(input_path),
        "-o",
        str(output_path),
        "--overlap",
        str(bn.get("overlap", 0.0)),
        "--min_conf",
        str(bn.get("min_conf", 0.25)),
        "--sensitivity",
        str(bn.get("sensitivity", 1.0)),
        "--sf_thresh",
        str(bn.get("sf_thresh", 0.03)),
        "--lat",
        str(bn.get("lat", -1)),
        "--lon",
        str(bn.get("lon", -1)),
        "--week",
        str(bn.get("week", -1)),
        "--locale",
        str(bn.get("locale", "en")),
        "-t",
        str(bn.get("threads", 4)),
        "-b",
        str(bn.get("batch_size", 1)),
        "--fmin",
        str(bn.get("fmin", 0)),
        "--fmax",
        str(bn.get("fmax", 15000)),
        "--merge_consecutive",
        str(bn.get("merge_consecutive", 1)),
    ]
    rtypes = bn.get("rtype", ["csv"])
    if isinstance(rtypes, str):
        rtypes = [rtypes]
    for rt in rtypes:
        argv.extend(["--rtype", rt])
    if bn.get("combine_results"):
        argv.append("--combine_results")
    if bn.get("skip_existing_results"):
        argv.append("--skip_existing_results")
    if bn.get("show_progress"):
        argv.append("--show_progress")
    clf = bn.get("classifier")
    if clf:
        argv.extend(["--classifier", str(resolve_under_root(clf))])
    slist = bn.get("slist")
    if slist:
        argv.extend(["--slist", str(resolve_under_root(slist))])
    return argv


def experiment_fingerprint(exp: Dict[str, Any]) -> str:
    """Stable hash of birdnet + alignment settings for summary.json."""
    payload = {
        "birdnet": exp.get("birdnet", {}),
        "alignment": exp.get("alignment", {}),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
