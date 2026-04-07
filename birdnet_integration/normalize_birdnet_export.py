#!/usr/bin/env python3
"""Normalize BirdNET-Analyzer *.BirdNET.results.csv into unified JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from birdnet_integration.integration_config import load_experiment_config, resolve_under_root

# Official BirdNET-Analyzer CSV (see birdnet_analyzer.analyze.utils.CSV_HEADER)
CSV_COLS = [
    "Start (s)",
    "End (s)",
    "Scientific name",
    "Common name",
    "Confidence",
    "File",
]


def iter_birdnet_result_csvs(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    yield from root.rglob("*.BirdNET.results.csv")


def read_birdnet_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # tolerate column name drift
    cols = {c.strip(): c for c in df.columns}
    rename = {}
    for want in CSV_COLS:
        if want in df.columns:
            continue
        for k, v in cols.items():
            if k.replace(" ", "") == want.replace(" ", ""):
                rename[v] = want
                break
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}, have {list(df.columns)}")
    return df


def row_to_record(
    run: str,
    row: pd.Series,
    source_override: Optional[str] = None,
    time_offset: float = 0.0,
    segment_index: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = float(row["Start (s)"]) + time_offset
    t1 = float(row["End (s)"]) + time_offset
    sci = str(row["Scientific name"]).strip()
    com = str(row["Common name"]).strip()
    species_code = f"{sci}_{com}" if sci or com else "Unknown"
    file_val = str(row["File"]).strip()
    src = source_override if source_override is not None else file_val
    return {
        "run": run,
        "input_path": file_val,
        "source_file": src,
        "segment_index": segment_index,
        "start_sec": t0,
        "end_sec": t1,
        "species_code": species_code,
        "confidence": float(row["Confidence"]),
        "birdnet_raw_row": {c: row[c] for c in row.index},
    }


def normalize_directory(
    input_dir: Path,
    run: str,
    output_jsonl: Path,
    *,
    filtered_sidecar_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    """Write one JSON object per line (detection). Returns line count."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for csv_path in sorted(iter_birdnet_result_csvs(input_dir)):
            df = read_birdnet_csv(csv_path)
            stem = csv_path.name.replace(".BirdNET.results.csv", "")
            side: Optional[Dict[str, Any]] = None
            if filtered_sidecar_map and stem in filtered_sidecar_map:
                side = filtered_sidecar_map[stem]
            for _, row in df.iterrows():
                offset = 0.0
                src_override: Optional[str] = None
                seg_idx: Optional[int] = None
                if side:
                    offset = float(side["start_sec"])
                    src_override = str(side["source_file"])
                    seg_idx = int(side["segment_index"])
                rec = row_to_record(
                    run,
                    row,
                    source_override=src_override,
                    time_offset=offset,
                    segment_index=seg_idx,
                )
                out.write(json.dumps(rec) + "\n")
                n += 1
    return n


def load_clean_birds_sidecars(clean_birds: Path) -> Dict[str, Dict[str, Any]]:
    """Map segment stem (filename without .wav) -> sidecar dict."""
    out: Dict[str, Dict[str, Any]] = {}
    for js in sorted(clean_birds.glob("*.json")):
        with open(js, "r", encoding="utf-8") as f:
            meta = json.load(f)
        stem = js.stem
        out[stem] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize BirdNET CSV exports to JSONL")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_config.yaml",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="baseline_birdnet or filtered_birdnet folder")
    parser.add_argument("--run", choices=("baseline", "filtered"), required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    exp, pipeline = load_experiment_config(args.experiment, None)
    comp = resolve_under_root(exp["paths"]["comparison_dir"])
    comp.mkdir(parents=True, exist_ok=True)
    out_path = args.output_jsonl
    if out_path is None:
        name = "baseline_normalized.jsonl" if args.run == "baseline" else "filtered_normalized.jsonl"
        out_path = comp / name

    inp = resolve_under_root(args.input_dir)
    side_map = None
    if args.run == "filtered":
        clean = resolve_under_root(pipeline["data"]["output_dir"]) / "clean_birds"
        side_map = load_clean_birds_sidecars(clean)

    n = normalize_directory(inp, args.run, out_path, filtered_sidecar_map=side_map)
    print(f"Wrote {n} detections to {out_path}")


if __name__ == "__main__":
    main()
