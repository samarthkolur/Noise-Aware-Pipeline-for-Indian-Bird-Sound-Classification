#!/usr/bin/env python3
"""Map BirdNET baseline intervals onto pipeline 3s segments; merge filtered arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from birdnet_integration.integration_config import load_experiment_config, resolve_under_root
from birdnet_integration.normalize_birdnet_export import iter_birdnet_result_csvs, read_birdnet_csv


def canonical_segment_id(source_file: str, start_sec: float, end_sec: float) -> str:
    try:
        p = str(Path(source_file).resolve())
    except OSError:
        p = source_file
    return f"{p}::{start_sec:.3f}-{end_sec:.3f}"


def same_audio_source(a: str, b: str) -> bool:
    pa, pb = Path(a), Path(b)
    try:
        if pa.exists() and pb.exists():
            return pa.resolve() == pb.resolve()
    except OSError:
        pass
    return pa.name.lower() == pb.name.lower()


def overlap_fraction(t0: float, t1: float, s0: float, s1: float) -> float:
    ov = max(0.0, min(t1, s1) - max(t0, s0))
    dlen = max(1e-9, t1 - t0)
    slen = max(1e-9, s1 - s0)
    return ov / min(dlen, slen)


def detection_maps_to_segment(
    t0: float,
    t1: float,
    s0: float,
    s1: float,
    tau: float,
    use_midpoint: bool,
) -> bool:
    if use_midpoint:
        mid = (t0 + t1) / 2.0
        if s0 <= mid <= s1:
            return True
    return overlap_fraction(t0, t1, s0, s1) >= tau


def collect_pipeline_segments(output_dir: Path) -> List[Dict[str, Any]]:
    """Union sidecars from clean_birds, noise, uncertain."""
    rows: List[Dict[str, Any]] = []
    for sub, bucket in (
        ("clean_birds", "bird"),
        ("noise", "noise"),
        ("uncertain", "uncertain"),
    ):
        d = output_dir / sub
        if not d.is_dir():
            continue
        for js in sorted(d.glob("*.json")):
            with open(js, "r", encoding="utf-8") as f:
                meta = json.load(f)
            src = str(meta["source_file"])
            s0 = float(meta["start_sec"])
            s1 = float(meta["end_sec"])
            stem = js.stem
            rows.append(
                {
                    "segment_id": canonical_segment_id(src, s0, s1),
                    "source_file": src,
                    "start_sec": s0,
                    "end_sec": s1,
                    "segment_index": int(meta.get("segment_index", -1)),
                    "pipeline_bucket": bucket,
                    "segment_stem": stem,
                    "wav_path": str(meta.get("output_path", "")),
                }
            )
    return rows


def load_baseline_detection_rows(baseline_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for csv_path in iter_birdnet_result_csvs(baseline_dir):
        df = read_birdnet_csv(csv_path)
        for _, row in df.iterrows():
            t0 = float(row["Start (s)"])
            t1 = float(row["End (s)"])
            sci = str(row["Scientific name"]).strip()
            com = str(row["Common name"]).strip()
            species = f"{sci}_{com}" if sci or com else "Unknown"
            rows.append(
                {
                    "file": str(row["File"]).strip(),
                    "start_sec": t0,
                    "end_sec": t1,
                    "species_code": species,
                    "confidence": float(row["Confidence"]),
                }
            )
    return rows


def load_filtered_for_stem(filtered_dir: Path, stem: str) -> List[Dict[str, Any]]:
    path = filtered_dir / f"{stem}.BirdNET.results.csv"
    if not path.is_file():
        return []
    df = read_birdnet_csv(path)
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        t0 = float(row["Start (s)"])
        t1 = float(row["End (s)"])
        sci = str(row["Scientific name"]).strip()
        com = str(row["Common name"]).strip()
        species = f"{sci}_{com}" if sci or com else "Unknown"
        out.append(
            {
                "start_sec": t0,
                "end_sec": t1,
                "species_code": species,
                "confidence": float(row["Confidence"]),
            }
        )
    return out


def align(
    exp: Dict[str, Any],
    pipeline: Dict[str, Any],
) -> Tuple[pd.DataFrame, Path]:
    align_cfg = exp.get("alignment", {})
    tau = float(align_cfg.get("overlap_tau", 0.5))
    use_mid = bool(align_cfg.get("use_midpoint_fallback", True))

    out_root = resolve_under_root(pipeline["data"]["output_dir"])
    baseline_dir = resolve_under_root(exp["paths"]["baseline_birdnet_out"])
    filtered_dir = resolve_under_root(exp["paths"]["filtered_birdnet_out"])
    comp_dir = resolve_under_root(exp["paths"]["comparison_dir"])
    comp_dir.mkdir(parents=True, exist_ok=True)

    segments = collect_pipeline_segments(out_root)
    if not segments:
        raise RuntimeError(f"No segment JSON sidecars under {out_root}")

    base_dets = load_baseline_detection_rows(baseline_dir)

    records: List[Dict[str, Any]] = []
    for seg in segments:
        s0, s1 = seg["start_sec"], seg["end_sec"]
        src = seg["source_file"]
        b_list: List[Dict[str, Any]] = []
        for d in base_dets:
            if not same_audio_source(d["file"], src):
                continue
            if not detection_maps_to_segment(
                d["start_sec"], d["end_sec"], s0, s1, tau, use_mid
            ):
                continue
            b_list.append(
                {
                    "species_code": d["species_code"],
                    "confidence": d["confidence"],
                    "start_sec": d["start_sec"],
                    "end_sec": d["end_sec"],
                }
            )

        f_list: List[Dict[str, Any]] = []
        if seg["pipeline_bucket"] == "bird":
            f_list = load_filtered_for_stem(filtered_dir, seg["segment_stem"])
            # Global times for filtered (segment file is [0, duration))
            f_list = [
                {
                    **x,
                    "start_sec": x["start_sec"] + s0,
                    "end_sec": x["end_sec"] + s0,
                }
                for x in f_list
            ]

        b_species = sorted({x["species_code"] for x in b_list})
        f_species = sorted({x["species_code"] for x in f_list})
        records.append(
            {
                **seg,
                "baseline_detection_count": len(b_list),
                "filtered_detection_count": len(f_list),
                "baseline_species": json.dumps(b_species),
                "filtered_species": json.dumps(f_species),
                "baseline_detections_json": json.dumps(b_list),
                "filtered_detections_json": json.dumps(f_list),
            }
        )

    df = pd.DataFrame.from_records(records)
    out_path = comp_dir / "segment_alignment.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Parquet write failed ({exc}), writing CSV instead.", file=sys.stderr)
        out_path = comp_dir / "segment_alignment.csv"
        df.to_csv(out_path, index=False)
    return df, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment_alignment table")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_config.yaml",
    )
    args = parser.parse_args()
    exp, pipeline = load_experiment_config(args.experiment, None)
    df, path = align(exp, pipeline)
    print(f"Segments: {len(df)}  wrote {path}")


if __name__ == "__main__":
    main()
