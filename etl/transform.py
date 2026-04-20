"""
ETL -- Transform Step
Applies existing preprocessing pipeline (resample -> normalize -> segment -> classify)
and writes outputs to data/processed/ with the structure:

    data/processed/<species>/   <- bird segments
    data/processed/noise/       <- noise segments
"""

import os
import sys
import glob
import csv

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Allow importing from preprocessing/ without running segment_audio as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.segment_audio import (
    normalize_waveform,
    split_into_segments,
    classify_segment,
    TARGET_SR,
    SEGMENT_LENGTH,
)

INPUT_ROOT  = os.path.join("data", "IBC53")
OUTPUT_ROOT = os.path.join("data", "processed")


def transform_dataset(input_root=INPUT_ROOT, output_root=OUTPUT_ROOT):
    """
    Process every .wav file under `input_root`.
    Bird segments -> output_root/<species>/
    Noise segments -> output_root/noise/

    Returns a summary dict with counts.
    """
    if not os.path.isdir(input_root):
        raise FileNotFoundError(
            f"[TRANSFORM] Input directory not found: '{input_root}'\n"
            "Run:  python download_ibc53.py  first."
        )

    audio_files = glob.glob(os.path.join(input_root, "**", "*.wav"), recursive=True)
    if not audio_files:
        raise RuntimeError(f"[TRANSFORM] No .wav files found under '{input_root}'.")

    print(f"[TRANSFORM] Found {len(audio_files)} files under '{input_root}'.")
    print(f"[TRANSFORM] Output -> '{output_root}'")

    total = bird_total = noise_total = 0
    per_species: dict = {}

    for file_path in tqdm(audio_files, desc="Transforming"):
        species_name = os.path.basename(os.path.dirname(file_path))

        try:
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        except Exception as exc:
            print(f"[WARN] Skipping '{file_path}': {exc}")
            continue

        y        = normalize_waveform(y, sr)
        segments = split_into_segments(y, sr, SEGMENT_LENGTH)
        base     = os.path.splitext(os.path.basename(file_path))[0]
        # species name safe for use in filenames (spaces -> underscores)
        safe_species = species_name.replace(" ", "_")

        for i, segment in enumerate(segments):
            label = classify_segment(segment, sr)

            if label == "bird":
                out_dir  = os.path.join(output_root, species_name)
                filename = f"{base}_seg{i:03d}.wav"
            else:
                # Prefix with species to prevent cross-species filename collision
                # Format: {species}__{source_file}__seg{index}.wav
                out_dir  = os.path.join(output_root, "noise")
                filename = f"{safe_species}__{base}__seg{i:03d}.wav"

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)

            # Safe write: skip if already exists (guards against partial re-runs)
            if os.path.exists(out_path):
                tqdm.write(f"[SKIP] Already exists: '{out_path}'")
                total += 1
                if label == "bird":
                    bird_total += 1
                else:
                    noise_total += 1
                per_species.setdefault(species_name, {"bird": 0, "noise": 0})
                per_species[species_name][label] += 1
                continue

            sf.write(out_path, segment, sr)

            total += 1
            if label == "bird":
                bird_total += 1
                per_species.setdefault(species_name, {"bird": 0, "noise": 0})
                per_species[species_name]["bird"] += 1
            else:
                noise_total += 1
                per_species.setdefault(species_name, {"bird": 0, "noise": 0})
                per_species[species_name]["noise"] += 1

    bird_pct  = bird_total  / total * 100 if total else 0
    noise_pct = noise_total / total * 100 if total else 0

    print(f"\n{'-'*56}")
    print("  TRANSFORM SUMMARY")
    print(f"{'-'*56}")
    print(f"  Total segments : {total}")
    print(f"  Bird           : {bird_total}  ({bird_pct:.1f}%)")
    print(f"  Noise          : {noise_total}  ({noise_pct:.1f}%)")
    print(f"{'-'*56}")
    # Compact machine-readable summary line
    print(f"[TRANSFORM] bird: {bird_total} | noise: {noise_total}\n")

    # Sanity check -- IBC53 should yield ~14,000+ segments
    EXPECTED_MIN = 10_000
    if total < EXPECTED_MIN:
        print(
            f"[WARN] Only {total} segments produced. "
            f"Expected >= {EXPECTED_MIN}. "
            "Some source files may have failed to load -- check [WARN] lines above."
        )

    _write_report(per_species, output_root)

    return {
        "total": total,
        "bird":  bird_total,
        "noise": noise_total,
        "output_root": output_root,
    }


def _write_report(per_species, output_root):
    report_path = os.path.join("data", "transform_report.csv")
    os.makedirs("data", exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["species", "bird_segments", "noise_segments", "bird_ratio"]
        )
        writer.writeheader()
        for species, stats in sorted(per_species.items()):
            total = stats["bird"] + stats["noise"]
            writer.writerow({
                "species":        species,
                "bird_segments":  stats["bird"],
                "noise_segments": stats["noise"],
                "bird_ratio":     round(stats["bird"] / total, 4) if total else 0,
            })
    print(f"[TRANSFORM] Report written to '{report_path}'.")


if __name__ == "__main__":
    transform_dataset()
