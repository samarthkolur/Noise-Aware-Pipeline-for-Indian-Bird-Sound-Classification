"""
Baseline BirdNET Evaluation -- Raw IBC53 Audio

Runs BirdNET (as a black-box inference tool) on every raw .wav file in
data/IBC53/ without any preprocessing.

Ground truth  : folder name (scientific species name, underscore -> space)
BirdNET input : full-length raw recording
Top prediction: highest-confidence detection across all 3-second windows

Output: results/baseline_predictions.json
"""

import os
import sys
import glob
import json
import warnings
from datetime import datetime

from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)  # pydub/ffmpeg noise

# -- BirdNET -------------------------------------------------------------------
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# -- Config --------------------------------------------------------------------
INPUT_ROOT   = os.path.join("data", "IBC53")
OUTPUT_FILE  = os.path.join("results", "baseline_predictions.json")

# India centre coordinates -- boosts confidence for Indian species
LATITUDE     = 20.0
LONGITUDE    = 78.0
ANALYSIS_DATE = datetime(year=2023, month=6, day=15)   # peak Indian bird season
MIN_CONF     = 0.1   # low threshold to capture all detections for evaluation


def get_top_prediction(detections):
    """Return the highest-confidence detection, or None if list is empty."""
    if not detections:
        return None
    return max(detections, key=lambda d: d["confidence"])


def normalize_species(folder_name):
    """Convert folder name like 'Corvus_splendens' -> 'corvus splendens'."""
    return folder_name.replace("_", " ").lower()


def run_baseline(input_root=INPUT_ROOT, output_file=OUTPUT_FILE):
    if not os.path.isdir(input_root):
        print(f"[ERROR] Input directory not found: '{input_root}'")
        print("Run:  python download_ibc53.py  first.")
        sys.exit(1)

    wav_files = glob.glob(os.path.join(input_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"[ERROR] No .wav files found under '{input_root}'.")
        sys.exit(1)

    print(f"[BASELINE] Loading BirdNET model ...")
    analyzer = Analyzer()
    print(f"[BASELINE] Model ready. Analysing {len(wav_files)} raw files ...")
    print(f"[BASELINE] Input: '{input_root}'")
    print(f"[BASELINE] Output: '{output_file}'\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []

    for file_path in tqdm(wav_files, desc="Baseline BirdNET"):
        species_folder  = os.path.basename(os.path.dirname(file_path))
        gt_species      = normalize_species(species_folder)

        try:
            recording = Recording(
                analyzer,
                file_path,
                lat=LATITUDE,
                lon=LONGITUDE,
                date=ANALYSIS_DATE,
                min_conf=MIN_CONF,
            )
            recording.analyze()
            detections = recording.detections
        except Exception as exc:
            detections = []
            tqdm.write(f"[WARN] Failed on '{file_path}': {exc}")

        top = get_top_prediction(detections)
        top_sci  = top["scientific_name"].lower() if top else None
        top_conf = top["confidence"] if top else None
        correct  = (top_sci == gt_species) if top_sci else False

        results.append({
            "file":             os.path.normpath(file_path),
            "ground_truth":     gt_species,
            "detections":       [
                {
                    "scientific_name": d["scientific_name"],
                    "common_name":     d["common_name"],
                    "confidence":      round(d["confidence"], 4),
                    "start_time":      d["start_time"],
                    "end_time":        d["end_time"],
                }
                for d in detections
            ],
            "top_prediction":   top_sci,
            "top_confidence":   round(top_conf, 4) if top_conf else None,
            "correct":          correct,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_det  = sum(1 for r in results if not r["detections"])
    acc     = correct / total * 100 if total else 0

    print(f"\n{'-'*56}")
    print("  BASELINE SUMMARY")
    print(f"{'-'*56}")
    print(f"  Files analysed   : {total}")
    print(f"  Correct (top-1)  : {correct}  ({acc:.1f}%)")
    print(f"  No detections    : {no_det}")
    print(f"  Output           : '{output_file}'")
    print(f"{'-'*56}\n")


if __name__ == "__main__":
    run_baseline()
