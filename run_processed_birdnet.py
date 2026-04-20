"""
Processed BirdNET Evaluation -- ETL-Processed Segments

Runs BirdNET on every 3-second segment in data/processed/.
Segments in species folders are labelled as that species.
Segments in the noise/ folder are labelled "noise".

Ground truth  : folder name
                  species folders -> scientific name (underscore -> space)
                  noise/          -> "noise"
BirdNET input : 3-second preprocessed segment
Top prediction: highest-confidence detection (or None = no bird detected)

Output: results/processed_predictions.json
"""

import os
import sys
import glob
import json
import warnings
from datetime import datetime

from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# -- Config --------------------------------------------------------------------
INPUT_ROOT  = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join("results", "processed_predictions.json")

LATITUDE      = 20.0
LONGITUDE     = 78.0
ANALYSIS_DATE = datetime(year=2023, month=6, day=15)
MIN_CONF      = 0.1


def get_top_prediction(detections):
    if not detections:
        return None
    return max(detections, key=lambda d: d["confidence"])


def get_ground_truth(folder_name):
    """
    Return normalised ground-truth label from a folder name.
    'noise' stays 'noise'; species folders are underscore->space, lowercased.
    """
    if folder_name == "noise":
        return "noise"
    return folder_name.replace("_", " ").lower()


def run_processed(input_root=INPUT_ROOT, output_file=OUTPUT_FILE):
    if not os.path.isdir(input_root):
        print(f"[ERROR] Processed directory not found: '{input_root}'")
        print("Run the ETL Transform step first (python -m etl.transform).")
        sys.exit(1)

    wav_files = glob.glob(os.path.join(input_root, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"[ERROR] No .wav files found under '{input_root}'.")
        sys.exit(1)

    print(f"[PROCESSED] Loading BirdNET model ...")
    analyzer = Analyzer()
    print(f"[PROCESSED] Model ready. Analysing {len(wav_files)} segments ...")
    print(f"[PROCESSED] Input: '{input_root}'")
    print(f"[PROCESSED] Output: '{output_file}'\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results      = []
    noise_fp     = 0   # noise segments where BirdNET detected a bird
    noise_tn     = 0   # noise segments with no detection
    noise_total  = 0

    for file_path in tqdm(wav_files, desc="Processed BirdNET"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        gt          = get_ground_truth(folder_name)

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

        top      = get_top_prediction(detections)
        top_sci  = top["scientific_name"].lower() if top else None
        top_conf = top["confidence"] if top else None

        if gt == "noise":
            noise_total += 1
            if top_sci is not None:
                correct   = False
                noise_fp += 1
            else:
                correct   = True   # correctly detected as non-bird
                noise_tn += 1
        else:
            correct = (top_sci == gt) if top_sci else False

        results.append({
            "file":           os.path.normpath(file_path),
            "ground_truth":   gt,
            "is_noise":       gt == "noise",
            "detections":     [
                {
                    "scientific_name": d["scientific_name"],
                    "common_name":     d["common_name"],
                    "confidence":      round(d["confidence"], 4),
                    "start_time":      d["start_time"],
                    "end_time":        d["end_time"],
                }
                for d in detections
            ],
            "top_prediction": top_sci,
            "top_confidence": round(top_conf, 4) if top_conf else None,
            "correct":        correct,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    bird_segs   = [r for r in results if not r["is_noise"]]
    bird_correct = sum(1 for r in bird_segs if r["correct"])
    bird_acc     = bird_correct / len(bird_segs) * 100 if bird_segs else 0
    fpr          = noise_fp / noise_total * 100 if noise_total else 0

    print(f"\n{'-'*60}")
    print("  PROCESSED SUMMARY")
    print(f"{'-'*60}")
    print(f"  Total segments       : {len(results)}")
    print(f"  Bird segments        : {len(bird_segs)}  "
          f"(correct: {bird_correct}, acc: {bird_acc:.1f}%)")
    print(f"  Noise segments       : {noise_total}")
    print(f"  Noise FP (bird detected on noise) : {noise_fp}  (FPR: {fpr:.1f}%)")
    print(f"  Output               : '{output_file}'")
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    run_processed()
