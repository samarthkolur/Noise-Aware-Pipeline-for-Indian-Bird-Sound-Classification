"""
Master Pipeline Orchestrator

Runs the complete ETL + evaluation pipeline end-to-end:

  Step 1  EXTRACT   -- verify / unzip IBC53 dataset
  Step 2  TRANSFORM -- preprocess audio into data/processed/
  Step 3  LOAD      -- build splits/train.csv and splits/val.csv
  Step 4  BASELINE  -- run BirdNET on raw IBC53 files
  Step 5  PROCESSED -- run BirdNET on preprocessed segments
  Step 6  METRICS   -- compute metrics and comparison

Usage:
    python run_pipeline.py               # run all steps
    python run_pipeline.py --from 4      # resume from step 4
    python run_pipeline.py --only 6      # run step 6 only
"""

import argparse
import glob
import os
import sys
import time


def _banner(step, title):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  STEP {step}: {title}")
    print(f"{bar}\n")


def step1_extract():
    _banner(1, "EXTRACT -- Verify / unzip IBC53 dataset")
    from etl.extract import extract_dataset
    extract_dataset()


def step2_transform():
    _banner(2, "TRANSFORM -- Preprocess audio -> data/processed/")
    from etl.transform import transform_dataset
    transform_dataset()


def step3_load():
    _banner(3, "LOAD -- Build train/val CSV splits")
    from etl.load import load_splits
    load_splits()


def _validate_processed(processed_dir="data/processed", min_segments=10_000):
    """
    Guard before BirdNET inference steps.
    Checks: directory exists, species + noise folders present, no empty folders,
    sufficient total segments.
    Raises RuntimeError on any failure.
    """
    if not os.path.isdir(processed_dir):
        raise RuntimeError(
            f"[VALIDATE] '{processed_dir}' not found. "
            "Run Step 2 (Transform) first."
        )

    folders = [
        f for f in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, f))
    ]
    if not folders:
        raise RuntimeError(
            f"[VALIDATE] '{processed_dir}' is empty. "
            "Run Step 2 (Transform) first."
        )

    if "noise" not in folders:
        raise RuntimeError(
            f"[VALIDATE] No 'noise' subfolder in '{processed_dir}'. "
            "Transform may be incomplete."
        )

    empty = [
        f for f in folders
        if not os.listdir(os.path.join(processed_dir, f))
    ]
    if empty:
        raise RuntimeError(
            f"[VALIDATE] Empty folders found in '{processed_dir}': {empty}. "
            "Re-run Step 2."
        )

    total = len(glob.glob(os.path.join(processed_dir, "**", "*.wav"), recursive=True))
    print(f"[VALIDATE] {len(folders)} folders, {total} total segments in '{processed_dir}'.")

    if total < min_segments:
        raise RuntimeError(
            f"[VALIDATE] Only {total} segments found -- expected >= {min_segments}. "
            "Step 2 (Transform) may not have finished. Re-run before proceeding."
        )

    print("[VALIDATE] Processed dataset looks good -- proceeding to BirdNET inference.")


def step4_baseline():
    _banner(4, "BASELINE -- BirdNET on raw IBC53 audio")
    _validate_processed()
    from run_baseline_birdnet import run_baseline
    run_baseline()


def step5_processed():
    _banner(5, "PROCESSED -- BirdNET on preprocessed segments")
    from run_processed_birdnet import run_processed
    run_processed()


def step6_metrics():
    _banner(6, "METRICS -- Compute metrics + comparison + plots")
    from evaluate_metrics import evaluate_all
    evaluate_all()


STEPS = {
    1: ("Extract",   step1_extract),
    2: ("Transform", step2_transform),
    3: ("Load",      step3_load),
    4: ("Baseline",  step4_baseline),
    5: ("Processed", step5_processed),
    6: ("Metrics",   step6_metrics),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the full BirdNET ETL evaluation pipeline."
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        metavar="N",
        help="Start from step N (default: 1 = beginning).",
    )
    parser.add_argument(
        "--only", dest="only_step", type=int, default=None,
        metavar="N",
        help="Run only step N (overrides --from).",
    )
    args = parser.parse_args()

    if args.only_step is not None:
        steps_to_run = [args.only_step]
    else:
        steps_to_run = list(range(args.from_step, max(STEPS) + 1))

    print("\n" + "=" * 60)
    print("  NOISE-AWARE BIRDNET PIPELINE")
    print("  IBC53 Indian Bird Sound Classification")
    print("=" * 60)
    print(f"  Steps to run: {steps_to_run}")

    t_total = time.time()
    failed  = False

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"[ERROR] Unknown step: {step_num}. Valid steps: 1-{max(STEPS)}.")
            sys.exit(1)

        name, fn = STEPS[step_num]
        t_step   = time.time()
        try:
            fn()
        except Exception as exc:
            print(f"\n[ERROR] Step {step_num} ({name}) failed: {exc}")
            import traceback
            traceback.print_exc()
            failed = True
            break

        elapsed = time.time() - t_step
        print(f"[DONE] Step {step_num} ({name}) completed in {elapsed:.1f}s")

    total_elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    if failed:
        print("  PIPELINE FAILED -- see error above")
    else:
        print(f"  PIPELINE COMPLETE -- {total_elapsed:.1f}s total")
        print("")
        print("  Output files:")
        print("    results/baseline_predictions.json")
        print("    results/processed_predictions.json")
        print("    results/baseline_metrics.json")
        print("    results/processed_metrics.json")
        print("    results/comparison.json")
        print("    results/plots/metrics_bar_chart.png")
        print("    results/plots/confusion_matrix_baseline.png")
        print("    results/plots/confusion_matrix_processed.png")
        print("    splits/train.csv")
        print("    splits/val.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
