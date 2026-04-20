"""
ETL -- Load Step
Scans data/processed/ and produces:

    splits/train.csv   (80%)
    splits/val.csv     (20%)

Each CSV contains: file_path, label
  - label = species name (scientific, space-separated) for bird segments
  - label = "noise" for noise segments
"""

import os
import glob
import csv
import random

PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR    = "splits"
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42


def _collect_files(processed_dir):
    """
    Walk processed_dir and return list of (file_path, label) tuples.
    Folder name == species (underscore form); converted to space-separated for label.
    """
    records = []

    for folder in sorted(os.listdir(processed_dir)):
        folder_path = os.path.join(processed_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label = "noise" if folder == "noise" else folder.replace("_", " ")

        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        for wav in wav_files:
            records.append((wav, label))

    return records


def _stratified_split(records, train_ratio, seed):
    """Split records 80/20 per label to preserve class balance."""
    from collections import defaultdict
    by_label = defaultdict(list)
    for rec in records:
        by_label[rec[1]].append(rec)

    train, val = [], []
    rng = random.Random(seed)

    for label_records in by_label.values():
        rng.shuffle(label_records)
        n_train = max(1, int(len(label_records) * train_ratio))
        train.extend(label_records[:n_train])
        val.extend(label_records[n_train:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _write_csv(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "label"])
        writer.writeheader()
        for file_path, label in records:
            writer.writerow({"file_path": file_path, "label": label})


def load_splits(processed_dir=PROCESSED_DIR, splits_dir=SPLITS_DIR,
                train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    """
    Build and write train/val CSV splits from processed_dir.
    Returns (train_path, val_path).
    """
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(
            f"[LOAD] Processed directory not found: '{processed_dir}'\n"
            "Run the Transform step first."
        )

    records = _collect_files(processed_dir)
    if not records:
        raise RuntimeError(
            f"[LOAD] No .wav files found under '{processed_dir}'."
        )

    print(f"[LOAD] Found {len(records)} segments across "
          f"{len(set(r[1] for r in records))} labels.")

    train_records, val_records = _stratified_split(records, train_ratio, seed)

    train_path = os.path.join(splits_dir, "train.csv")
    val_path   = os.path.join(splits_dir, "val.csv")

    _write_csv(train_path, train_records)
    _write_csv(val_path,   val_records)

    print(f"[LOAD] Train : {len(train_records)} segments -> '{train_path}'")
    print(f"[LOAD] Val   : {len(val_records)}   segments -> '{val_path}'")

    return train_path, val_path


if __name__ == "__main__":
    load_splits()
