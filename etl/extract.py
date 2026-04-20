"""
ETL -- Extract Step
Ensures IBC53 dataset is extracted and ready at data/IBC53/.
"""

import os
import zipfile
import glob
import shutil

RAW_DIR    = os.path.join("data", "raw")
EXTRACT_DIR = os.path.join("data", "IBC53")

MIN_SPECIES = 10
MIN_WAV     = 100


def _flatten_if_nested(base_dir):
    """Flatten single-wrapper Kaggle structure (one spurious subfolder)."""
    entries = [e for e in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, e))]
    if len(entries) != 1:
        return
    wrapper = os.path.join(base_dir, entries[0])
    inner   = os.listdir(wrapper)
    if not inner:
        return
    print(f"[EXTRACT] Flattening nested folder '{wrapper}' ...")
    for item in inner:
        src = os.path.join(wrapper, item)
        dst = os.path.join(base_dir, item)
        if os.path.exists(dst):
            if os.path.isdir(src) and os.path.isdir(dst):
                for sub in os.listdir(src):
                    shutil.move(os.path.join(src, sub), os.path.join(dst, sub))
                os.rmdir(src)
            else:
                shutil.move(src, dst)
        else:
            shutil.move(src, dst)
    os.rmdir(wrapper)
    print("[EXTRACT] Flatten complete.")


def _validate(extract_dir):
    species = [d for d in os.listdir(extract_dir)
               if os.path.isdir(os.path.join(extract_dir, d))]
    wavs    = glob.glob(os.path.join(extract_dir, "**", "*.wav"), recursive=True)

    if len(species) < MIN_SPECIES:
        raise RuntimeError(
            f"[EXTRACT] Only {len(species)} species folders found in '{extract_dir}'. "
            f"Expected >= {MIN_SPECIES}."
        )
    if len(wavs) < MIN_WAV:
        raise RuntimeError(
            f"[EXTRACT] Only {len(wavs)} .wav files found. Expected >= {MIN_WAV}."
        )
    print(f"[EXTRACT] Validated: {len(species)} species, {len(wavs)} .wav files.")


def extract_dataset(extract_dir=EXTRACT_DIR, raw_dir=RAW_DIR):
    """
    Ensure IBC53 dataset is extracted at `extract_dir`.
    If already present and non-empty, skips extraction.
    If a ZIP is found in `raw_dir`, extracts it.
    Returns the path to the extracted dataset.
    """
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"[EXTRACT] Dataset already at '{extract_dir}' -- skipping.")
        _validate(extract_dir)
        return extract_dir

    zip_files = glob.glob(os.path.join(raw_dir, "*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"[EXTRACT] No ZIP found in '{raw_dir}' and '{extract_dir}' is empty.\n"
            "Run:  python download_ibc53.py"
        )

    zip_path = zip_files[0]
    print(f"[EXTRACT] Extracting '{zip_path}' -> '{extract_dir}' ...")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    _flatten_if_nested(extract_dir)
    _validate(extract_dir)
    print("[EXTRACT] Done.")
    return extract_dir


if __name__ == "__main__":
    path = extract_dataset()
    species = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"[OK] {len(species)} species folders at '{path}'.")
