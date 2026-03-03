import os
import subprocess
import zipfile
import glob

# ─── Configuration ────────────────────────────────────────────────────────────
KAGGLE_DATASET    = "arghyasahoo/ibc53-indian-bird-call-dataset"
RAW_DIR           = os.path.join("data", "raw")
EXTRACT_DIR       = os.path.join("data", "IBC53")
ZIP_NAME          = "ibc53-indian-bird-call-dataset.zip"
ZIP_PATH          = os.path.join(RAW_DIR, ZIP_NAME)
MIN_ZIP_BYTES     = 1 * 1024 * 1024 * 1024   # 1 GB minimum
MIN_SPECIES       = 10
MIN_WAV_FILES     = 100


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fail(msg):
    """Raise RuntimeError with a clearly formatted message."""
    raise RuntimeError(f"\n{'─' * 60}\n{msg}\n{'─' * 60}")


def _flatten_if_nested(base_dir):
    """
    Kaggle sometimes wraps extracted contents in a single extra folder.
    If base_dir contains exactly one subdirectory and it contains further
    subdirectories, promote all contents one level up and remove the wrapper.

    Example:
        data/IBC53/ibc53-dataset/Corvus_splendens/  →  data/IBC53/Corvus_splendens/

    Safe to call on already-correct structure (no-op if ≥ 2 entries exist).
    """
    import shutil

    entries = [
        e for e in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, e))
    ]

    # Only flatten when there is exactly one folder and it looks like a wrapper
    if len(entries) != 1:
        return

    wrapper     = os.path.join(base_dir, entries[0])
    inner_items = os.listdir(wrapper)

    if not inner_items:
        return  # empty wrapper — leave it, validation will catch it

    print(f"[INFO] Detected nested folder structure — flattening '{wrapper}' ...")

    for item in inner_items:
        src = os.path.join(wrapper, item)
        dst = os.path.join(base_dir, item)
        if os.path.exists(dst):
            # Merge directories rather than overwrite
            if os.path.isdir(src) and os.path.isdir(dst):
                for sub in os.listdir(src):
                    shutil.move(os.path.join(src, sub), os.path.join(dst, sub))
                os.rmdir(src)
            else:
                shutil.move(src, dst)
        else:
            shutil.move(src, dst)

    os.rmdir(wrapper)
    print(f"[OK] Flattened — wrapper '{entries[0]}' removed.")


def validate_dataset():
    """
    Confirm data/IBC53/ exists, flatten any Kaggle wrapper folder,
    then verify minimum content thresholds.
    Raises RuntimeError with a clear message on any failure.
    """
    if not os.path.isdir(EXTRACT_DIR):
        _fail(
            f"[VALIDATION FAILED] Expected directory not found: '{EXTRACT_DIR}'"
        )

    # Auto-flatten single-wrapper Kaggle structure before checking contents
    _flatten_if_nested(EXTRACT_DIR)

    species_dirs = sorted([
        d for d in os.listdir(EXTRACT_DIR)
        if os.path.isdir(os.path.join(EXTRACT_DIR, d))
    ])
    if len(species_dirs) < MIN_SPECIES:
        _fail(
            f"[VALIDATION FAILED] Expected ≥ {MIN_SPECIES} species folders, "
            f"found {len(species_dirs)} in '{EXTRACT_DIR}'.\n"
            f"  Contents: {os.listdir(EXTRACT_DIR)}"
        )

    wav_files = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.wav"), recursive=True)
    if len(wav_files) < MIN_WAV_FILES:
        _fail(
            f"[VALIDATION FAILED] Expected ≥ {MIN_WAV_FILES} .wav files, "
            f"found {len(wav_files)} in '{EXTRACT_DIR}'."
        )

    examples = species_dirs[:3]
    print("\n─── Dataset Validation ───────────────────────────────")
    print(f"  [PASS] Species folders : {len(species_dirs)}")
    print(f"  [PASS] Total .wav files: {len(wav_files)}")
    print(f"  [INFO] Example species : {', '.join(examples)}")
    print("──────────────────────────────────────────────────────\n")


# ─── Step 1: Verify Kaggle credentials ───────────────────────────────────────
kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")

if not os.path.isfile(kaggle_json):
    _fail(
        "[ERROR] Kaggle API credentials not found.\n"
        "To fix this:\n"
        "  1. Go to https://www.kaggle.com/account\n"
        "  2. Click 'Create New API Token' — downloads kaggle.json\n"
        "  3. Move it to: C:\\Users\\<you>\\.kaggle\\kaggle.json"
    )

print("[OK] Kaggle credentials found.")


# ─── Step 2: Skip if dataset already exists ──────────────────────────────────
if os.path.isdir(EXTRACT_DIR) and os.listdir(EXTRACT_DIR):
    print(f"[SKIP] Dataset already exists at '{EXTRACT_DIR}'. Running validation...")
    validate_dataset()
    exit(0)


# ─── Step 3: Delete known partial download if present ────────────────────────
if os.path.isfile(ZIP_PATH):
    size_mb = os.path.getsize(ZIP_PATH) / 1e6
    print(f"[WARN] Existing zip found ({size_mb:.1f} MB) — checking if complete...")
    if os.path.getsize(ZIP_PATH) < MIN_ZIP_BYTES:
        print(f"[WARN] File is below {MIN_ZIP_BYTES / 1e9:.0f} GB threshold. "
              f"Deleting partial download and re-downloading.")
        os.remove(ZIP_PATH)
    else:
        print("[OK] Existing zip meets size threshold — skipping download.")


# ─── Step 4: Download via Kaggle CLI (streaming output) ──────────────────────
if not os.path.isfile(ZIP_PATH):
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[DOWNLOAD] Fetching '{KAGGLE_DATASET}' into '{RAW_DIR}' ...")
    print("─" * 60)

    # Use Popen to stream live output — avoids pipe buffer deadlock on Windows
    process = subprocess.Popen(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", RAW_DIR],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout stream
        text=True,
        bufsize=1,                  # line-buffered — ensures live output on Windows
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()
    print("─" * 60)

    if process.returncode != 0:
        _fail(
            f"[ERROR] Kaggle CLI exited with code {process.returncode}.\n"
            "Check that:\n"
            "  - kaggle is installed:  pip install kaggle\n"
            "  - You have accepted the dataset terms on Kaggle\n"
            "  - Your internet connection is stable"
        )

    print("[OK] Download process completed.")


# ─── Step 5: Locate zip ───────────────────────────────────────────────────────
if not os.path.isfile(ZIP_PATH):
    zips = glob.glob(os.path.join(RAW_DIR, "*.zip"))
    if not zips:
        _fail(
            f"[ERROR] No .zip file found in '{RAW_DIR}' after download.\n"
            "Kaggle may have saved the file under a different name.\n"
            f"Contents of '{RAW_DIR}': {os.listdir(RAW_DIR)}"
        )
    ZIP_PATH = zips[0]
    print(f"[INFO] Located zip at: {ZIP_PATH}")


# ─── Step 6: Verify minimum file size ────────────────────────────────────────
actual_bytes = os.path.getsize(ZIP_PATH)
actual_mb    = actual_bytes / 1e6

print(f"[SIZE] Downloaded file: {actual_mb:.1f} MB")

if actual_bytes < MIN_ZIP_BYTES:
    _fail(
        f"[ERROR] Downloaded file is too small — likely a partial download.\n"
        f"  Expected : ≥ {MIN_ZIP_BYTES / 1e9:.0f} GB\n"
        f"  Actual   : {actual_mb:.1f} MB\n"
        f"  Path     : {ZIP_PATH}\n"
        "Action: Delete the file manually and re-run this script."
    )

print(f"[PASS] File size meets minimum threshold ({MIN_ZIP_BYTES / 1e9:.0f} GB).")


# ─── Step 7: Verify zip integrity ────────────────────────────────────────────
print("[VERIFY] Checking zip integrity — this may take a moment...")

try:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad_file = zf.testzip()
except zipfile.BadZipFile as exc:
    _fail(
        f"[ERROR] File is not a valid zip archive: {exc}\n"
        f"  Path: {ZIP_PATH}\n"
        "Action: Delete and re-download."
    )

if bad_file:
    _fail(
        f"[ERROR] Zip integrity check failed — corrupt entry: '{bad_file}'\n"
        "Action: Delete the zip and re-download."
    )

print("[PASS] Zip integrity verified.")


# ─── Step 8: Extract ──────────────────────────────────────────────────────────
os.makedirs(EXTRACT_DIR, exist_ok=True)
print(f"[EXTRACT] Extracting to '{EXTRACT_DIR}' ...")

try:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
except Exception as exc:
    _fail(
        f"[ERROR] Extraction failed: {exc}\n"
        "The zip file may be corrupted. Delete it and re-download."
    )

print("[OK] Extraction complete.")


# ─── Step 9: Delete zip only after successful extraction ─────────────────────
os.remove(ZIP_PATH)
print(f"[CLEANUP] Deleted zip: {ZIP_PATH}")


# ─── Step 10: Validate extracted dataset ─────────────────────────────────────
validate_dataset()
