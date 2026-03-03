import os
import sys
import glob
import csv
import random
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on Windows without display
import matplotlib.pyplot as plt

# ─── Reuse feature functions from segment_audio ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment_audio as seg

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_ROOT   = "data/IBC53"
CSV_PATH     = os.path.join("data", "feature_distribution.csv")
PLOT_DIR     = os.path.join("data", "feature_plots")

FEATURES = [
    "RMS_dB",
    "ZCR",
    "SpectralFlatness",
    "CentroidMean",
    "CentroidStd",
    "AutocorrPeak",
]

THRESHOLDS = {
    "RMS_dB"         : seg.SILENCE_GATE_DB,
    "ZCR"            : seg.ZCR_MAX,
    "SpectralFlatness": 0.5,             # midpoint of [0, 1]
    "CentroidMean"   : seg.CENTROID_LOW_HZ,
    "CentroidStd"    : seg.CSTD_MAX,
    "AutocorrPeak"   : seg.AUTOCORR_THRESH,
}


# ─── Argument Parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Analyze feature distributions across IBC53 sub-frames."
)
parser.add_argument(
    "--max_files",
    type=int,
    default=50,
    help="Number of .wav files to randomly sample from data/IBC53 (default: 50).",
)
args = parser.parse_args()


# ─── Discover and Sample Files ────────────────────────────────────────────────
if not os.path.isdir(INPUT_ROOT):
    raise FileNotFoundError(
        f"[ERROR] Dataset not found at '{INPUT_ROOT}'. "
        "Run download_ibc53.py first."
    )

all_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.wav"), recursive=True)
if not all_files:
    raise RuntimeError(f"[ERROR] No .wav files found under '{INPUT_ROOT}'.")

random.seed(42)
sampled = random.sample(all_files, min(args.max_files, len(all_files)))
print(f"[INFO] Sampling {len(sampled)} files from {len(all_files)} total.")


# ─── Feature Extraction ───────────────────────────────────────────────────────
rows = []

for file_path in sampled:
    try:
        y, sr = librosa.load(file_path, sr=seg.TARGET_SR)
    except Exception as e:
        print(f"[WARN] Could not load {file_path}: {e}")
        continue

    y       = seg.normalize_waveform(y, sr)
    segments = seg.split_into_segments(y, sr, seg.SEGMENT_LENGTH)

    for segment in segments:
        subframes = seg.split_into_subframes(segment, sr, seg.SUB_FRAME_LENGTH)
        for frame in subframes:
            rms_db    = seg.compute_rms_db(frame)
            zcr       = seg.compute_zcr(frame)
            flatness  = seg.compute_spectral_flatness(frame)
            c_mean, c_std = seg.compute_centroid_stats(frame, sr)
            autocorr  = seg.compute_short_lag_autocorr_peak(
                            frame, sr,
                            seg.AUTOCORR_LAG_MIN_MS,
                            seg.AUTOCORR_LAG_MAX_MS
                        )
            rows.append({
                "RMS_dB"         : rms_db,
                "ZCR"            : zcr,
                "SpectralFlatness": flatness,
                "CentroidMean"   : c_mean,
                "CentroidStd"    : c_std,
                "AutocorrPeak"   : autocorr,
            })

print(f"[INFO] Extracted features from {len(rows)} sub-frames.")


# ─── Write CSV ────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FEATURES)
    writer.writeheader()
    writer.writerows(rows)

print(f"[REPORT] Feature distribution saved to: {CSV_PATH}")


# ─── Print Statistics ─────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print(f"  FEATURE STATISTICS  (n={len(rows)} sub-frames)")
print(f"{'─' * 60}")
print(f"  {'Feature':<20} {'Min':>10} {'Mean':>10} {'Max':>10}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")

for feat in FEATURES:
    vals = [r[feat] for r in rows]
    print(f"  {feat:<20} {min(vals):>10.4f} {np.mean(vals):>10.4f} {max(vals):>10.4f}")

print(f"{'─' * 60}\n")


# ─── Generate Histogram Plots ────────────────────────────────────────────────
os.makedirs(PLOT_DIR, exist_ok=True)

for feat in FEATURES:
    vals      = [r[feat] for r in rows]
    threshold = THRESHOLDS.get(feat)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=60, color="#4C9BE8", edgecolor="white", linewidth=0.4, alpha=0.85)

    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="#E84C4C",
            linewidth=2,
            linestyle="--",
            label=f"Threshold = {threshold}",
        )
        ax.legend(fontsize=10)

    ax.set_title(f"Feature Distribution — {feat}", fontsize=13, fontweight="bold")
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel("Sub-frame Count", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, f"{feat}.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Saved: {plot_path}")

print("\n[DONE] Calibration analysis complete.")
