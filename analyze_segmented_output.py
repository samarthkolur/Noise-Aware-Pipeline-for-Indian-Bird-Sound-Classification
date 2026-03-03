import os
import sys
import glob
import csv
import random
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Reuse feature functions from segment_audio ───────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segmentation"))
import segment_audio as seg

# ─── Configuration ────────────────────────────────────────────────────────────
SEGMENTED_ROOT = os.path.join("data", "segmented")
CSV_PATH       = os.path.join("data", "post_segmentation_feature_analysis.csv")
PLOT_DIR       = os.path.join("data", "post_segmentation_plots")

FEATURES = [
    "RMS_dB",
    "ZCR",
    "SpectralFlatness",
    "CentroidMean",
    "CentroidStd",
    "AutocorrPeak",
]

LABEL_COLORS = {
    "bird" : ("#4C9BE8", 0.65),   # blue, alpha
    "noise": ("#E8694C", 0.65),   # red-orange, alpha
}

# ─── Argument Parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Analyze feature distributions of bird vs noise segments post-segmentation."
)
parser.add_argument(
    "--max_segments_per_class",
    type=int,
    default=200,
    help="Max number of .wav segments to sample per class (default: 200).",
)
args = parser.parse_args()

# ─── Discover Segments ────────────────────────────────────────────────────────
if not os.path.isdir(SEGMENTED_ROOT):
    raise FileNotFoundError(
        f"[ERROR] Segmented output directory not found: '{SEGMENTED_ROOT}'\n"
        "Run segmentation/segment_audio.py first."
    )

label_files = {}
for label in ("bird", "noise"):
    label_dir = os.path.join(SEGMENTED_ROOT, label)
    if not os.path.isdir(label_dir):
        print(f"[WARN] '{label_dir}' not found — skipping label '{label}'.")
        continue
    files = glob.glob(os.path.join(label_dir, "**", "*.wav"), recursive=True)
    if not files:
        print(f"[WARN] No .wav files found under '{label_dir}' — skipping.")
        continue
    random.seed(42)
    label_files[label] = random.sample(files, min(args.max_segments_per_class, len(files)))
    print(f"[INFO] '{label}': sampling {len(label_files[label])} / {len(files)} segments.")

if not label_files:
    raise RuntimeError("[ERROR] No segments found for any label. Aborting.")


# ─── Feature Extraction ───────────────────────────────────────────────────────
def extract_subframe_features(wav_path):
    """Load a segment, split into 0.5s sub-frames, return list of feature dicts."""
    try:
        y, sr = librosa.load(wav_path, sr=seg.TARGET_SR)
    except Exception as e:
        print(f"[WARN] Could not load {wav_path}: {e}")
        return []

    subframes = seg.split_into_subframes(y, sr, seg.SUB_FRAME_LENGTH)
    rows = []
    for frame in subframes:
        c_mean, c_std = seg.compute_centroid_stats(frame, sr)
        rows.append({
            "RMS_dB"          : seg.compute_rms_db(frame),
            "ZCR"             : seg.compute_zcr(frame),
            "SpectralFlatness": seg.compute_spectral_flatness(frame),
            "CentroidMean"    : c_mean,
            "CentroidStd"     : c_std,
            "AutocorrPeak"    : seg.compute_short_lag_autocorr_peak(
                                    frame, sr,
                                    seg.AUTOCORR_LAG_MIN_MS,
                                    seg.AUTOCORR_LAG_MAX_MS
                                ),
        })
    return rows


all_rows = []   # {label, feature...}
label_data = {label: {f: [] for f in FEATURES} for label in label_files}

for label, files in label_files.items():
    for wav_path in files:
        for row in extract_subframe_features(wav_path):
            row["label"] = label
            all_rows.append(row)
            for feat in FEATURES:
                label_data[label][feat].append(row[feat])

print(f"[INFO] Total sub-frames extracted: {len(all_rows)}")


# ─── Write CSV ────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["label"] + FEATURES)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"[REPORT] CSV saved to: {CSV_PATH}")


# ─── Print Statistics per Label ───────────────────────────────────────────────
for label in label_files:
    n = sum(1 for r in all_rows if r["label"] == label)
    print(f"\n{'─' * 62}")
    print(f"  {label.upper()} STATISTICS  (n={n} sub-frames)")
    print(f"{'─' * 62}")
    print(f"  {'Feature':<22} {'Min':>10} {'Mean':>10} {'Max':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
    for feat in FEATURES:
        vals = label_data[label][feat]
        if vals:
            print(
                f"  {feat:<22} "
                f"{min(vals):>10.4f} "
                f"{float(np.mean(vals)):>10.4f} "
                f"{max(vals):>10.4f}"
            )

print(f"\n{'─' * 62}\n")


# ─── Overlay Histogram Plots ─────────────────────────────────────────────────
os.makedirs(PLOT_DIR, exist_ok=True)

for feat in FEATURES:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    all_vals = [v for label in label_files for v in label_data[label][feat]]
    if not all_vals:
        plt.close(fig)
        continue

    # Shared bin edges so both histograms are directly comparable
    bins = np.linspace(min(all_vals), max(all_vals), 55)

    for label in label_files:
        vals = label_data[label][feat]
        color, alpha = LABEL_COLORS.get(label, ("#888888", 0.6))
        ax.hist(
            vals,
            bins=bins,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.3,
            label=f"{label.capitalize()} (n={len(vals)})",
        )

    # Threshold overlay where relevant
    thresholds = {
        "RMS_dB"          : seg.SILENCE_GATE_DB,
        "ZCR"             : seg.ZCR_MAX,
        "AutocorrPeak"    : seg.AUTOCORR_THRESH,
        "CentroidMean"    : seg.CENTROID_LOW_HZ,
        "CentroidStd"     : seg.CSTD_MAX,
    }
    if feat in thresholds:
        ax.axvline(
            x=thresholds[feat],
            color="#333333",
            linewidth=1.8,
            linestyle="--",
            label=f"V2 threshold = {thresholds[feat]}",
        )

    ax.set_title(
        f"Bird vs Noise — {feat}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel("Sub-frame Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, f"{feat}.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Saved: {plot_path}")

print("\n[DONE] Post-segmentation analysis complete.")
