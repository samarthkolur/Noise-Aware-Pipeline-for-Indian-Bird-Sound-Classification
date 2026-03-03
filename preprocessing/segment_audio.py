import librosa
import soundfile as sf
import numpy as np
import os
import glob
import csv
import argparse
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_ROOT      = "data/IBC53"
OUTPUT_ROOT     = "data/segmented"
SEGMENT_LENGTH  = 3.0    # seconds
SUB_FRAME_LENGTH = 0.5   # seconds
TARGET_SR       = 48000  # Hz
RMS_REF_LEVEL    = 0.05    # target normalized amplitude reference

# ─── Noise Segregation V2 — Tunable Constants ────────────────────────────────
SILENCE_GATE_DB      = -42.0   # sub-frame silence cutoff (dB, normalized space)
CENTROID_LOW_HZ      = 1000.0  # lower bound of expected bird centroid (Hz)
CENTROID_HIGH_HZ     = 10000.0 # upper bound of expected bird centroid (Hz)
AUTOCORR_THRESH      = 0.70    # autocorr peak threshold for insect detection
AUTOCORR_LAG_MIN_MS  = 5.0     # insect lag window start (ms)
AUTOCORR_LAG_MAX_MS  = 20.0    # insect lag window end (ms)
ZCR_MAX              = 0.30    # ZCR normalization ceiling
CSTD_MAX             = 2500.0  # centroid std normalization ceiling (Hz)
w_zcr                = 0.25    # weight: ZCR contribution to noise score
w_flat               = 0.30    # weight: spectral flatness contribution
w_cent               = 0.15    # weight: centroid out-of-band flag
w_cstd               = 0.15    # weight: centroid std (wind) contribution
w_auto               = 0.15    # weight: insect autocorrelation flag
NOISE_SCORE_THRESH   = 0.50    # score >= this → sub-frame is noise
VOTE_THRESHOLD       = 0.50    # bird-vote fraction to classify segment as bird

# ─── Global Counters (Reporting) ─────────────────────────────────────────────
total_segments   = 0
total_bird       = 0
total_noise      = 0
per_species_stats = {}   # species_name -> {total, bird, noise}


# ─── Startup Check ──────────────────────────────────────────────────────
if not os.path.isdir(INPUT_ROOT):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset directory not found: '{INPUT_ROOT}'\n"
        "Please download the dataset first by running:\n"
        "    python download_ibc53.py\n"
    )


# ─── Waveform Utilities ───────────────────────────────────────────────────────

def normalize_waveform(y, sr):
    """
    Normalize waveform amplitude so the 95th-percentile RMS across
    0.5-second frames equals RMS_REF_LEVEL.
    Eliminates gain-level bias across recordings from different equipment.

    Args:
        y  (np.ndarray): Raw waveform samples.
        sr (int)       : Sample rate in Hz.

    Returns:
        np.ndarray: Level-normalized waveform.
    """
    frame_len   = int(SUB_FRAME_LENGTH * sr)
    n_frames    = len(y) // frame_len
    if n_frames == 0:
        return y  # too short to normalize — return as-is

    frames      = y[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms_values  = np.sqrt(np.mean(frames ** 2, axis=1))
    p95_rms     = np.percentile(rms_values, 95)
    scale       = RMS_REF_LEVEL / (p95_rms + 1e-9)
    return y * scale


def split_into_segments(y, sr, segment_length=SEGMENT_LENGTH):
    """
    Split a waveform into fixed-length non-overlapping segments.
    Trailing audio shorter than segment_length is discarded.

    Args:
        y              (np.ndarray): Waveform samples.
        sr             (int)       : Sample rate in Hz.
        segment_length (float)     : Segment duration in seconds.

    Returns:
        list[np.ndarray]: List of equal-length segment arrays.
    """
    samples_per_segment = int(segment_length * sr)
    n_segments          = len(y) // samples_per_segment
    return [
        y[i * samples_per_segment : (i + 1) * samples_per_segment]
        for i in range(n_segments)
    ]


def split_into_subframes(segment, sr, subframe_length=SUB_FRAME_LENGTH):
    """
    Divide a single segment into fixed-length sub-frames.
    Trailing samples shorter than subframe_length are discarded.

    Args:
        segment         (np.ndarray): Segment waveform samples.
        sr              (int)       : Sample rate in Hz.
        subframe_length (float)     : Sub-frame duration in seconds.

    Returns:
        list[np.ndarray]: List of equal-length sub-frame arrays.
    """
    samples_per_frame = int(subframe_length * sr)
    n_frames          = len(segment) // samples_per_frame
    return [
        segment[i * samples_per_frame : (i + 1) * samples_per_frame]
        for i in range(n_frames)
    ]


# ─── Feature Extraction (Noise Segregation V2) ───────────────────────────────

def compute_rms_db(frame):
    """
    Compute RMS energy of a frame in decibels.

    Args:
        frame (np.ndarray): Sub-frame waveform samples.

    Returns:
        float: RMS level in dB. Returns -120.0 for silent/zero frames.
    """
    rms = np.sqrt(np.mean(frame ** 2))
    return 20.0 * np.log10(rms + 1e-9)


def compute_zcr(frame):
    """
    Compute mean zero-crossing rate of a frame.

    Args:
        frame (np.ndarray): Sub-frame waveform samples.

    Returns:
        float: Mean ZCR in crossings per sample (range ~0.0 – 1.0).
    """
    return float(librosa.feature.zero_crossing_rate(frame).mean())


def compute_spectral_flatness(frame):
    """
    Compute mean spectral flatness of a frame.
    0.0 = perfectly tonal (pure sine); 1.0 = white noise.

    Args:
        frame (np.ndarray): Sub-frame waveform samples.

    Returns:
        float: Mean spectral flatness (range 0.0 – 1.0).
    """
    return float(librosa.feature.spectral_flatness(y=frame).mean())


def compute_centroid_stats(frame, sr):
    """
    Compute mean and standard deviation of spectral centroid across
    short-time frames within the sub-frame.

    Args:
        frame (np.ndarray): Sub-frame waveform samples.
        sr    (int)       : Sample rate in Hz.

    Returns:
        tuple[float, float]: (mean_centroid_hz, std_centroid_hz)
    """
    centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
    return float(centroid.mean()), float(centroid.std())


def compute_short_lag_autocorr_peak(frame, sr, lag_min_ms, lag_max_ms):
    """
    Compute the maximum normalized autocorrelation peak within a
    short lag window [lag_min_ms, lag_max_ms].
    High values indicate periodic / insect-like signals.

    Args:
        frame       (np.ndarray): Sub-frame waveform samples.
        sr          (int)       : Sample rate in Hz.
        lag_min_ms  (float)     : Start of lag window in milliseconds.
        lag_max_ms  (float)     : End of lag window in milliseconds.

    Returns:
        float: Maximum normalized autocorrelation in lag window (range 0.0 – 1.0).
    """
    lag_min = int(lag_min_ms * sr / 1000)
    lag_max = int(lag_max_ms * sr / 1000)

    # Full normalized autocorrelation via numpy
    autocorr = np.correlate(frame, frame, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]          # keep non-negative lags
    norm      = autocorr[0] + 1e-9                    # normalize by zero-lag (energy)
    autocorr  = autocorr / norm

    # Guard against lag window exceeding frame length
    lag_max = min(lag_max, len(autocorr) - 1)
    if lag_min >= lag_max:
        return 0.0

    return float(np.max(autocorr[lag_min : lag_max + 1]))


# ─── Classification (Noise Segregation V2) ───────────────────────────────────

def _score_subframe(frame, sr):
    """
    Compute noise score S ∈ [0, 1] for a single 0.5-second sub-frame.
    Returns None if the frame is below the silence gate.

    Args:
        frame (np.ndarray): Sub-frame waveform samples (normalized).
        sr    (int)       : Sample rate in Hz.

    Returns:
        float | None: Noise score, or None for silent frames.
    """
    # Silence gate — fast exit
    if compute_rms_db(frame) < SILENCE_GATE_DB:
        return None

    # Feature extraction
    zcr                       = compute_zcr(frame)
    flatness                  = compute_spectral_flatness(frame)
    mean_centroid, std_centroid = compute_centroid_stats(frame, sr)
    autocorr_peak             = compute_short_lag_autocorr_peak(
                                    frame, sr,
                                    AUTOCORR_LAG_MIN_MS,
                                    AUTOCORR_LAG_MAX_MS
                                )

    # Derived flags
    centroid_flag = 1.0 if (
        mean_centroid < CENTROID_LOW_HZ or mean_centroid > CENTROID_HIGH_HZ
    ) else 0.0
    insect_flag   = 1.0 if autocorr_peak > AUTOCORR_THRESH else 0.0

    # Noise score — high flatness = noise (V1 bug corrected)
    def norm(x, max_val):
        return min(max(x / (max_val + 1e-9), 0.0), 1.0)

    S = (
        w_zcr  * norm(zcr,         ZCR_MAX)
      + w_flat  * norm(flatness,    1.0)
      + w_cent  * centroid_flag
      + w_cstd  * norm(std_centroid, CSTD_MAX)
      + w_auto  * insect_flag
    )
    return S


def classify_segment(segment, sr):
    """
    Classify a 3-second segment as 'bird' or 'noise' using
    Noise Segregation V2: sub-frame majority voting with
    per-file normalized amplitude and signal-processing features.

    Args:
        segment (np.ndarray): Segment waveform samples (normalized).
        sr      (int)       : Sample rate in Hz.

    Returns:
        str: 'bird' or 'noise'.
    """
    subframes   = split_into_subframes(segment, sr, SUB_FRAME_LENGTH)
    bird_votes  = 0
    noise_votes = 0

    for frame in subframes:
        score = _score_subframe(frame, sr)
        if score is None:
            continue                          # silence — excluded from vote
        if score >= NOISE_SCORE_THRESH:
            noise_votes += 1
        else:
            bird_votes += 1

    active_frames = bird_votes + noise_votes

    if active_frames == 0:
        return "noise"                        # fully silent segment

    if bird_votes / active_frames >= VOTE_THRESHOLD:
        return "bird"

    return "noise"


# ─── Core Pipeline ────────────────────────────────────────────────────────────

def segment_audio(file_path, species_name):
    """
    Load one audio file, normalize, segment, classify, and write segments.

    Args:
        file_path    (str): Absolute path to source .wav file.
        species_name (str): Species label (used as output subfolder name).
    """
    y, sr = librosa.load(file_path, sr=TARGET_SR)

    y = normalize_waveform(y, sr)

    segments  = split_into_segments(y, sr, SEGMENT_LENGTH)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for i, segment in enumerate(segments):
        label   = classify_segment(segment, sr)
        out_dir = os.path.join(OUTPUT_ROOT, label, species_name)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"{base_name}_seg{i:03d}.wav")
        sf.write(out_file, segment, sr)

        # ── Update counters ──────────────────────────────────────────────────
        global total_segments, total_bird, total_noise, per_species_stats

        total_segments += 1
        if label == "bird":
            total_bird += 1
        else:
            total_noise += 1

        if species_name not in per_species_stats:
            per_species_stats[species_name] = {"total": 0, "bird": 0, "noise": 0}
        per_species_stats[species_name]["total"] += 1
        per_species_stats[species_name][label]   += 1


def process_all():
    """
    Discover all .wav files under INPUT_ROOT and process each one.
    Prints a summary and writes a CSV report after completion.
    """
    audio_files = glob.glob(f"{INPUT_ROOT}/**/*.wav", recursive=True)
    print(f"Searching inside: {INPUT_ROOT}")
    print(f"Found {len(audio_files)} files")

    for file_path in tqdm(audio_files):
        species_name = os.path.basename(os.path.dirname(file_path))
        segment_audio(file_path, species_name)

    # ── Console Summary ───────────────────────────────────────────────────────
    bird_pct  = (total_bird  / total_segments * 100) if total_segments else 0
    noise_pct = (total_noise / total_segments * 100) if total_segments else 0

    print("\n" + "─" * 56)
    print("  SEGMENTATION SUMMARY")
    print("─" * 56)
    print(f"  Total segments : {total_segments}")
    print(f"  Bird           : {total_bird}  ({bird_pct:.1f}%)")
    print(f"  Noise          : {total_noise}  ({noise_pct:.1f}%)")
    print("─" * 56 + "\n")

    # ── CSV Report ────────────────────────────────────────────────────────────
    report_path = os.path.join("data", "segmentation_report.csv")
    os.makedirs("data", exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["species", "total_segments", "bird_segments",
                        "noise_segments", "bird_ratio"]
        )
        writer.writeheader()
        for species, stats in sorted(per_species_stats.items()):
            bird_ratio = (
                round(stats["bird"] / stats["total"], 4)
                if stats["total"] > 0 else 0.0
            )
            writer.writerow({
                "species"        : species,
                "total_segments" : stats["total"],
                "bird_segments"  : stats["bird"],
                "noise_segments" : stats["noise"],
                "bird_ratio"     : bird_ratio,
            })

    print(f"[REPORT] Segmentation report written to: {report_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    parser = argparse.ArgumentParser(
        description="Segment audio files and classify segments as bird or noise (V2).",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help=f"Root directory containing species subfolders of .wav files. "
             f"Defaults to module constant INPUT_ROOT ('{INPUT_ROOT}').",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=f"Root directory where bird/ and noise/ output folders are written. "
             f"Defaults to module constant OUTPUT_ROOT ('{OUTPUT_ROOT}').",
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        default=None,
        metavar="N",
        help="Calibration mode: sample N sub-frames, print feature statistics "
             "(min/mean/max), then exit — no files written, no classification run.",
    )

    args = parser.parse_args()

    if args.input_root is not None:
        INPUT_ROOT = args.input_root
        print(f"[ARG] INPUT_ROOT overridden to: '{INPUT_ROOT}'")

    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root
        print(f"[ARG] OUTPUT_ROOT overridden to: '{OUTPUT_ROOT}'")

    # ─── Calibration Mode ─────────────────────────────────────────────────────
    if args.calibrate is not None:
        N = args.calibrate
        print(f"[CALIBRATE] Sampling up to {N} sub-frames from '{INPUT_ROOT}' ...")

        all_files = glob.glob(f"{INPUT_ROOT}/**/*.wav", recursive=True)
        if not all_files:
            raise RuntimeError(f"[ERROR] No .wav files found under '{INPUT_ROOT}'.")

        random.seed(42)
        random.shuffle(all_files)

        buckets = {k: [] for k in [
            "RMS_dB", "ZCR", "SpectralFlatness",
            "CentroidMean", "CentroidStd", "AutocorrPeak",
        ]}
        collected = 0

        for file_path in all_files:
            if collected >= N:
                break
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)
            except Exception:
                continue

            y = normalize_waveform(y, sr)

            for segment in split_into_segments(y, sr, SEGMENT_LENGTH):
                if collected >= N:
                    break
                for frame in split_into_subframes(segment, sr, SUB_FRAME_LENGTH):
                    if collected >= N:
                        break
                    c_mean, c_std = compute_centroid_stats(frame, sr)
                    buckets["RMS_dB"].append(compute_rms_db(frame))
                    buckets["ZCR"].append(compute_zcr(frame))
                    buckets["SpectralFlatness"].append(compute_spectral_flatness(frame))
                    buckets["CentroidMean"].append(c_mean)
                    buckets["CentroidStd"].append(c_std)
                    buckets["AutocorrPeak"].append(
                        compute_short_lag_autocorr_peak(
                            frame, sr, AUTOCORR_LAG_MIN_MS, AUTOCORR_LAG_MAX_MS
                        )
                    )
                    collected += 1

        print(f"\n{'─' * 62}")
        print(f"  CALIBRATION REPORT  (n={collected} sub-frames sampled)")
        print(f"{'─' * 62}")
        print(f"  {'Feature':<22} {'Min':>10} {'Mean':>10} {'Max':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
        for feat, vals in buckets.items():
            if vals:
                print(
                    f"  {feat:<22} "
                    f"{min(vals):>10.4f} "
                    f"{float(np.mean(vals)):>10.4f} "
                    f"{max(vals):>10.4f}"
                )
        print(f"{'─' * 62}")
        print("[CALIBRATE] Done. No files written.")

    else:
        process_all()
