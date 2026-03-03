# Noise-Aware Preprocessing Pipeline for Indian Bird Sound Classification

A lightweight, rule-based preprocessing pipeline designed to segregate meaningful
bird audio from environmental noise before fine-tuning a pre-trained bioacoustic
classifier on regional Indian bird species from the IBC53 dataset.

---

## 1. Project Overview

Pre-trained bioacoustic models such as BirdNET are trained on curated, clean
recordings and lack robustness to the diverse noise conditions present in
field-recorded datasets. When fine-tuned directly on raw segmented audio, these
models receive training examples labeled as bird species that actually contain
wind, insect stridulation, or silence, resulting in inflated false positive rates
and degraded generalization.

This project addresses the problem of **training dataset contamination** by
introducing a signal-processing-based noise segregation stage that routes each
3-second audio segment to either a `bird/` or `noise/` folder before any model
training begins. The pipeline is fully rule-based, computationally efficient,
and requires no additional labels or neural components.

---

## 2. Dataset Description

**IBC53 — Indian Bird Call Dataset**

| Property | Value |
|----------|-------|
| Source | Kaggle: `arghyasahoo/ibc53-indian-bird-call-dataset` |
| Species | 53 Indian bird species |
| Total recordings | 1,368 WAV files |
| Recording conditions | Field recordings, varying equipment, background noise |
| Audio format | Variable sample rates, mono and stereo |

The dataset contains real-world field recordings where background noise is common.
Insect stridulation, wind, and ambient environmental sounds frequently overlap
bird vocalizations. Recording gain levels vary significantly between files and
between species folders.

---

## 3. Pipeline Architecture

```
data/IBC53/<species>/*.wav
        |
        v
[1] librosa.load(sr=48000)          -- Resample all recordings to 48 kHz
        |
        v
[2] normalize_waveform()            -- Per-file amplitude normalization
        |                              (95th-percentile RMS -> 0.05 target)
        v
[3] split_into_segments()           -- Fixed 3-second, non-overlapping windows
        |                              (144,000 samples/segment at 48 kHz)
        v
[4] classify_segment() [V2]         -- Sub-frame majority voting
        |
        +-- split_into_subframes()  -- 6 x 0.5s sub-frames per segment
        |                              (24,000 samples/frame at 48 kHz)
        |
        +-- _score_subframe()       -- Per-frame noise score S in [0, 1]
        |       RMS_dB              -- Silence gate: discard if < -42.0 dB
        |       ZCR                 -- Zero-crossing rate (wind/broadband noise)
        |       SpectralFlatness    -- Tonal vs noise-like spectrum
        |       CentroidMean        -- Out-of-band flag ([1000, 10000] Hz)
        |       CentroidStd         -- Variance (wind instability)
        |       AutocorrPeak        -- Insect stridulation (lag 5-20 ms)
        |
        +-- Majority vote           -- >= 50% bird sub-frames => bird
        |
        v
data/segmented/bird/<species>/      -- Clean bird segments for training
data/segmented/noise/<species>/     -- Rejected noise segments
```

### Noise Score Formula

Each active (non-silent) sub-frame is scored:

```
S = w_zcr * norm(ZCR, 0.30)
  + w_flat * norm(Flatness, 1.0)
  + w_cent * centroid_out_of_band_flag
  + w_cstd * norm(CentroidStd, 2500)
  + w_auto * insect_flag

Weights: w_zcr=0.25, w_flat=0.30, w_cent=0.15, w_cstd=0.15, w_auto=0.15
Sum of weights = 1.0
```

A sub-frame with S >= 0.50 votes noise. A segment is classified as noise if
fewer than 50% of its active sub-frames vote bird. Silent sub-frames (below
the silence gate) are excluded from voting. A fully silent segment is labeled
noise.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Per-file normalization | Removes recording-equipment gain bias across species |
| Sub-frame voting | Handles mixed segments robustly; single outlier sub-frames don't dominate |
| Relative silence gate | Operates in normalized amplitude space, not raw dB |
| Autocorr lag window 5-20 ms | Targets insect stridulation repetition rate (50-200 Hz), not carrier frequency |
| Flatness direction: high = noise | Corrects a V1 inversion bug; tonal signals (low flatness) are preserved as bird |

---

## 4. Statistical Calibration Process

Before running full segmentation, feature distributions are analyzed across
a random sample of the dataset to ensure thresholds are data-driven rather
than arbitrary.

**Calibration tools:**

```bash
# Inline quick calibration (no files written)
python segmentation/segment_audio.py --calibrate 500

# Full feature distribution analysis with CSV and histogram plots
python calibrate_features.py --max_files 50
```

**Calibration results (n=2,952 sub-frames, 50 files):**

| Feature | Min | Mean | Max | V2 Threshold |
|---------|-----|------|-----|--------------|
| RMS_dB | -98.07 | -37.24 | -16.44 | -42.0 (gate) |
| ZCR | 0.001 | 0.122 | 0.491 | 0.30 (ceiling) |
| SpectralFlatness | 0.0000 | 0.010 | 0.676 | 0.50 (midpoint) |
| CentroidMean (Hz) | 541.96 | 4061.83 | 11308.62 | [1000, 10000] |
| CentroidStd (Hz) | 58.63 | 481.41 | 3858.14 | 2500.0 (ceiling) |
| AutocorrPeak | 0.009 | 0.361 | 0.972 | 0.70 (threshold) |

Calibration outputs are saved to `data/feature_distribution.csv` and
`data/feature_plots/` for review before committing to a threshold configuration.

---

## 5. Segmentation Results

Full segmentation was run over all 1,368 IBC53 recordings.

| Metric | Value |
|--------|-------|
| Total source files | 1,368 |
| Total segments produced | 14,535 |
| Bird segments | 13,256 (91.2%) |
| Noise segments | 1,279 (8.8%) |
| Output location | `data/segmented/` |
| Per-species report | `data/segmentation_report.csv` |

The 8.8% noise rejection rate removes segments containing predominantly
wind, insect noise, or silence while preserving the vast majority of
bird-containing segments including distant and low-energy calls.

Post-segmentation feature separation can be analyzed using:

```bash
python analyze_segmented_output.py --max_segments_per_class 500
```

This generates overlay histograms comparing bird vs noise segment feature
distributions, saved to `data/post_segmentation_plots/`.

---

## 6. Experimental Plan

Three BirdNET fine-tuning experiments are planned to quantify the impact
of the noise segregation pipeline:

| Experiment | Dataset | Labels | Description |
|------------|---------|--------|-------------|
| A — Baseline | Raw 3-second segments (no filtering) | 53 species | Upper bound on noise contamination |
| B — V2 Clean | `data/segmented/bird/` only | 53 species | Effect of noise removal on species accuracy |
| C — V2 + Noise Class | `data/segmented/bird/` + sampled `noise/` | 53 species + noise | BirdNET learns to abstain on noise inputs |

**Training configuration (all experiments):**
- Model: BirdNET pre-trained feature extractor
- Frozen layers: All convolutional blocks
- Fine-tuned layer: Classification head only (linear probing)
- Optimizer: Adam, lr=1e-3, weight decay=1e-4
- Epochs: 20, batch size: 32, early stopping (patience=5)

**Evaluation metrics:**
- Top-1 Accuracy, Macro F1, Weighted F1
- Per-species Recall (confusion matrix)
- False Positive Rate on pure-noise audio
- Accuracy degradation at varying SNR levels (0, 5, 10 dB)

---

## 7. Repository Structure

```
Noise-Aware-Pipeline-for-Indian-Bird-Sound-Classification/
|
+-- segmentation/
|   +-- segment_audio.py          # Core pipeline: normalize, segment, classify (V2)
|
+-- download_ibc53.py              # Dataset download, validation, auto-flatten
+-- calibrate_features.py          # Pre-segmentation feature distribution analysis
+-- analyze_segmented_output.py    # Post-segmentation bird vs noise comparison
+-- create_train_val_split.py      # Train/validation split (80/20)
+-- baseline_test.py               # BirdNET baseline inference test
+-- requirements.txt               # Python dependencies
+-- .gitignore                     # Excludes data/, audio files, venv/
+-- README.md
|
+-- data/                          # [gitignored]
    +-- IBC53/                     # Raw dataset (53 species, 1368 files)
    +-- segmented/
    |   +-- bird/                  # Accepted segments (91.2%)
    |   +-- noise/                 # Rejected segments (8.8%)
    +-- feature_distribution.csv   # Calibration feature stats
    +-- feature_plots/             # Pre-segmentation histograms
    +-- post_segmentation_plots/   # Bird vs noise comparison histograms
    +-- segmentation_report.csv    # Per-species routing summary
```

---

## 8. Setup Instructions

**Prerequisites:** Python 3.9+, Kaggle API credentials

```bash
# 1. Clone the repository
git clone <repo-url>
cd Noise-Aware-Pipeline-for-Indian-Bird-Sound-Classification

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Kaggle credentials
# Place kaggle.json at: C:\Users\<you>\.kaggle\kaggle.json
# (Download from https://www.kaggle.com/account -> Create New API Token)

# 5. Download and validate dataset
python download_ibc53.py

# 6. (Optional) Calibrate thresholds before segmentation
python segmentation/segment_audio.py --calibrate 500

# 7. Run full segmentation pipeline
python segmentation/segment_audio.py

# 8. Run train/validation split
python create_train_val_split.py

# 9. (Optional) Analyze post-segmentation feature distributions
python analyze_segmented_output.py --max_segments_per_class 500
```

---

## 9. Future Work

1. **Ground truth evaluation:** Manually annotate a stratified sample of ~400
   segments and compute Noise Detection Precision, Recall, F1, and Bird
   Preservation Rate. Apply McNemar's test to confirm statistical significance
   of V2 improvements over the no-filtering baseline.

2. **Threshold tuning via grid search:** Systematically vary `NOISE_SCORE_THRESH`
   and `VOTE_THRESHOLD` against the ground truth subset to identify the
   Precision-Recall operating point that maximises bird preservation while
   maintaining noise rejection.

3. **BirdNET fine-tuning experiments:** Execute Experiments A, B, and C as
   defined in Section 6 and compare Top-1 Accuracy, Macro F1, and False
   Positive Rate on pure-noise test audio.

4. **Robustness testing:** Evaluate model accuracy degradation by mixing
   species recordings with wind and insect noise at 0, 5, and 10 dB SNR.

5. **Deep learning segregation:** Replace the rule-based classifier with a
   lightweight binary CNN (bird vs noise) trained on the V2-labeled segments
   as a supervised alternative, and measure whether it improves upon V2 on
   the ground truth subset.

---

*Minor Project — B.E. / B.Tech Computer Science*
*Dataset: IBC53 Indian Bird Call Dataset (Kaggle)*
*Classifier: BirdNET (pre-trained, Kahl et al.)*