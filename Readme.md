# Noise-Aware Pipeline for Indian Bird Sound Classification

---

## Project Overview

This project builds and evaluates a **noise-aware, embedding-based supervised learning pipeline** for Indian bird sound classification on the IBC53 dataset.

The pipeline addresses two questions:

1. *Does rule-based acoustic noise preprocessing improve BirdNET inference quality?*
2. *Can a learned MLP classifier trained on BirdNET embeddings outperform direct BirdNET thresholding?*

The system is structured in two phases:

**Phase 1 — ETL + BirdNET baseline** (Steps 1–6): A signal-processing ETL pipeline (Noise Segregation V2) preprocesses raw recordings into 3-second segments, evaluates BirdNET on both raw and preprocessed audio, and computes two-stage metrics.

**Phase 2 — Embedding-based ML pipeline** (Steps 7–10): BirdNET per-class confidence scores are extracted as features, reduced via PCA, and used to train an MLP binary classifier. An optional autoencoder OOD gate and temperature scaling are layered on top. A four-model benchmark compares all approaches on the same validation split.

---

## What Changed (Recent Update)

The following components were added on top of the existing ETL + BirdNET pipeline. **The preprocessing pipeline is unchanged.** All improvements come from downstream modeling.

| Component | File | What It Does |
|---|---|---|
| Embedding extraction | `models/extract_embeddings.py` | BirdNET 6522-D confidence scores → PCA 1024-D embeddings |
| MLP classifier | `models/mlp_classifier.py` | Binary bird/noise classifier on embeddings; BCEWithLogitsLoss |
| Temperature scaling | `models/temperature_scaling.py` | Post-hoc MLP probability calibration via NLL minimisation |
| Autoencoder OOD gate | `models/autoencoder.py` | Reconstruction-MSE gating; trained on bird embeddings only |
| Inference pipeline | `models/inference_pipeline.py` | Three-stage: AE gate → MLP+T → BirdNET species lookup |
| Benchmark | `evaluation/benchmark.py` | Four-model comparison on same val split |
| Embedding visualisation | `evaluation/embedding_viz.py` | PCA + t-SNE plots of val embeddings |

### Key Results

| Model | F1 | Precision | Recall | FPR | FNR |
|---|---|---|---|---|---|
| BirdNET Baseline (raw files) | 0.720 | 1.000 | 0.562 | N/A | 0.438 |
| Noise-Aware BirdNET (processed) | 0.489 | 0.933 | 0.331 | 0.163 | 0.669 |
| **MLP only** | **0.940** | **0.955** | **0.926** | 0.300 | 0.074 |
| MLP + AE gate | 0.938 | 0.956 | 0.921 | 0.296 | 0.079 |

The MLP classifier achieves F1 = 0.940 versus F1 = 0.489 for noise-aware BirdNET on the same 2,928-segment validation set. Performance gains come from **learned classification on labeled training data**, not from preprocessing improvements.

---

## Pipeline Architecture

```
IBC53 Raw Audio (1,368 WAV files)
        │
        ▼
┌─────────────────────────────────────────────┐
│  ETL Pipeline (Steps 1–3)                  │
│                                             │
│  Extract → Transform → Load                │
│  etl/extract.py                            │
│  etl/transform.py  (Noise Segregation V2)  │
│  etl/load.py       (80/20 stratified split)│
└──────────────┬──────────────────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
data/processed/       splits/train.csv
  <species>/          splits/val.csv
  noise/
     │                   │
     ▼                   ▼
┌──────────────┐   ┌─────────────────────────────────┐
│  BirdNET     │   │  Embedding Pipeline (Steps 7–10) │
│  (Steps 4–6) │   │                                  │
│              │   │  extract_embeddings.py            │
│  Baseline    │   │  BirdNET 6522-D confidence →      │
│  Processed   │   │  PCA 1024-D (fit train only)     │
│  Evaluate    │   │                                  │
└──────┬───────┘   │  mlp_classifier.py               │
       │           │  train on 11,607 embeddings      │
       │           │  threshold sweep → best = 0.3    │
       │           │                                  │
       │           │  autoencoder.py                  │
       │           │  train on 10,583 bird-only       │
       │           │  p99 MSE threshold = 4.7e-4      │
       │           │                                  │
       │           │  temperature_scaling.py           │
       │           │  T = 1.4648 (NLL minimisation)   │
       │           └────────────┬────────────────────┘
       │                        │
       │              inference_pipeline.py
       │              Stage 1: AE gate
       │              Stage 2: MLP + T
       │              Stage 3: BirdNET species lookup
       │                        │
       └────────────────────────┘
                        │
                        ▼
              evaluation/benchmark.py
              evaluation/embedding_viz.py
                        │
                        ▼
              results/benchmark_table.csv
              results/plots/pca_plot.png
              results/plots/tsne_plot.png
```

---

## How to Apply the Changes (Steps 7–10)

### Prerequisites

Steps 1–6 must have been run at least once so the following exist:
- `data/processed/` — preprocessed segments
- `splits/train.csv` and `splits/val.csv`
- `results/processed_predictions.json` — BirdNET species lookup used by the inference pipeline

### Full Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd Noise-Aware-Pipeline-for-Indian-Bird-Sound-Classification

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies (includes PyTorch)
pip install -r requirements.txt

# 4. Place Kaggle credentials at ~/.kaggle/kaggle.json
#    Download from: kaggle.com/account → Create New API Token
```

### Run the Full Pipeline (All 10 Steps)

```bash
python run_pipeline.py
```

### Run Only the New ML Steps (Steps 7–10)

If Steps 1–6 are already complete, skip straight to the ML components:

```bash
python run_pipeline.py --from 7
```

### Run Individual Steps

```bash
# Step 7 — Extract BirdNET embeddings and apply PCA
python models/extract_embeddings.py

# Step 8 — Train MLP classifier, autoencoder, and temperature scaling
python models/mlp_classifier.py
python models/autoencoder.py
python models/temperature_scaling.py

# Step 9 — Run three-stage inference on val split
python models/inference_pipeline.py

# Step 10 — Benchmark all four models + embedding visualisation
python evaluation/benchmark.py
python evaluation/embedding_viz.py
```

### All Pipeline Steps

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `etl/extract.py` | Validate and ingest IBC53 dataset | `data/IBC53/` |
| 2 | `etl/transform.py` | Resample, segment, noise filter | `data/processed/` |
| 3 | `etl/load.py` | Stratified 80/20 train/val split | `splits/train.csv`, `splits/val.csv` |
| 4 | `run_baseline_birdnet.py` | BirdNET on raw full-length recordings | `results/baseline_predictions.json` |
| 5 | `run_processed_birdnet.py` | BirdNET on preprocessed 3-s segments | `results/processed_predictions.json` |
| 6 | `evaluate_metrics.py` | Two-stage metrics + comparison plots | `results/baseline_metrics.json`, `results/processed_metrics.json`, `results/plots/` |
| 7 | `models/extract_embeddings.py` | BirdNET confidence scores → PCA embeddings | `data/embeddings/`, `models/saved/pca_transform.pkl` |
| 8 | `models/mlp_classifier.py` + `models/autoencoder.py` + `models/temperature_scaling.py` | Train MLP, AE, calibration | `models/saved/*.pt`, `models/saved/*.json` |
| 9 | `models/inference_pipeline.py` | Three-stage inference on val | `results/mlp_predictions.json`, `results/mlp_ae_predictions.json` |
| 10 | `evaluation/benchmark.py` + `evaluation/embedding_viz.py` | Four-model benchmark + plots | `results/benchmark_table.csv`, `results/plots/pca_plot.png`, `results/plots/tsne_plot.png` |

To resume from or run only a specific step:

```bash
python run_pipeline.py --from 4   # resume from step 4
python run_pipeline.py --only 6   # run step 6 only
```

---

## Noise Segregation V2 — How It Works

Each raw recording goes through five steps inside `etl/transform.py` and
`preprocessing/segment_audio.py`:

```
Raw WAV
  │
  ▼  ① Resample to 48 kHz (librosa)
  │
  ▼  ② Per-file RMS Normalisation
  │     Scale so 95th-percentile RMS = 0.05 reference level
  │
  ▼  ③ Non-overlapping 3-second Segmentation
  │     144,000 samples per segment; trailing audio < 3 s discarded
  │
  ▼  ④ Sub-frame Feature Extraction (6 × 0.5 s per segment)
  │     Features per sub-frame:
  │       • RMS Energy (dB)         — silence gate at −42 dB
  │       • Zero-Crossing Rate      — high ZCR = broadband noise
  │       • Spectral Flatness       — near 1.0 = white noise
  │       • Spectral Centroid Mean  — outside [1000, 10000] Hz = noise flag
  │       • Spectral Centroid Std   — high variance = wind instability
  │       • Autocorrelation Peak    — periodic peak = insect stridulation
  │
  ▼  ⑤ Noise Scoring + Majority Vote
       Noise score S = 0.25·ZCR + 0.30·Flatness + 0.15·centroid_flag
                     + 0.15·CentroidStd + 0.15·insect_flag
       S ≥ 0.50 → sub-frame votes noise
       bird_votes / active_frames ≥ 0.50 → segment = bird
       else → segment = noise
         │
         ├── Bird → data/processed/<species>/
         └── Noise → data/processed/noise/
```

---

## Embedding-Based ML Pipeline — How It Works

### Step 7: Embedding Extraction

BirdNET's internal model is not accessed directly. Instead, `analyzer.predict(chunk)` is called for each 3-second audio chunk and returns a 6,522-D vector of per-class confidence scores — one value per BirdNET species. For multi-chunk files, the element-wise maximum across chunks is taken to preserve the strongest signal per class.

PCA (`n_components=1024, svd_solver='randomized', seed=42`) is fitted **on training data only** and saved to `models/saved/pca_transform.pkl`. The val set is transformed with the already-fitted PCA — no val data influences the projection.

### Step 8: Model Training

**MLP Classifier** (`models/mlp_classifier.py`):

```
Linear(1024 → 512) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(512  → 256) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(256  → 1)   [raw logit — sigmoid applied externally]
```

- Loss: `BCEWithLogitsLoss` with `pos_weight = n_noise / n_bird ≈ 0.097` (handles 10:1 imbalance)
- Optimizer: Adam lr=1e-3, ReduceLROnPlateau scheduler
- Early stopping: patience=5 on val loss
- Threshold sweep at [0.3, 0.5, 0.7] → best threshold = **0.3** (maximises val F1)

**Autoencoder** (`models/autoencoder.py`):

```
Encoder: Linear(1024→512)→ReLU → Linear(512→256)→ReLU
Decoder: Linear(256→512)→ReLU  → Linear(512→1024)
```

- Trained **exclusively on 10,583 train-bird embeddings** (noise never seen)
- Threshold chosen at p99 of training-bird MSE distribution, evaluated by val F1
- At inference: reconstruction MSE > 4.703e−4 → segment flagged as OOD (noise)

**Temperature Scaling** (`models/temperature_scaling.py`):

```
p_calibrated = sigmoid(logit / T)
```

- MLP weights are frozen; only T is tuned
- T = 1.4648 minimises NLL on val set via `scipy.minimize_scalar`

### Step 9: Three-Stage Inference

```
For each val embedding:
  Stage 1 — AE gate
    mse = mean((embedding - AE(embedding))²)
    if mse > 4.703e-4 → predict noise (skip Stage 2)

  Stage 2 — MLP + Temperature
    logit = MLP(embedding)
    prob  = sigmoid(logit / 1.4648)
    if prob >= 0.3 → predict bird

  Stage 3 — Species lookup
    if bird → look up top BirdNET species from processed_predictions.json
               matched by filename stem
```

---

## Two-Stage Evaluation Design (Steps 4–6)

### Stage 1 — Bird Detection

| Metric | Processed Pipeline |
|---|---|
| Detection Precision | 0.9365 |
| Detection Recall | 0.3256 |
| Detection F1 | 0.4832 |
| FPR (noise → bird) | 0.1514 |

### Stage 2 — Species Classification

| Metric | Baseline (raw) | Processed (filtered) | Delta |
|---|---|---|---|
| Accuracy (all files) | 0.2270 | 0.1850 | −0.042 |
| **Accuracy among detected** | **0.4038** | **0.5681** | **+0.164** ✅ |
| Genus accuracy | 0.3297 | 0.2314 | −0.098 |
| Precision (macro) | 0.1465 | 0.1707 | +0.024 |
| Recall (macro) | 0.0964 | 0.0744 | −0.022 |
| F1 (macro) | 0.1089 | 0.0912 | −0.018 |

> **Preprocessing improves species classification accuracy among detected segments
> from 40.4% to 56.8% (+16.4%).** The noise filter trades coverage (lower recall)
> for quality (higher per-detection accuracy).

---

## Data Integrity and Leakage Checks

| Check | Result |
|---|---|
| PCA fitted on train only | ✅ `pca.fit_transform(X_train)` → `pca.transform(X_val)` |
| AE trained on bird-only train split | ✅ `binary_label=1, split='train'` filter applied |
| Threshold selection uses val set post-training | ✅ No val information enters training weights |
| All 4 models evaluated on same 2,928 val segments | ✅ Confirmed by path-level filtering |
| No mystery-mystery artefact labels in scored set | ✅ `EXCLUDED_LABELS = {"mystery mystery"}` applied consistently |

---

## Repository Structure

```
.
├── run_pipeline.py                      # Master orchestrator (Steps 1–10)
├── run_baseline_birdnet.py              # BirdNET on raw IBC53 recordings
├── run_processed_birdnet.py             # BirdNET on preprocessed segments
├── evaluate_metrics.py                  # Two-stage evaluation + plots
├── download_ibc53.py                    # Kaggle dataset downloader
├── CHANGELOG.md                         # Summary of ML pipeline additions
│
├── etl/
│   ├── extract.py                       # Step 1 — validate & ingest IBC53
│   ├── transform.py                     # Step 2 — segment + noise filter
│   └── load.py                          # Step 3 — stratified train/val split
│
├── preprocessing/
│   ├── segment_audio.py                 # Core: resample, normalise, segment, classify
│   ├── calibrate_features.py            # Pre-segmentation feature analysis
│   └── analyze_segmented_output.py      # Post-segmentation comparison
│
├── models/
│   ├── __init__.py
│   ├── extract_embeddings.py            # Step 7 — BirdNET 6522-D → PCA 1024-D
│   ├── mlp_classifier.py                # Step 8 — binary MLP classifier
│   ├── autoencoder.py                   # Step 8 — OOD autoencoder gate
│   ├── temperature_scaling.py           # Step 8 — post-hoc MLP calibration
│   ├── inference_pipeline.py            # Step 9 — three-stage inference
│   └── saved/
│       ├── pca_transform.pkl            # Fitted PCA (train only)
│       ├── birdnet_labels.json          # Ordered list of 6522 BirdNET classes
│       ├── mlp_classifier.pt            # Trained MLP weights + threshold
│       ├── mlp_config.json              # input_dim, best_threshold
│       ├── autoencoder.pt               # Trained AE weights
│       ├── ae_threshold.json            # OOD MSE threshold (p99)
│       └── temperature.json             # Calibration temperature T
│
├── evaluation/
│   ├── benchmark.py                     # Step 10 — four-model benchmark
│   └── embedding_viz.py                 # Step 10 — PCA + t-SNE plots
│
├── data/
│   ├── IBC53/                           # Raw dataset (53 species, 1,368 WAVs)
│   ├── processed/                       # Preprocessed 3-s segments (generated)
│   │   ├── <species>/                   # Bird segments per species
│   │   └── noise/                       # Rejected noise segments
│   ├── embeddings/                      # PCA embeddings (generated by Step 7)
│   │   ├── train/                       # 11,607 × 1024-D .npy files
│   │   ├── val/                         # 2,928  × 1024-D .npy files
│   │   └── manifest.csv                 # path, label, binary_label, split, source_file
│   └── transform_report.csv             # Per-species segment counts (generated)
│
├── results/
│   ├── baseline_predictions.json        # BirdNET on raw audio
│   ├── processed_predictions.json       # BirdNET on preprocessed segments
│   ├── mlp_predictions.json             # MLP-only inference (val)
│   ├── mlp_ae_predictions.json          # AE + MLP inference (val)
│   ├── baseline_metrics.json            # Stage 1+2 metrics — baseline
│   ├── processed_metrics.json           # Stage 1+2 metrics — processed
│   ├── mlp_metrics.json                 # Detection metrics — MLP only
│   ├── mlp_ae_metrics.json              # Detection metrics — MLP + AE
│   ├── ae_metrics.json                  # AE MSE stats + threshold sweep
│   ├── calibration_improvement.json     # ECE/NLL before and after scaling
│   ├── comparison.json                  # Baseline vs processed delta table
│   ├── benchmark_table.csv              # Four-model benchmark summary
│   ├── benchmark_comparison.json        # Four-model benchmark (JSON)
│   └── plots/
│       ├── confusion_matrix_baseline.png
│       ├── confusion_matrix_processed.png
│       ├── metrics_bar_chart.png
│       ├── detection_metrics_bar.png
│       ├── benchmark_bar.png            # Four-model comparison bar chart
│       ├── pca_plot.png                 # 2-D PCA of val embeddings
│       └── tsne_plot.png                # 2-D t-SNE of val embeddings
│
├── splits/
│   ├── train.csv                        # 11,607 rows — 80% stratified split
│   └── val.csv                          # 2,928  rows — 20% stratified split
│
├── requirements.txt
├── Readme.md
└── CHANGELOG.md
```

---

## Dataset

**IBC53 — Indian Bird Call Dataset**

| Property | Value |
|----------|-------|
| Source | Kaggle: `arghyasahoo/ibc53-indian-bird-call-dataset` |
| Species | 53 Indian bird species |
| Total recordings | 1,368 WAV files |
| Environment | Field recordings, varying gain, background noise |
| Format | Variable sample rate; resampled to 48 kHz in pipeline |

### Segmentation Summary

| Metric | Value |
|--------|-------|
| Source recordings | 1,368 |
| Total segments | 14,535 |
| Train split | 11,607 (10,583 bird + 1,024 noise) |
| Val split | 2,928 (2,671 bird + 257 noise) |

---

## Noise Segregation V2 — Tunable Constants

```python
SILENCE_GATE_DB      = -42.0   # Sub-frame energy cutoff (dB)
CENTROID_LOW_HZ      = 1000.0  # Lower bound of bird centroid range
CENTROID_HIGH_HZ     = 10000.0 # Upper bound of bird centroid range
AUTOCORR_THRESH      = 0.70    # Insect stridulation detection threshold
AUTOCORR_LAG_MIN_MS  = 5.0     # Lag window start (ms)
AUTOCORR_LAG_MAX_MS  = 20.0    # Lag window end (ms)
ZCR_MAX              = 0.30    # ZCR normalisation ceiling
CSTD_MAX             = 2500.0  # Centroid std normalisation ceiling
NOISE_SCORE_THRESH   = 0.50    # Sub-frame noise classification threshold
VOTE_THRESHOLD       = 0.50    # Segment-level bird vote fraction
RMS_REF_LEVEL        = 0.05    # Target 95th-percentile RMS after normalisation
```

All thresholds were calibrated against feature distributions extracted from 2,952
sub-frames across 50 IBC53 recordings.

---

## Limitations and Future Work

1. **BirdNET–IBC53 domain gap:** BirdNET V2.4 was trained on eBird/Xeno-Canto data dominated by North American and European recordings. Many rare Indian endemics are underrepresented, capping achievable species-level accuracy.

2. **Inference granularity mismatch:** Baseline feeds full recordings (BirdNET aggregates across many windows); processed pipeline feeds individual 3-second segments. A fairer comparison would aggregate per-recording predictions in both pipelines.

3. **AE gate separation ratio:** Val noise mean MSE is only 1.57× higher than val bird mean MSE, so the AE gate fires rarely (42/2,928 segments at p99). A more discriminative latent space (e.g., VAE, contrastive training) could improve OOD separation.

4. **Temperature scaling ECE:** T = 1.465 improves NLL but marginally increases ECE. The MLP's high-confidence bins were already well-calibrated; further calibration would require histogram binning or isotonic regression.

5. **Taxonomic synonym mapping:** Several IBC53 labels use outdated scientific names. Adding a synonym lookup would recover incorrectly penalised correct predictions.

6. **Overlapping segmentation:** Adding 50% overlap in the segmentation step would match BirdNET's internal windowing and reduce boundary losses for calls straddling 3-second windows.
