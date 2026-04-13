# Noise-Aware Pipeline for Indian Bird Sound Classification

A production-ready, data-centric ML pipeline that segregates clean bird vocalizations from background noise using **BirdNET V2.4** embeddings. Built for the **iBC53** dataset (50 Indian bird species), the system implements automated preprocessing with Noise Segregation V2, 1024-dimensional embedding extraction, binary classification with optimized thresholds, autoencoder-based anomaly gating, three-class inference routing, and active learning through hard-negative mining.

## Results

Evaluated on the iBC53 dataset (9,794 segments: 7,829 bird across 50 species + 1,965 noise).

| Metric | BirdNET Baseline | Noise-Aware Pipeline | Improvement |
|--------|-----------------|---------------------|-------------|
| **Accuracy** | 0.499 | **0.922** | +42.3 pp |
| **Precision** | 0.773 | **0.946** | +17.3 pp |
| **Recall** | 0.529 | **0.958** | +42.9 pp |
| **F1 Score** | 0.628 | **0.952** | +32.4 pp |
| **FPR** (noise misclassified as bird) | 0.619 | **0.218** | 65% reduction |
| **FNR** (bird missed) | 0.471 | **0.042** | 91% reduction |

<p align="center">
  <img src="results/comparison_graphs/metrics_comparison.png" width="48%" />
  <img src="results/comparison_graphs/error_comparison.png" width="48%" />
</p>

---

## Pipeline Architecture

```
                            ┌──────────────────────────────────────────────────┐
                            │           Noise-Aware Pipeline                   │
                            │                                                  │
  Raw Audio (iBC53/)        │  ┌────────────┐    ┌──────────────┐             │
  ├── Species_A/  ──────────┼─▶│ Preprocess  │───▶│ Noise Seg V2 │             │
  ├── Species_B/            │  │ 48kHz mono  │    │ (bird/noise  │             │
  └── ...                   │  │ 3s segments │    │  routing)    │             │
                            │  │ RMS silence │    └──────┬───────┘             │
                            │  │ removal     │           │                     │
                            │  └─────────────┘    ┌──────┴───────┐             │
                            │                     │              │             │
                            │              processed/bird  processed/noise     │
                            │                     │              │             │
                            │                     ▼              ▼             │
                            │              ┌─────────────────────────┐         │
                            │              │  BirdNET V2.4 Encoder   │         │
                            │              │  (1024D penultimate)    │         │
                            │              │  → HDF5 + manifest.csv  │         │
                            │              └────────────┬────────────┘         │
                            │                           │                     │
                            │                           ▼                     │
                            │              ┌─────────────────────────┐         │
                            │              │   MLP Classifier Head   │         │
                            │              │   1024 → 512 → 256 → 1 │         │
                            │              │   (Binary: bird/noise)  │         │
                            │              └────────────┬────────────┘         │
                            │                           │                     │
                            │              ┌────────────┴────────────┐         │
                            │              │  Autoencoder Gate       │         │
                            │              │  (reconstruction error  │         │
                            │              │   anomaly detection)    │         │
                            │              └────────────┬────────────┘         │
                            │                           │                     │
                            │              ┌────────────┴────────────┐         │
                            │              │  Three-Class Routing    │         │
                            │              │                        │         │
                            │         prob > 0.7    0.3–0.7    prob < 0.3     │
                            │              │          │            │           │
                            │              ▼          ▼            ▼           │
                            │        clean_birds/  uncertain/   noise/        │
                            │                         │                       │
                            │                    Manual Review                │
                            │                         │                       │
                            │                   Active Learning               │
                            │                   (Mine → Retrain)              │
                            └──────────────────────────────────────────────────┘
```

### Stages

| Stage | Command | Description |
|-------|---------|-------------|
| **1. Preprocess** | `--stage preprocess` | Resample to 48 kHz mono, 3s segments, RMS silence drop, Noise Segregation V2 routing |
| **2. Embed** | `--stage embed` | Extract 1024D embeddings from BirdNET V2.4 penultimate layer into HDF5 |
| **3. Train** | `--stage train` | Train MLP classifier (focal/BCE loss), F1-based checkpointing, autoencoder training |
| **4. Infer** | `--stage infer` | Three-class routing: `clean_birds/` / `noise/` / `uncertain/` with optional AE gating |
| **5. Evaluate** | `--stage evaluate` | Confusion matrix, per-class metrics, threshold sweep, FPR/FNR analysis |
| **6. Mine** | `--stage mine` | False positive + false negative mining for active learning |

---

## Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> `birdnetlib` ships the official BirdNET V2.4 TFLite model (~50 MB). The `ai-edge-litert` package provides the TFLite runtime for Python 3.12+.

### 3. Data Preparation

Place raw audio files organized by species inside `iBC53/` (configurable in `config.yaml`):

```
iBC53/
├── Acridotheres fuscus/
│   ├── 10.wav
│   └── 11.wav
├── Cyornis unicolor/
│   └── 1.wav
├── ...                    # 50 species total
└── noise/                 # optional (V2 full mode auto-generates noise)
    └── ambient_01.wav
```

---

## Usage

### Full Pipeline

Runs all stages sequentially (preprocess → embed → train → infer → evaluate → mine):

```bash
python run_pipeline.py --config config.yaml
```

### Individual Stages

```bash
python run_pipeline.py --config config.yaml --stage preprocess
python run_pipeline.py --config config.yaml --stage embed
python run_pipeline.py --config config.yaml --stage train
python run_pipeline.py --config config.yaml --stage infer
python run_pipeline.py --config config.yaml --stage evaluate
python run_pipeline.py --config config.yaml --stage mine
```

### Baseline vs Pipeline Comparison

After training and evaluation, generate comparison metrics and plots:

```bash
python compute_baseline_metrics.py
```

This computes the BirdNET baseline from `comparison/baseline_normalized.jsonl`, recomputes pipeline metrics from the trained classifier, and generates:
- `comparison/baseline_metrics.json`
- `comparison/pipeline_metrics.json`
- `comparison/comparison_table.json`
- `results/comparison_graphs/metrics_comparison.png`
- `results/comparison_graphs/error_comparison.png`

### Visual Evaluation Plots

```bash
python evaluate_visual.py --config config.yaml
```

Generates `results/confusion_matrix.png`, `results/metrics_bar_chart.png`, and `results/threshold_comparison.png`.

---

## Pipeline Modes

Controlled by `pipeline.mode` in `config.yaml`:

| Mode | V2 Scoring | Processed Outputs | Use Case |
|------|------------|-------------------|----------|
| **baseline** | Off | One folder per species; manual `noise/` only | Classic BirdNET + MLP without noise segregation |
| **filtered** | On; drops noise segments | Only V2-bird segments per species | Cleaner species embeddings; fewer segments |
| **full** | On; routes noise to `processed_dir/noise/` | Species folders + auto-generated `noise/` | Bird vs noise without manual noise clips (default) |

Preprocessing order: **segment (3s) → RMS silence drop → V2 score (filtered/full) → save WAV + JSON sidecar**.

---

## Three-Class Inference Routing

Instead of a hard binary decision, inference uses three confidence bands:

| Probability | Decision | Destination |
|-------------|----------|-------------|
| `prob > 0.7` | **Bird** | `outputs/clean_birds/` |
| `0.3 ≤ prob ≤ 0.7` | **Uncertain** | `outputs/uncertain/` |
| `prob < 0.3` | **Noise** | `outputs/noise/` |

An optional **autoencoder gate** (reconstruction-error anomaly detector) rejects out-of-distribution segments before classification, routing them directly to `noise/`.

---

## Noise Segregation V2

The V2 noise segregation algorithm classifies each 3-second segment as bird or noise using subframe-level acoustic features:

1. Each segment is split into 0.5s subframes
2. Per-subframe features: ZCR, spectral flatness, spectral centroid, centroid stability, insect-band autocorrelation
3. Weighted noise score: `S = w_zcr * ZCR + w_flat * Flatness + w_centroid * CentroidFlag + ...`
4. Each subframe votes noise if `S >= threshold`
5. Majority vote across subframes determines the segment label

Segments labeled as noise in `full` mode are saved to `data/processed/noise/` with filenames encoding the original source: `{OriginalSpecies}__{FileNum}_segXXXX.wav`.

---

## Model Architecture

### Embedding Encoder

**BirdNET V2.4** (TFLite) — the penultimate `GLOBAL_AVG_POOL/Mean` layer produces 1024-dimensional embeddings. XNNPACK delegates are disabled to preserve internal tensor access. Embeddings are stored in per-species HDF5 files with a global `manifest.csv`.

### Binary Classifier

```
Input (1024D)
  → Linear(1024, 512) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(512, 256)  → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256, 1)    → sigmoid → probability
```

**657,921 parameters** | Focal loss (gamma=2.0) | Adam optimizer | Cosine annealing | Early stopping (patience=7)

### Autoencoder (Optional Anomaly Gate)

Symmetric autoencoder trained on bird-only embeddings. At inference, segments with reconstruction error above the P95 threshold are rejected as out-of-distribution and routed to `noise/` before reaching the classifier.

---

## Evaluation Details

### Pipeline Evaluation (`evaluate.py`)

Produces `results/metrics.json` containing:
- Metrics at threshold 0.5 and F1-optimal threshold
- Per-class precision, recall, and FPR on noise
- Threshold curve (50-point sweep)
- Recall-at-minimum-precision analysis
- Probability distribution statistics per class
- Autoencoder reconstruction error statistics

### Baseline Comparison (`compute_baseline_metrics.py`)

Compares raw BirdNET detection confidence (threshold=0.5) against the trained pipeline classifier:

**Ground truth**: derived from `data/embeddings/manifest.csv` where noise=0 and any species=1.

**Baseline prediction**: for each manifest segment, the maximum BirdNET detection confidence from `comparison/baseline_normalized.jsonl` is thresholded at 0.5. Segments with no BirdNET detection default to confidence=0.

**Key mapping**: noise segments in the manifest encode their original species source in the filename (`SpeciesName__FileNum`), which is parsed to recover the matching BirdNET JSONL key.

---

## Active Learning Loop

```
1. Train → Infer → Evaluate
2. Review outputs/uncertain/ and outputs/noise/
3. Move verified birds to verified_birds/
4. Run: miner.update_dataset(verified_dir, target_species="<species>")
5. Retrain → repeat
```

The mining stage automates step 2 by scanning `noise/` for potential false negatives and exporting them to `outputs/false_negatives/` for review.

---

## Project Structure

```
Noise-Aware-Pipeline-for-Indian-Bird-Sound-Classification/
│
├── config.yaml                          # Central YAML configuration
├── run_pipeline.py                      # CLI entry point (6 stages)
├── evaluate.py                          # Standalone evaluation → results/metrics.json
├── evaluate_visual.py                   # Visual plots → results/*.png
├── compute_baseline_metrics.py          # BirdNET baseline vs pipeline comparison
├── clean_pipeline_outputs.py            # Delete processed/embeddings/checkpoints
├── requirements.txt                     # Python dependencies
│
├── preprocessing/                       # Stage 1: Audio preprocessing
│   ├── preprocessing.py                 #   Core: resample, segment, silence drop, V2 routing
│   ├── noise_segregation_v2.py          #   Subframe noise scoring + majority vote
│   ├── audio_loader.py                  #   Audio loading utilities
│   ├── feature_extractor.py             #   Mel-spectrogram / MFCC extraction
│   ├── noise_reduction.py               #   Spectral gating / bandpass denoising
│   └── preprocess_cli.py               #   Standalone CLI for preprocessing
│
├── embedding/                           # Stage 2: Embedding extraction
│   ├── embedding.py                     #   BirdNET TFLite encoder, HDF5 storage, manifest
│   ├── embedding_model.py               #   Legacy CNN encoder (unused)
│   └── extract_embeddings.py            #   Legacy .pt extraction (unused)
│
├── dataset/                             # Dataset & DataLoader
│   ├── dataset.py                       #   EmbeddingDataset, splits, class weights, sampling
│   ├── bird_dataset.py                  #   Legacy spectrogram dataset (unused)
│   └── data_utils.py                    #   Legacy split/weight utilities (unused)
│
├── models/                              # Neural network architectures
│   ├── classifier.py                    #   MLP classification head (1024→512→256→1)
│   ├── autoencoder.py                   #   Symmetric AE for reconstruction-error gating
│   └── attention.py                     #   Attention pooling (experimental, unused)
│
├── training/                            # Stage 3: Model training
│   ├── trainer.py                       #   Classifier training loop with early stopping
│   ├── autoencoder_trainer.py           #   AE training on bird-only embeddings
│   └── metrics.py                       #   Accuracy, F1, confusion matrix, threshold search
│
├── inference/                           # Stage 4: Inference & routing
│   ├── predictor.py                     #   End-to-end: raw audio → preprocess → embed → classify → route
│   └── postprocessing.py                #   Temporal smoothing (experimental, unused)
│
├── mining/                              # Stage 6: Active learning
│   └── mining.py                        #   FP/FN mining, uncertain export, dataset update
│
├── birdnet_integration/                 # BirdNET-Analyzer experiment tools
│   ├── run_baseline_birdnet.py          #   Run BirdNET-Analyzer on raw audio
│   ├── run_filtered_birdnet.py          #   Run BirdNET on pipeline-filtered audio
│   ├── normalize_birdnet_export.py      #   Convert BirdNET CSVs to JSONL
│   ├── align_segments.py                #   Align BirdNET detections to pipeline segments
│   ├── compare_baseline_filtered.py     #   Metrics + plots for baseline vs filtered
│   ├── run_full_experiment.py           #   Orchestrate full baseline vs filtered experiment
│   ├── integration_config.py            #   Experiment config loader
│   └── experiment_config.yaml           #   BirdNET experiment parameters
│
├── utils/                               # Shared utilities
│   ├── config.py                        #   YAML config loader
│   ├── logger.py                        #   Console logging setup
│   ├── io_utils.py                      #   Atomic file saving
│   └── verify_env.py                    #   Torch/CUDA environment check
│
├── tests/                               # Unit tests
│   └── test_noise_segregation_v2.py     #   Tests for V2 noise scoring
│
├── iBC53/                               # Raw audio input (species subdirectories)
├── data/
│   ├── processed/                       #   Preprocessed 3s WAV segments
│   └── embeddings/                      #   HDF5 embeddings + manifest.csv
├── checkpoints/                         #   Trained model weights
│   ├── best_model.pt                    #   MLP classifier checkpoint
│   ├── best_model_meta.json             #   Label map, optimal threshold
│   ├── autoencoder.pt                   #   AE checkpoint
│   └── autoencoder_meta.json            #   Reconstruction threshold
├── outputs/                             #   Inference results
│   ├── clean_birds/                     #   High-confidence bird segments
│   ├── noise/                           #   High-confidence noise segments
│   ├── uncertain/                       #   Segments for manual review
│   ├── false_positives/                 #   Mined false positives
│   └── false_negatives/                 #   Mined false negatives
├── results/                             #   Evaluation outputs
│   ├── metrics.json                     #   Full evaluation metrics
│   ├── confusion_matrix.png             #   Confusion matrix heatmap
│   ├── metrics_bar_chart.png            #   Per-metric bar chart
│   ├── threshold_comparison.png         #   Threshold sweep visualization
│   └── comparison_graphs/               #   Baseline vs pipeline plots
│       ├── metrics_comparison.png       #   Accuracy/Precision/Recall/F1
│       └── error_comparison.png         #   FPR and FNR comparison
└── comparison/                          #   Baseline comparison data
    ├── baseline_normalized.jsonl        #   Normalized BirdNET detections
    ├── baseline_metrics.json            #   Baseline confusion matrix + metrics
    ├── pipeline_metrics.json            #   Pipeline confusion matrix + metrics
    └── comparison_table.json            #   Side-by-side comparison
```

---

## Configuration Reference

All parameters are in `config.yaml`. Key settings:

```yaml
pipeline:
  mode: full                # baseline | filtered | full

embedding:
  model_name: birdnet       # birdnet (required for real embeddings)
  embedding_dim: 1024       # BirdNET V2.4 penultimate layer

model:
  binary: true              # bird/noise binary classification
  hidden_dims: [512, 256]   # MLP hidden layer sizes

training:
  loss: focal               # cross_entropy | focal | label_smoothing
  epochs: 50
  early_stopping:
    patience: 7

inference:
  confidence_threshold: auto  # 'auto' = F1-optimal from training
  high_threshold: 0.7         # prob > 0.7 → bird
  low_threshold: 0.3          # prob < 0.3 → noise

autoencoder:
  enabled: true              # reconstruction-error anomaly gating
  recon_threshold: auto      # 'auto' = P95 from training
```

---

## Key Technical Details

- **BirdNET V2.4**: Embeddings from the penultimate `GLOBAL_AVG_POOL` layer (1024D) via TFLite with XNNPACK disabled (`BUILTIN_WITHOUT_DEFAULT_DELEGATES`) for intermediate tensor access
- **Noise Segregation V2**: Subframe-level acoustic analysis (ZCR, spectral flatness, centroid stability, insect-band autocorrelation) with majority vote
- **Focal Loss**: Addresses class imbalance (80% bird / 20% noise) with gamma=2.0 down-weighting easy examples
- **F1-Based Checkpointing**: Model saved on best validation F1, not loss, for recall-sensitive bioacoustic tasks
- **Optimal Threshold**: Post-training 50-point threshold sweep; F1-optimal threshold stored in `best_model_meta.json` and used automatically during inference
- **Autoencoder Gating**: Reconstruction error on bird-only training distribution; P95 threshold rejects OOD inputs before classification
- **Audio Backend**: `soundfile` (not torchaudio) for cross-platform robustness without FFmpeg dependency
- **Stratified Splits**: Train 75% / Val 15% / Test 10% with class-proportional stratification (seed=42)

---

## Dataset

**iBC53**: 50 Indian bird species from the Indian Bird Call dataset. After preprocessing with Noise Segregation V2:

| Split | Total | Bird | Noise |
|-------|-------|------|-------|
| Full dataset | 9,794 | 7,829 | 1,965 |
| Train (75%) | 7,345 | — | — |
| Validation (15%) | 1,469 | — | — |
| Test (10%) | 980 | 783 | 197 |

---

## License

This project is for academic and research purposes.
