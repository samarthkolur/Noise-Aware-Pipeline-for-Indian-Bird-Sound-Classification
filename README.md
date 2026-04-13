# Noise-Aware Pipeline for Indian Bird Sound Classification

A production-ready, data-centric ML pipeline that segregates clean bird vocalizations from background noise using **BirdNET V2.4** embeddings. Supports automated preprocessing, 1024-dimensional embedding extraction, binary classification with optimized thresholds, three-class inference routing (bird / noise / uncertain), and active learning through hard-negative and false-negative mining.

## Pipeline Architecture

```
Raw Audio → Preprocessing → BirdNET Embeddings → Classifier → Routing
                                                      ↓
                                              ┌───────┼───────┐
                                          clean_birds  uncertain  noise
                                                          ↓
                                                   Manual Review
                                                          ↓
                                                  Dataset Update → Retrain
```

### Stages

| Stage | Command | Description |
|-------|---------|-------------|
| **1. Preprocess** | `--stage preprocess` | Resample to 48 kHz mono, 3s chunks, RMS silence drop; optional **V2** routing (`pipeline.mode`) |
| **2. Embed** | `--stage embed` | Extract 1024D embeddings from BirdNET V2.4 penultimate layer → HDF5 |
| **3. Train** | `--stage train` | Train MLP classifier (BCE/CE), F1-based checkpointing, auto-threshold optimization |
| **4. Infer** | `--stage infer` | Three-class routing: `clean_birds/` / `noise/` / `uncertain/` |
| **5. Evaluate** | `--stage evaluate` | Text metrics: confusion matrix, threshold curve, recall-at-precision (`evaluate.py`) |
| **6. Mine** | `--stage mine` | False positive + false negative mining for active learning |

After training, optionally run **`evaluate_visual.py`** (not part of `run_pipeline`) to save **plots** — see [Standalone evaluation](#standalone-evaluation) below.

### Pipeline modes (`pipeline.mode` in `config.yaml`)

| Mode | V2 scoring | Processed outputs | Typical use |
|------|------------|-------------------|-------------|
| **baseline** | Off | One folder per raw species; optional manual `noise/` | Classic BirdNET + MLP; you supply non-bird audio in `raw_dir/noise/` if you need class 0 |
| **filtered** | On; **drops** V2-noise segments | Only V2-bird segments under each species folder | Cleaner species embeddings; **do not** treat binary metrics as a noise-rejection benchmark unless you add a separate noise test set |
| **full** | On; **routes** V2-noise to `processed_dir/noise/` | Species folders + auto-filled `noise/` | Bird vs noise **without** manual noise clips; embeddings get both classes for valid binary evaluation |

Preprocessing order: **segment → RMS silence drop → V2 (filtered/full) → save WAV + JSON** (+ optional `segments_manifest.csv`). Embeddings are unchanged: **BirdNET 1024D** from every saved WAV.

### Valid binary evaluation checklist

1. **Full embedding set** must include label 0 (`noise/` in HDF5). Use `pipeline.mode: full`, or add audio under `raw_dir/noise/`.
2. **Test split** must contain both classes; otherwise F1, confusion matrix, and **FPR on noise** are misleading (`evaluate.py` will log an error and set `binary_eval_valid: false` in `results/metrics.json`).
3. Compare runs (baseline vs filtered vs full) using the **same** random seed and split settings.

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

> **Note:** `birdnetlib` ships the official BirdNET V2.4 TFLite model (~50 MB). The `ai-edge-litert` package provides the TFLite runtime for Python 3.12+.

### 3. Data Preparation

Place raw audio files organized by species inside the directory set in `config.yaml` (default: `iBC53/`):

```
iBC53/
├── Cyornis unicolor/
│   ├── 1.wav
│   └── 2.wav
├── Parus cinereus/
│   └── 1.wav
└── noise/           ← optional noise class
    └── wind_01.wav
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
# Preprocess: segment + resample + silence removal
python run_pipeline.py --config config.yaml --stage preprocess

# Embed: extract BirdNET 1024D embeddings → HDF5
python run_pipeline.py --config config.yaml --stage embed

# Train: MLP classifier with F1-based checkpointing
python run_pipeline.py --config config.yaml --stage train

# Infer: three-class routing (bird / noise / uncertain)
python run_pipeline.py --config config.yaml --stage infer

# Evaluate: confusion matrix + threshold analysis (text + metrics JSON)
python run_pipeline.py --config config.yaml --stage evaluate

# Mine: false positive + false negative mining
python run_pipeline.py --config config.yaml --stage mine
```

**Visual plots** (not a `run_pipeline` stage; run after training when `checkpoints/best_model.pt` exists):

```bash
python evaluate_visual.py --config config.yaml
```

### Standalone evaluation

#### Text report (`evaluate.py`)

Same logic as `--stage evaluate`:

```bash
python evaluate.py --config config.yaml
```

Outputs (console + JSON):
- Confusion matrix at default and optimal thresholds
- Precision / Recall / F1 at each threshold
- Per-class precision/recall and **FPR on noise** (false bird predictions on true noise), when the test set contains both classes
- Recall-at-minimum-precision sweep
- Probability distribution per class
- `results/metrics.json` (configurable via `evaluation.results_dir`)

#### Visual plots (`evaluate_visual.py`)

Run **after** you have trained checkpoints (`checkpoints/best_model.pt`). This script saves figures under `results/` (not invoked by `run_pipeline.py`):

```bash
python evaluate_visual.py --config config.yaml
```

Optional baseline comparison (prints deltas only):

```bash
python evaluate_visual.py --config config.yaml --compare-json results/metrics_baseline.json
```

Generated files:
- `results/confusion_matrix.png` — heatmap
- `results/metrics_bar_chart.png` — accuracy / precision / recall / F1
- `results/threshold_comparison.png` — binary only (0.5 vs best-F1 threshold)
- `results/metrics.json` — structured metrics (may overwrite the file from `evaluate.py`)

---

## Configuration

All parameters are in `config.yaml`. Key settings:

```yaml
embedding:
  model_name: birdnet         # birdnet | placeholder
  birdnet_model_path: auto    # 'auto' = detect from birdnetlib

model:
  binary: true                # true = bird/noise, false = per-species

inference:
  confidence_threshold: auto  # 'auto' = use optimal from training
  high_threshold: 0.7         # above → bird
  low_threshold: 0.3          # below → noise
                              # between → uncertain
```

---

## Three-Class Routing

Instead of a hard binary decision, inference uses **three confidence bands**:

| Probability | Decision | Destination |
|-------------|----------|-------------|
| `prob > high_threshold` (0.7) | **Bird** | `outputs/clean_birds/` |
| `prob < low_threshold` (0.3) | **Noise** | `outputs/noise/` |
| Between thresholds | **Uncertain** | `outputs/uncertain/` |

The `uncertain/` folder is designed for **manual review** — listen to these segments and move confirmed birds to a verified directory, then run `miner.update_dataset()` to retrain.

---

## Active Learning Loop

```
1. Train → Infer → Evaluate
2. Review outputs/uncertain/ and outputs/noise/
3. Move verified birds to verified_birds/
4. Run: miner.update_dataset(verified_dir, target_species="<species>")
5. Retrain → repeat
```

The mining stage automates step 2 by scanning `noise/` for potential false negatives (bird sounds misclassified as noise) and exporting them to `outputs/false_negatives/` for review.

---

## Project Structure

```
├── config.yaml                          # Central configuration
├── run_pipeline.py                      # CLI entry-point (all stages)
├── evaluate.py                          # Standalone text evaluation + metrics JSON
├── evaluate_visual.py                   # Plots (heatmap, bar charts) + metrics JSON → results/
├── clean_pipeline_outputs.py            # Optional: delete processed/embeddings/checkpoints
├── requirements.txt                     # Core pipeline dependencies
├── requirements-birdnet-analyzer.txt    # Official BirdNET-Analyzer dependencies
├── requirements-birdnet-integration.txt # Additional integration dependencies
│
├── birdnet_integration/       # Tools for integrating and comparing BirdNET predictions
├── birdnet_weights/           # Manually downloaded BirdNET TFLite models
├── checkpoints/               # Trained model checkpoints
├── comparison/                # Outputs of baseline vs filtered comparison scripts
├── data/                      # Auto-generated processed audio chunks + embeddings
├── dataset/                   # PyTorch Dataset, splits, class weighting
├── embedding/                 # BirdNET TFLite encoder, HDF5 storage
├── iBC53/                     # Default input directory for raw audio files by species
├── inference/                 # Three-class predictor + routing
├── mining/                    # False positive/negative mining, dataset update
├── models/                    # MLP classifier architecture
├── outputs/                   # Inference sorted files (clean_birds/, noise/, uncertain/)
├── preprocessing/             # Segmentation, silence removal, Noise Segregation V2
├── results/                   # Evaluation metrics, confusion matrix, and threshold plots
├── tests/                     # Unit tests (e.g. V2 scoring)
├── training/                  # Training loop, metrics, threshold optimization
└── utils/                     # Config loader, logger
```

---

## Key Technical Details

- **BirdNET V2.4**: Embeddings extracted from the penultimate `GLOBAL_AVG_POOL` layer (1024D) via TFLite with XNNPACK disabled for intermediate tensor access
- **F1-Based Checkpointing**: Model checkpoints are saved based on best validation F1 (not val_loss) for recall-sensitive tasks
- **Optimal Threshold**: After training, the pipeline sweeps 50 threshold candidates and saves the F1-optimal threshold to `best_model_meta.json`
- **Audio Backend**: Uses `soundfile` (not torchaudio) for cross-platform robustness without FFmpeg dependency
