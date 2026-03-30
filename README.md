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
| **1. Preprocess** | `--stage preprocess` | Resample to 48 kHz mono, segment into 3s chunks, drop silent segments (RMS threshold) |
| **2. Embed** | `--stage embed` | Extract 1024D embeddings from BirdNET V2.4 penultimate layer → HDF5 |
| **3. Train** | `--stage train` | Train MLP classifier (BCE/CE), F1-based checkpointing, auto-threshold optimization |
| **4. Infer** | `--stage infer` | Three-class routing: `clean_birds/` / `noise/` / `uncertain/` |
| **5. Evaluate** | `--stage evaluate` | Confusion matrix, threshold curve, recall-at-precision analysis |
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

# Evaluate: confusion matrix + threshold analysis
python run_pipeline.py --config config.yaml --stage evaluate

# Mine: false positive + false negative mining
python run_pipeline.py --config config.yaml --stage mine
```

### Standalone Evaluation

```bash
python evaluate.py --config config.yaml
```

Outputs:
- Confusion matrix at default and optimal thresholds
- Precision / Recall / F1 at each threshold
- Recall-at-minimum-precision sweep
- Probability distribution per class

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
├── config.yaml              # Central configuration
├── run_pipeline.py          # CLI entry-point (all stages)
├── evaluate.py              # Standalone evaluation script
├── requirements.txt
│
├── preprocessing/           # Audio loading, segmentation, silence removal
├── embedding/               # BirdNET TFLite encoder, HDF5 storage
├── dataset/                 # PyTorch Dataset, splits, class weighting
├── models/                  # MLP classifier architecture
├── training/                # Training loop, metrics, threshold optimization
├── inference/               # Three-class predictor + routing
├── mining/                  # False positive/negative mining, dataset update
└── utils/                   # Config loader, logger
```

---

## Key Technical Details

- **BirdNET V2.4**: Embeddings extracted from the penultimate `GLOBAL_AVG_POOL` layer (1024D) via TFLite with XNNPACK disabled for intermediate tensor access
- **F1-Based Checkpointing**: Model checkpoints are saved based on best validation F1 (not val_loss) for recall-sensitive tasks
- **Optimal Threshold**: After training, the pipeline sweeps 50 threshold candidates and saves the F1-optimal threshold to `best_model_meta.json`
- **Audio Backend**: Uses `soundfile` (not torchaudio) for cross-platform robustness without FFmpeg dependency
