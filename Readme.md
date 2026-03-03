# Noise-Aware Bird Segregation Pipeline (V2)

A multi-stage deep-learning pipeline for separating bird vocalizations from noise in audio recordings, designed to work upstream of [BirdNET](https://github.com/kahst/BirdNET-Analyzer) for Indian bird species classification using the **IBC53** dataset.

## Architecture

```
Raw Audio → Segmentation (3s @ 48kHz)
          → Embedding Extraction (BirdNET + YAMNet + OpenL3)
          → Binary Classifier (MLP with Focal Loss)
          → OOD Detection (Mahalanobis / OCSVM / IForest / Autoencoder)
          → Source Separation (HPSS harmonic ratio)
          → Temporal Smoothing (sliding window / majority vote)
          → Ensemble Decision (multi-signal voting)
          → Hard-Negative Mining (iterative retraining)
          → Noise-Aware Dataset → BirdNET Fine-Tuning
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `pipeline/stage1_segmentation.py` | Resample to 48 kHz, segment into 3s overlapping windows |
| 2 | `pipeline/stage2_embeddings.py` | Extract embeddings via BirdNET, YAMNet, OpenL3 |
| 3 | `pipeline/stage3_classifier.py` | MLP bird-vs-noise classifier with Focal Loss |
| 4 | `pipeline/stage4_ood.py` | OOD detection ensemble (4 methods) |
| 5 | `pipeline/stage5_source_separation.py` | HPSS harmonic energy ratio filtering |
| 6 | `pipeline/stage6_temporal.py` | Temporal consistency smoothing |
| 7 | `pipeline/stage7_ensemble.py` | Multi-signal ensemble decision |
| 8 | `pipeline/stage8_hard_negatives.py` | Iterative hard-negative mining |

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset (IBC53)

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d arghyasahoo/ibc53-indian-bird-call-dataset
unzip ibc53-indian-bird-call-dataset.zip -d data/
```

Final structure:
```
data/
└── iBC53/
    ├── Corvus_splendens/
    ├── Acridotheres_fuscus/
    └── ... (53 species)
```

## Usage

### Run Full Pipeline
```bash
python run_pipeline.py --stage all
```

### Run Individual Stages
```bash
python run_pipeline.py --stage segment          # Stage 1: Segment audio
python run_pipeline.py --stage embed             # Stage 2: Extract embeddings
python run_pipeline.py --stage embed --max-files 50  # Limit files for testing
python run_pipeline.py --stage train             # Stage 3: Train classifier
python run_pipeline.py --stage ood               # Stage 4: Train OOD detectors
python run_pipeline.py --stage hpss              # Stage 5: Compute harmonic ratios
python run_pipeline.py --stage infer             # Stages 6-8: Inference + mining
python run_pipeline.py --stage evaluate          # Run evaluation
```

### Ablation Study
```bash
python run_pipeline.py --stage all --ablation
```

## Configuration

All hyperparameters are in `config.py`:
- Audio: sample rate, segment length, overlap
- Embeddings: model selection, fusion toggle
- Classifier: MLP hidden dims, learning rate, focal loss parameters
- OOD: thresholds, contamination factors
- Ensemble: per-signal thresholds, minimum agreement
- Stage enable/disable flags for ablation

## Output structure

```
models/                         # Saved model weights
evaluation/results/             # Metrics JSON, plots (ROC, PR, confusion matrix)
data/noise_aware_dataset/       # Final segregated dataset
    ├── bird/                   # Segments classified as bird
    └── noise/                  # Segments classified as noise
```

## Expected Performance

| Configuration | Accuracy |
|---|---|
| Handcrafted thresholds | 80–85% |
| Classical ML on features | 90–93% |
| Embedding classifier only | 92–95% |
| Embedding + OOD + Ensemble | **95–98%** |