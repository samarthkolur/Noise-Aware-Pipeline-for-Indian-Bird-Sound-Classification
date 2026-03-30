# Bioacoustic Noise Segregation Pipeline

A production-ready, PyTorch-based machine learning pipeline designed to segregate clean bird vocalizations from background noise. Built with a modular, data-centric architecture, this pipeline supports automated preprocessing, high-dimensional embedding extraction (e.g., BirdNET), downstream classification, and active learning/hard-negative mining.

## Architecture & Pipeline Explanation

The workflow is divided into five distinct stages, all orchestrated by `run_pipeline.py` and configured centrally via `config.yaml`.

1. **Preprocessing (`--stage preprocess`)**
   Takes raw audio of any length from `data/raw/`, resamples it strictly to 48 kHz mono, and chops it into non-overlapping 3-second segments. Built-in RMS-thresholding automatically drops completely silent segments. Outputs `.wav` files and rich `.json` metadata sidecars to `data/processed/`.

2. **Embedding Extraction (`--stage embed`)**
   Loads the 3s segments and passes them through a heavy feature encoder (like BirdNET or a custom CNN). The 1024D floating-point embeddings are extracted and stored efficiently in a compressed HDF5 dataset (`data/embeddings/`) partitioned by species, alongside an auto-generated `manifest.csv` index tracker for O(1) dataloading.

3. **Training (`--stage train`)**
   Trains a lightweight Multi-Layer Perceptron (MLP) classification head natively on the HDF5 embeddings. Supports both Binary Classification (`BCEWithLogitsLoss` for Bird vs. Noise) and Multiclass paradigms. Features automatic inverse-frequency class weighting, AdamW optimization, Cosine Annealing learning rates, early stopping, and automatic checkpointing to `checkpoints/`.

4. **Inference (`--stage infer`)**
   An end-to-end evaluation engine. You point it at a raw audio file or directory; it safely segments the audio in a secure memory tempfile, extracts embeddings, runs the trained MLP classifier, and automatically routes the resulting 3s `.wav` segments (with their metadata tracking `.json`) into `outputs/clean_birds/` or `outputs/noise/` based on confidence logic.

5. **Mining / Active Learning (`--stage mine`)**
   A specialized loop for identifying False Positives (e.g., wind noise masquerading as a bird). It uses the inference engine to export high-confidence target predictions out of known-noise directories into a manual `review/` folder. After human verification, `Miner.update_dataset()` dynamically resizes the core `embeddings.h5` database, appending the new verified hard negatives and updating the `manifest.csv` for immediate iterative retraining.

---

## Setup Instructions

1. **Environment Setup**
   It is highly recommended to use a virtual environment (`venv` or `conda`).
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `soundfile` library is used natively for robust, cross-platform `.wav` audio I/O.)*

3. **Data Preparation**
   Place your raw audio files organized by species folder inside the raw directory configured in `config.yaml` (default: `data/raw/`).
   ```text
   data/raw/
   ├── Cyornis unicolor/
   │   ├── XC12345.wav
   │   └── XC67890.wav
   └── noise/
       └── wind_01.wav
   ```

---

## Commands

The pipeline is completely config-driven. Ensure your parameters match your environment in `config.yaml`, then execute `run_pipeline.py`.

### Run the Full Pipeline
Executes Preprocessing → Embedding → Training → Inference continuously:
```bash
python run_pipeline.py --config config.yaml
```

### Run Individual Stages
If you only need to run a specific part of the pipeline:

**1. Preprocess audio:**
```bash
python run_pipeline.py --stage preprocess
```

**2. Extract embeddings:**
```bash
python run_pipeline.py --stage embed
```

**3. Train classifier:**
```bash
python run_pipeline.py --stage train
```

**4. Run End-to-End Inference:**
```bash
python run_pipeline.py --stage infer
```

**5. Hard-Negative Mining (Extracting False Positives):**
```bash
python run_pipeline.py --stage mine
```
