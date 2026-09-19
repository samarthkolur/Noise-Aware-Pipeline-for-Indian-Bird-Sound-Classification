# Noise-Aware BirdNET V3

**A data-centric pipeline for false-positive suppression in passive acoustic monitoring.**

> Pre-trained bird classifiers produce false positives on environmental noise.
> This pipeline builds a better *dataset* — not a better model — using hard-negative
> mining, active learning, and multi-signal ensemble verification.

---

## Architecture

```
Raw Audio → Segmentation → BirdNET Embeddings → Hard-Negative Curation
    → Binary Classifier (RF) → OOD Filter → Active Learning
    → Post-Processing → Ensemble Decision → Clean Dataset
```

| Stage | Description | Key Method |
|-------|-------------|------------|
| 1 | Segmentation | 3s windows @ 48 kHz |
| 2 | Embedding Extraction | BirdNET 1024-d |
| 3 | Hard-Negative Dataset Curation | Pseudo-labels + spectral flatness filter |
| 4 | Binary Classifier | Random Forest (primary), MLP (optional) |
| 5 | OOD Detection | Mahalanobis + Isolation Forest (AND gate) |
| 6 | Active Learning *(optional)* | Uncertainty sampling → expert review |
| 7 | Post-Processing | Temporal smoothing + harmonic ratio check |
| 8 | Ensemble Decision | Confidence-weighted voting |

---

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Run full pipeline (unattended)
python run_pipeline.py --stage all --skip-active-learning

# Run individual stages
python run_pipeline.py --stage segment
python run_pipeline.py --stage embed
python run_pipeline.py --stage curate
python run_pipeline.py --stage train
python run_pipeline.py --stage ood
python run_pipeline.py --stage postprocess
python run_pipeline.py --stage ensemble
python run_pipeline.py --stage evaluate

# With active learning
python run_pipeline.py --stage active

# Ablation studies
python run_pipeline.py --stage ablation          # Pipeline ablation
python run_pipeline.py --stage noise-ablation    # Noise Segregation V2 ablation

# Limit files for testing
python run_pipeline.py --stage embed --max-files 20
```

---

## Noise Segregation V2 — Ablation Study

The Noise Segregation V2 module uses a **weighted combination of three hand-crafted acoustic features** to detect environmental noise and override BirdNET false positives:

| Feature | Weight | Description |
|---------|--------|-------------|
| **Spectral Flatness** | 0.50 | Broadband noise detector (traffic, wind, rain). Values near 1.0 = flat spectrum = noise. |
| **Zero-Crossing Rate** | 0.30 | Percussive/impulsive noise detector. High ZCR = clicks, static, broadband. |
| **Insect Periodicity** | 0.20 | Periodic envelope detector via autocorrelation. Catches cicada/cricket buzz. |

### Ablation Results

An ablation study was conducted to quantify each feature's contribution. Each configuration re-labels the entire dataset, retrains a fresh Random Forest classifier, and evaluates on a held-out validation set (57 segments).

| Configuration | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| **Original weights** (0.5 / 0.3 / 0.2) | **0.9474** | **1.0000** | **0.7000** | **0.8235** |
| Equal weights (1/3 each) | 0.9474 | 1.0000 | 0.7000 | 0.8235 |
| Without spectral flatness | 0.9123 | 1.0000 | 0.5000 | 0.6667 |
| Without ZCR | 0.9474 | 1.0000 | 0.7000 | 0.8235 |
| Without insect periodicity | 0.9474 | 1.0000 | 0.7000 | 0.8235 |

### Deltas from Original

| Configuration | ΔAccuracy | ΔPrecision | ΔRecall | ΔF1 |
|---------------|-----------|------------|---------|-----|
| Equal weights | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Without spectral flatness | **−0.0351** | +0.0000 | **−0.2000** | **−0.1569** |
| Without ZCR | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Without insect periodicity | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

### Feature Importance Ranking

| Rank | Feature | F1 Drop When Removed |
|------|---------|---------------------|
| 1 (Most important) | **Spectral Flatness** | −0.1569 |
| 2 | Zero-Crossing Rate | 0.0000 |
| 3 (Least important) | Insect Periodicity | 0.0000 |

### Key Findings

1. **Spectral flatness is the dominant feature** — removing it causes a 15.69% F1 drop and 20% recall degradation. It is the only feature that directly overrides BirdNET false positives in this dataset (2 overrides when absent vs. 0 when present).

2. **ZCR and insect periodicity have minimal marginal impact** — on the IBC53 dataset (Indian bird recordings), these features do not trigger overrides. This is expected: the dataset contains primarily tonal bird species with low ZCR, and the strict insect periodicity detector (requiring ≥2 consistent autocorrelation peaks > 0.6) correctly avoids triggering on bird trills.

3. **Equal weights match original performance** — the original weights are justified but not uniquely optimal on this dataset. The spectral flatness contribution is diluted under equal weights but still sufficient.

4. **Perfect precision maintained** — all configurations achieve 1.000 precision, confirming the noise segregation is conservative and never incorrectly labels a bird as noise.

> **Run the ablation:** `python run_pipeline.py --stage noise-ablation`
> **Full report:** `evaluation/results/noise_segregation_ablation_results.json`

---

## Configuration

All settings in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASSIFIER_TYPE` | `"rf"` | Primary classifier: `"rf"` or `"mlp"` |
| `RF_N_ESTIMATORS` | `300` | Number of RF trees |
| `OOD_METHODS` | `["mahalanobis", "iforest"]` | OOD detectors |
| `BIRD_CONFIDENCE_HIGH` | `0.5` | BirdNET conf → bird label |
| `BIRD_CONFIDENCE_LOW` | `0.1` | BirdNET conf → noise label |
| `AL_EXPORT_TOP_K` | `200` | Max uncertain samples per AL round |
| `ENSEMBLE_THRESHOLD` | `0.5` | Weighted score threshold for bird |

---

## Project Structure

```
NoiseAwareBirdNET_V2/
├── config.py                     # Central configuration
├── run_pipeline.py               # Main orchestrator
├── requirements.txt
├── pipeline/
│   ├── stage1_segmentation.py
│   ├── stage2_embeddings.py
│   ├── stage3_hard_negative_dataset.py
│   ├── stage4_binary_classifier.py
│   ├── stage5_ood_filter.py
│   ├── stage6_active_learning.py
│   ├── stage7_postprocessing.py
│   ├── stage8_ensemble.py
│   └── noise_segregation_v2.py   # Multi-feature noise scoring + ablation configs
├── experiments/
│   └── ablation_noise_segregation.py  # Noise Segregation V2 ablation study
├── evaluation/
│   └── evaluate.py
├── tests/
│   ├── test_noise_segregation.py # Noise segregation feature tests
│   ├── test_classifier.py
│   ├── test_embeddings.py
│   ├── test_ood_save_load.py
│   ├── test_active_learning.py
│   └── test_pipeline_e2e.py
├── data/
│   ├── iBC53/                    # Raw dataset
│   ├── segmented/                # 3s segments
│   ├── hard_negative_dataset/    # Curated dataset
│   ├── noise_aware_dataset/      # Final output
│   └── active_learning/          # Expert review CSVs
├── features/embeddings/          # .npy files
├── models/                       # Saved classifiers + OOD
└── evaluation/results/           # Metrics, plots, ablation results
```

---

## Key Design Decisions

1. **BirdNET-only embeddings** — YAMNet/OpenL3 removed. BirdNET's 1024-d representation already captures species-discriminative features.

2. **Random Forest primary** — Trains in <1 minute on BirdNET embeddings, enabling rapid iteration. MLP available for comparison.

3. **Hard-negative mining at dataset level** — Instead of iterative retraining (old Stage 8), hard negatives are identified upfront using spectral flatness as a BirdNET false-positive detector.

4. **AND-gate OOD** — Both Mahalanobis and Isolation Forest must agree for bird classification. Conservative but high-precision.

5. **Optional active learning** — Expert-in-the-loop for maximum data efficiency, but fully skippable for automated deployment.

6. **Noise Segregation V2** — Multi-feature weighted scoring (spectral flatness, ZCR, insect periodicity) with ablation-ready architecture. Each feature can be independently disabled to quantify its contribution.

---

## Output

- `data/noise_aware_dataset/bird/` — Verified bird segments
- `data/noise_aware_dataset/noise/` — Rejected noise segments
- `evaluation/results/` — Metrics (JSON), confusion matrix, ROC, PR curves
- `evaluation/results/noise_segregation_ablation_results.json` — Ablation study results
- `data/active_learning/` — Expert review CSVs and progress tracking