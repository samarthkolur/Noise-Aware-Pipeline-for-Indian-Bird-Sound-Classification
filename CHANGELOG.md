# CHANGELOG — Noise-Aware Bird Detection Pipeline

## Branch: boom_boom

---

## 🔹 Overview

This update introduces a complete transition from confidence-based BirdNET thresholding to an embedding-based supervised learning pipeline using MLP and optional autoencoder gating.

The preprocessing pipeline remains unchanged. All improvements arise from downstream modeling.

---

## 🔹 Added Components

### 1. Embedding Extraction

* File: `models/extract_embeddings.py`
* BirdNET per-class confidence (6522-D) used as features
* PCA applied (train-only fit) → 1024-D embeddings
* Outputs:

  * `data/embeddings/train/`
  * `data/embeddings/val/`
  * `data/embeddings/manifest.csv`
  * `models/saved/pca_transform.pkl`

---

### 2. MLP Classifier

* File: `models/mlp_classifier.py`
* Architecture:

  * 1024 → 512 → 256 → 1
* Loss: BCEWithLogitsLoss
* Best threshold selected via validation sweep: **0.3**
* Outputs:

  * `models/saved/mlp_classifier.pt`
  * `models/saved/mlp_config.json`
  * `results/mlp_metrics.json`

---

### 3. Temperature Scaling

* File: `models/temperature_scaling.py`
* Optimized temperature: **T = 1.4648**
* Improved NLL, slight increase in ECE (documented behavior)
* Outputs:

  * `models/saved/temperature.json`
  * `results/calibration_improvement.json`

---

### 4. Autoencoder (OOD Gating)

* File: `models/autoencoder.py`
* Trained on bird embeddings only
* Threshold selected via percentile sweep: **p99**
* Outputs:

  * `models/saved/autoencoder.pt`
  * `models/saved/ae_threshold.json`
  * `results/ae_metrics.json`

---

### 5. Inference Pipeline

* File: `models/inference_pipeline.py`

* Three-stage inference:

  1. AE gating (optional)
  2. MLP classification
  3. BirdNET species lookup

* Outputs:

  * `results/mlp_predictions.json`
  * `results/mlp_ae_predictions.json`

---

### 6. Benchmarking

* File: `evaluation/benchmark.py`

* Compared:

  * BirdNET baseline
  * Noise-aware BirdNET
  * MLP
  * MLP + AE

* Output:

  * `results/benchmark_table.csv`
  * `results/benchmark_comparison.json`

---

### 7. Visualization

* File: `evaluation/embedding_viz.py`
* PCA (train-fit, val-transform)
* t-SNE (val-only)
* Outputs:

  * `results/plots/pca_plot.png`
  * `results/plots/tsne_plot.png`

---

## 🔹 Key Findings

* MLP significantly outperforms BirdNET thresholding:

  * F1: **0.940 vs 0.489**
* AE provides marginal precision gain, slight recall trade-off
* Temperature scaling improves NLL but not ECE
* Performance gains come from **learned classification**, not preprocessing

---

## 🔹 Important Notes

* Preprocessing pipeline remains unchanged across all experiments
* No data leakage:

  * PCA fit on train only
  * AE trained on train-bird only
  * Validation strictly separated
* BirdNET is used as a **feature extractor**, not final classifier

---

## 🔹 Cleanup

* Removed duplicate artifact:

  * `results/autoencoder.pt`

---

## 🔹 Final Status

* Pipeline: Complete
* Evaluation: Validated
* Results: Reproducible
* Ready for report submission
