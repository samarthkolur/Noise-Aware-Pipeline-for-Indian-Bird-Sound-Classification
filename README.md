# Noise-Aware Pipeline for Indian Bird Sound Classification

This repository implements a bird-vs-noise pipeline built around BirdNET V2.4 embeddings.

The code currently supports:
- Preprocessing (resample, segment, noise reduction, RMS silence filtering)
- Optional Noise Segregation V2 routing (baseline, filtered, full modes)
- BirdNET embedding extraction to HDF5 plus manifest.csv
- Binary classifier training (MLP) followed by mandatory bird-only autoencoder training (OOD gate)
- Inference: autoencoder-based OOD rejection gate (mandatory), then MLP with three-band routing (clean_birds, uncertain, noise)
- Evaluation on test split or full manifest
- Baseline-vs-pipeline comparison utilities
- Mining of false positives, false negatives, and uncertain samples

## Project Entry Points

Primary root scripts:
- run_pipeline.py
- evaluate.py
- compute_baseline_metrics.py
- research/run_research_suite.py (3-system benchmark, stats, errors, plots)
- generate_synthetic_noise.py
- clean_pipeline_outputs.py
- app.py

Compatibility wrappers are also available in scripts/ and call the root scripts.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional BirdNET integration tools:

```bash
pip install -r requirements-birdnet-integration.txt
```

## BirdNET Weights

The embedding extractor expects BirdNET V2.4 weights.

Default resolution behavior is configured in config.yaml via embedding.birdnet_model_path (auto). The repository comments indicate the expected filename:
- BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite

If auto lookup fails, set embedding.birdnet_model_path to an explicit .tflite path.

## Data Layout

Default paths from config.yaml:
- Raw audio: iBC53/
- Processed segments: data/processed/
- Embeddings and manifest: data/embeddings/
- Checkpoints: checkpoints/
- Inference outputs: outputs/
- Evaluation outputs: results/
- Baseline comparison outputs: comparison/

Expected raw structure:
- iBC53/<species_name>/*.wav
- Optional iBC53/noise/*.wav

## Run the Pipeline

Run all stages:

```bash
python run_pipeline.py --config config.yaml
```

Run one stage:

```bash
python run_pipeline.py --config config.yaml --stage preprocess
python run_pipeline.py --config config.yaml --stage embed
python run_pipeline.py --config config.yaml --stage train
python run_pipeline.py --config config.yaml --stage infer
python run_pipeline.py --config config.yaml --stage evaluate
python run_pipeline.py --config config.yaml --stage mine
```

Supported stage names are exactly:
- preprocess
- embed
- train
- infer
- evaluate
- mine

## Pipeline Modes

Set pipeline.mode in config.yaml:
- baseline: no V2 routing
- filtered: keep only V2 bird-like segments
- full: keep bird-like segments and route noise-like segments into processed noise

## Evaluation

Evaluate held-out test split (same split logic used by training):

```bash
python evaluate.py --config config.yaml
```

Evaluate full manifest:

```bash
python evaluate.py --config config.yaml --full-dataset
```

Writes results/metrics.json and (if present) backs up prior metrics to results/metrics_previous_run.json.

## Baseline vs Pipeline Comparison

Compare BirdNET baseline JSONL with trained pipeline predictions:

```bash
python compute_baseline_metrics.py --config config.yaml
```

Full-manifest comparison mode:

```bash
python compute_baseline_metrics.py --config config.yaml --full-dataset
```

Expected baseline input file:
- comparison/baseline_normalized.jsonl

Writes:
- comparison/baseline_metrics.json
- comparison/pipeline_metrics.json
- comparison/comparison_table.json
- results/comparison_graphs/metrics_comparison.png
- results/comparison_graphs/error_comparison.png

## Research benchmark suite (paper-ready)

After embeddings, baseline JSONL, and training (MLP + autoencoder checkpoints) are available, run:

```bash
python research/run_research_suite.py --config config.yaml
# or: python scripts/run_research_suite.py --config config.yaml
```

This evaluates **three** systems on the **same held-out test split** (BirdNET @0.5, MLP-only @0.5, MLP+AE @0.5), writes paired **t-test** and **Wilcoxon** results, mines error exemplars with heuristics, and saves PCA / t-SNE / AE histogram / MLP interpretability plots.

Outputs include:
- `results/benchmark_comparison.json`, `results/benchmark_table.csv`
- `results/statistical_tests.json`
- `results/error_analysis.json`, `results/error_samples/` (audio copies)
- `results/plots/pca_plot.png`, `tsne_plot.png`, `ae_error_distribution.png`, `feature_importance.png`
- `results/report_snippets.txt` (formal paragraphs for the report)

Options: `--baseline-jsonl PATH`, `--threshold 0.5`, `--top-k 10`, `--skip-plots`, `--skip-tsne` (faster).

## Optional Utilities

Generate synthetic noise files into iBC53/noise/:

```bash
python generate_synthetic_noise.py --config config.yaml --n_files 50
```

Clean cached artifacts (processed, embeddings, checkpoints):

```bash
python clean_pipeline_outputs.py
python clean_pipeline_outputs.py --dry-run
```

## Streamlit App

Run the UI:

```bash
streamlit run app.py
```

## OOD rejection gate (autoencoder)

Training fits a bottleneck autoencoder with MSE on **bird-only** embeddings (label `1`) in the same 1024-D BirdNET space. The validation split supplies a reconstruction-error distribution; **τ_AE** is the configured percentile (default 95th) on bird validation errors. At inference and evaluation, each embedding is reconstructed first: if per-sample MSE **exceeds τ_AE**, the segment is labeled noise (OOD) and **is not passed to the MLP**; otherwise the MLP applies the usual three-band decision. Artifacts: `checkpoints/autoencoder.pt`, `checkpoints/ae_threshold.json` (and legacy `autoencoder_meta.json`). Old checkpoints without threshold JSON still load if `autoencoder_meta.json` is present next to the weights.

## Notes

- The repository includes both config.yaml and configs/config.yaml. Root scripts default to config.yaml unless you pass a different --config path.
- Comparison quality depends on key alignment between manifest rows and comparison/baseline_normalized.jsonl.
- Reported metrics depend on your local data, preprocessing mode, and trained checkpoints.
