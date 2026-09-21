# Noise-Aware Pipeline for Indian Bird Sound Classification

**BirdNET embeddings + Focal-loss MLP + Autoencoder OOD gating.**

Implementation of "Noise-Aware Pipeline for Indian Bird Sound Classification
Using BirdNET Embeddings, Focal-Loss MLP, and Autoencoder Gating"
(Kolur et al., DSU). `design.md` is the canonical engineering document for
this project — read it first. This file is a quick-start pointer, not a
second source of truth.

---

## Architecture

```
Raw audio (data/iBC53/<species>/*.wav)
    → resample + 3s non-overlapping segmentation + RMS silence gate
    → Noise Segregation V2 (6-subframe, 5-feature weighted vote) + Bird Guard + Bird Rescue
    → BirdNET V2.4 embedding extraction (1024-D, HDF5-cached)
    → Focal-loss MLP classifier + bird-only Autoencoder OOD gate
    → Three-band router (bird / uncertain / noise) + hard-example mining
```

See `design.md` §6 for the full architecture, §10 for the phased delivery
plan, and §26 for a documented gap between this repo's own benchmark numbers
and the paper's Table I (short version: no real iBC53 noise-class corpus is
available here, so the noise class is a synthetic substitute).

## Quick Start

### Dashboard (recommended for exploration)

```bash
streamlit run app/dashboard.py
```

One UI covering every phase below: run each script with live-streamed output,
run inference on a WAV file, browse the benchmark/analytics results, run the
test suite, and read `design.md` section-by-section. See design.md DD-021.

### CLI

```bash
uv sync --extra dev

# Phase 2: build the manifest from raw audio (non-overlapping 3s segments)
python scripts/prepare_data.py --allow-synthetic

# Phase 4: extract BirdNET embeddings into the HDF5 cache
python scripts/extract_embeddings.py

# Phase 5: train the Focal-loss MLP + bird-only Autoencoder
python scripts/train.py

# Phase 6: run full inference on a folder of WAV files
python scripts/infer.py --input path/to/wavs/

# Phase 7: three-way benchmark + evaluation report
python scripts/benchmark.py

# Phase 8: export to ONNX
python scripts/export_onnx.py

# Tests
uv run pytest                 # fast suite
uv run pytest -m slow         # + real BirdNET integration tests
```

## Configuration

All paths, hyperparameters, and thresholds live in [`config.yaml`](config.yaml)
and are loaded via [`pipeline/config.py`](pipeline/config.py)'s Pydantic
model — see design.md DD-004. No magic numbers belong in source code.

## Project Structure

See `design.md` §14 for the authoritative repository structure. Top level:

```
pipeline/     Core library modules (audio, noise segregation, embedding, model, ...)
scripts/      CLI entry points, one per delivery phase
tests/        pytest suite (unit + integration, see design.md §14)
data/         Corpus, manifest, HDF5 embedding cache (mostly gitignored)
artifacts/    Trained model weights + config.json (weights gitignored)
outputs/      Inference routing output + evaluation_report.json + plots/
```

## Status

All 8 delivery phases are implemented and passing (84+ tests, 92%+ coverage,
`ruff`/`mypy` clean). See `design.md` §21–§26 for current phase, known issues,
and technical debt — most notably, this repo's `outputs/evaluation_report.json`
is a real but reduced-scale run, not a paper Table I reproduction.
