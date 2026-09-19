# Design Document
## Noise-Aware Pipeline for Indian Bird Sound Classification
### Using BirdNET Embeddings, Focal-Loss MLP, and Autoencoder Gating

**Version 1.0 | Research Paper Implementation | Domain: Bioacoustics, Edge AI, Passive Acoustic Monitoring**
**Department of Computer Science & Technology, Dayananda Sagar University**

---

## 1. Executive Summary

India hosts over 1,300 bird species — roughly 13% of global avian diversity — yet systematic monitoring of this biodiversity remains constrained by terrain, cost, and the scarcity of trained ornithologists in remote areas. Passive Acoustic Monitoring (PAM) offers a scalable, non-invasive route to continuous biodiversity assessment, but generates enormous volumes of audio that exceed the capacity of manual ornithological review.

**This project implements the noise-aware machine-listening pipeline** described in the paper "Noise-Aware Pipeline for Indian Bird Sound Classification Using BirdNET Embeddings, Focal-Loss MLP, and Autoencoder Gating" (Kolur et al., DSU). The pipeline closes a critical gap: BirdNET, trained predominantly on global temperate-zone recordings, misses up to **57.4% of vocalisations from Indian species** under realistic field-noise conditions when applied at a fixed confidence threshold of 0.5.

The system combines six tightly coupled components:
1. RMS-based silence rejection
2. Subframe-level Noise Segregation V2 with harmonic Bird Guard
3. MLP-based Bird Rescue (second-chance classification)
4. Frozen BirdNET V2.4 embedding extraction (1024-D)
5. Focal-loss binary MLP classifier (Indian bird vs. noise)
6. Autoencoder OOD rejection gate + three-band confidence router

**Paper-reported results on iBC53 (10,044 segments: 9,790 bird / 254 noise):**

| System | Accuracy | F1 | FNR | ROC-AUC |
|---|---|---|---|---|
| BirdNET Baseline (threshold 0.5) | 0.440 | 0.598 | 57.4% | 0.724 |
| MLP Only | **0.999** | **1.000** | 0.04% | **0.998** |
| MLP + AE Gate | 0.995 | 0.998 | 0.43% | 0.994 |

This implementation document defines the problem, scope, system architecture, technology stack, dataset handling, training pipeline, evaluation protocol, and phased delivery plan to reproduce and extend those results in a clean, production-quality codebase.

---

## 2. Problem Statement

**Core research question**: Can a modular, noise-aware preprocessing pipeline — combining acoustic heuristics, domain-adapted frozen embeddings, and a lightweight learned classifier with OOD gating — dramatically close the false-negative gap that off-the-shelf BirdNET confidence thresholding leaves on Indian PAM recordings?

Five structural barriers make naive deployment of BirdNET insufficient for Indian PAM:

- **Species gap**: Many endemic Indian species are absent or under-represented in BirdNET's training data (trained predominantly on North American and European recordings), causing near-zero confidence scores on genuine vocalisations.
- **Noise regime gap**: Indian soundscapes feature dense insect choruses (crickets, cicadas, katydids) that spectrally overlap with bird calls in the 3–8 kHz range — far more severe than temperate-forest environments in BirdNET's training distribution.
- **Calibration gap**: BirdNET outputs sigmoid-transformed logits, not calibrated probabilities. A universal threshold misclassifies noise-dominated segments as birds (FPR inflation) and genuine Indian calls as noise (FNR inflation) simultaneously.
- **OOD gap**: Standard classifiers assign overconfident probabilities to inputs outside their training distribution. No OOD rejection mechanism exists in the off-the-shelf BirdNET pipeline.
- **Corpus cleaning gap**: High FNR corrupts the training corpora for downstream species-level classifiers, compounding the problem in any PAM workflow that relies on BirdNET-screened data as an annotation shortcut.

---

## 3. Objectives

1. Implement Noise Segregation V2 — a six-subframe, five-feature, weighted-voting acoustic scorer — with harmonic Bird Guard (harmonic ratio + spectral peak prominence checks) to prevent edge-truncation of genuine calls.
2. Implement Bird Rescue — an MLP re-check on segments initially routed to the noise class — to recover faint or partially masked calls that pass the acoustic heuristics as noise despite being genuine vocalisations.
3. Extract and cache frozen BirdNET V2.4 embeddings (1024-D, TFLite runtime) into HDF5 for reproducible downstream experiments without re-running BirdNET inference on each training cycle.
4. Train a focal-loss binary MLP classifier (1024 → 512 → 256 → 1, focal loss γ=2.0) on the frozen embeddings, addressing the 38.5:1 class imbalance without aggressive upsampling.
5. Train a bird-only autoencoder (1024 → 128 → 1024) to model the in-distribution bird embedding manifold; derive an OOD gating threshold τ_AE at the 99th-percentile reconstruction error over validation bird embeddings.
6. Implement a three-band confidence router (p ≥ 0.7 = bird, 0.3 < p < 0.7 = uncertain, p ≤ 0.3 or OOD = noise) with automated hard-example mining for human-in-the-loop active-learning cycles.
7. Reproduce the paper's three-way benchmark on iBC53 (10,044 segments) within ±1% of the reported metrics (Table I), with full statistical validation (paired t-test + Wilcoxon signed-rank, p < 0.05).
8. Ensure every pipeline component is deployable on CPU-only hardware (no GPU requirement) and is in principle quantisable to TFLite/ONNX for future edge deployment on low-power embedded recorders (Raspberry Pi-class, AudioMoth companion compute).
9. Document a reproducible benchmark protocol so that results can be re-derived from raw iBC53 audio without reading any source code, following only `design.md §9` and `design.md §10`.

---

## 4. Scope

### In Scope
- Binary bird-versus-noise classification on iBC53 (53 Indian species, 10,044 three-second segments).
- All six pipeline components (§3 Objectives 1–6) implemented as independent, testable Python modules.
- Frozen BirdNET V2.4 TFLite embedding extraction (`birdnet_analyzer` library or direct TFLite runtime).
- Focal-loss MLP and bird-only autoencoder training in PyTorch (CPU wheels; no CUDA required).
- Three-way benchmark reproduction (BirdNET baseline, MLP Only, MLP + AE Gate).
- Statistical validation (paired t-test, Wilcoxon signed-rank) on the full manifest.
- HDF5 embedding cache to decouple BirdNET inference from classifier training.
- Stratified 75/15/10 train/val/test split, reproducible via a fixed random seed and a saved manifest CSV.
- Automated hard-example mining (likely false positives and false negatives exported to a review directory).
- Configuration via a single `config.yaml` entry point (matching the paper's stated design).
- Docker-first development environment (`git clone && ./scripts/bootstrap.sh && docker compose watch`).
- CI/CD (GitHub Actions): lint, typecheck, test, build, security scanning.

### Out of Scope (Phase 1)
- Species-level classification within the bird class — the pipeline produces binary decisions only.
- Cross-dataset or cross-regional validation (Western Ghats, Indo-Gangetic plain, Himalayan foothills) — flagged as Future Work (§17).
- Real-time streaming inference — the pipeline processes pre-segmented 3-second clips.
- Fine-tuning BirdNET's convolutional encoder — embeddings are frozen throughout.
- Custom PCB or embedded hardware prototyping — edge feasibility is demonstrated analytically (§5 resource budget), not via a hardware build in Phase 1.
- Multi-class / multi-label species output.

---

## 5. Target Users

| User | Primary Need | Interaction Mode |
|---|---|---|
| Bioacoustics researcher | Reproduce paper results and run ablations | CLI training pipeline + config.yaml |
| PAM deployment engineer | Screen raw field recordings for bird content | CLI inference script on a folder of WAV files |
| Conservation data curator | Curate clean bird audio for downstream classifiers | Three-band router output + hard-example review queue |
| ML engineer (edge AI) | Deploy the MLP + AE gate on a low-power recorder | ONNX / TFLite export from the trained models |

---

## 6. System Architecture

### 6.0 Pipeline State Model

The pipeline processes audio in stages. Each stage produces a structured artifact:

```
Raw WAV (3 s segment)
    ↓ [Preprocessing]
SegmentRecord {
  path: str,
  species_label: str | "noise",
  rms_db: float,
  silence_rejected: bool
}
    ↓ [Noise Segregation V2 + Bird Guard]
SegmentRecord + {
  noise_score: float,          # weighted subframe vote [0.0–1.0]
  noise_class: "bird" | "noise",
  bird_guard_triggered: bool   # True if harmonic/peak check overrode noise vote
}
    ↓ [Bird Rescue MLP]
SegmentRecord + {
  rescue_prob: float | None,   # None if segment was already "bird"
  rescued: bool
}
    ↓ [BirdNET Embedding]
SegmentRecord + {
  embedding: ndarray[1024],    # cached in HDF5
  birdnet_confidence: float    # max species confidence (baseline only)
}
    ↓ [MLP Classifier]
SegmentRecord + {
  mlp_prob: float,             # P(bird)
  mlp_prediction: "bird" | "noise"
}
    ↓ [AE Gate]
SegmentRecord + {
  ae_mse: float,
  ood_rejected: bool           # ae_mse > τ_AE
}
    ↓ [Three-Band Router]
SegmentRecord + {
  final_band: "bird" | "uncertain" | "noise",
  routed_to: str               # output directory path
}
```

This state record is the single data contract between pipeline stages. Each stage reads from the record it receives and appends its own fields; it never modifies upstream fields. The full per-segment record is written to `outputs/manifest_results.csv` at the end of an inference run.

### 6.1 Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       INPUT LAYER                                │
│  iBC53 corpus (*.wav, 48 kHz mono, 3 s segments)                 │
│  Synthetic noise augmentation (pink/brown/white/band-limited)    │
│  RMS silence rejection (< −40 dB → discard)                      │
└────────────────────────────┬─────────────────────────────────────┘
                              │ per-segment WAV
┌────────────────────────────▼─────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│  • Resample → 48 kHz mono (librosa / soundfile)                  │
│  • 3 s non-overlapping segmentation                              │
│  • RMS silence gate (−40 dB threshold)                           │
│  • Segment manifest CSV (path, label, split)                     │
└────────────────────────────┬─────────────────────────────────────┘
                              │ silent-rejected: discard
                              │ surviving segments
┌────────────────────────────▼─────────────────────────────────────┐
│                  NOISE SEGREGATION V2 LAYER                      │
│  6 subframes × 0.5 s per 3 s segment                             │
│  Per-subframe features:                                          │
│    • ZCR (w=0.25) — insect high-frequency stridulation           │
│    • Spectral Flatness (w=0.30) — broadband vs. tonal            │
│    • Centroid Flag (w=0.15) — spectral balance instability        │
│    • Centroid Std (w=0.15) — temporal instability                 │
│    • Insect Periodicity Flag (w=0.15) — 3–8 kHz autocorr peak    │
│  Majority vote → "bird-like" | "noise-like"                      │
│  Bird Guard override (harmonic ratio >0.3 OR peak/median >3.0)   │
│  Bird Rescue MLP re-check on noise-routed segments               │
└────────────────────────────┬─────────────────────────────────────┘
                              │ confirmed bird-like OR rescued
                              │ confirmed noise → noise/ folder
┌────────────────────────────▼─────────────────────────────────────┐
│                  EMBEDDING LAYER                                  │
│  BirdNET V2.4 TFLite (frozen, no fine-tuning)                    │
│  Penultimate global average pooling → 1024-D embedding           │
│  HDF5 cache (segment_id → embedding array)                       │
│  manifest CSV gains embedding_cached: bool column               │
└────────────────────────────┬─────────────────────────────────────┘
                              │ 1024-D embedding vectors
┌────────────────────────────▼─────────────────────────────────────┐
│                  TRAINING PIPELINE                                │
│  Stratified 75/15/10 split (fixed seed, saved to split.csv)      │
│  Focal-Loss MLP (PyTorch, CPU):                                  │
│    1024 → 512 → BN+ReLU+Dropout(0.3)                            │
│    → 256 → BN+ReLU+Dropout(0.3) → 1 → sigmoid                   │
│    Focal loss γ=2.0; Adam + cosine annealing; early stop val-F1   │
│  Bird-Only Autoencoder (PyTorch, CPU):                           │
│    Encoder: 1024 → 512 → 128                                     │
│    Decoder: 128 → 512 → 1024                                     │
│    MSE reconstruction loss; trained on bird embeddings only       │
│  τ_AE = P99 reconstruction MSE on val bird embeddings            │
└────────────────────────────┬─────────────────────────────────────┘
                              │ trained model artifacts (.pt / .onnx)
┌────────────────────────────▼─────────────────────────────────────┐
│                  INFERENCE PIPELINE                               │
│  1. AE Gate: MSE > τ_AE → OOD → route to noise/uncertain        │
│  2. MLP Classifier: P(bird) via sigmoid                          │
│  3. Three-Band Router:                                           │
│       p ≥ 0.70 → bird/   (high-confidence bird)                  │
│       0.30 < p < 0.70 → uncertain/  (human review queue)         │
│       p ≤ 0.30 OR OOD → noise/  (confident noise)               │
│  4. Hard-example mining: export likely FP/FN to review/          │
└────────────────────────────┬─────────────────────────────────────┘
                              │ manifest_results.csv
┌────────────────────────────▼─────────────────────────────────────┐
│                  EVALUATION LAYER                                 │
│  Three-way benchmark (BirdNET baseline, MLP Only, MLP+AE Gate)   │
│  Metrics: Accuracy, Precision, Recall, F1, FPR, FNR              │
│  Curves: ROC-AUC, PR-AUC (test split only)                       │
│  Statistical validation: paired t-test + Wilcoxon (p < 0.05)    │
│  Output: evaluation_report.json + plots/                         │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Noise Segregation V2 — Feature Detail

| Feature | Formula / Method | Discriminative Role | Weight |
|---|---|---|---|
| Zero-Crossing Rate (ZCR) | sign-change density in time domain | High in high-frequency insect stridulation | 0.25 |
| Spectral Flatness | geometric mean / arithmetic mean of power spectrum | Near 1.0 for broadband noise; near 0 for tonal calls | 0.30 |
| Centroid Flag (`CFlag`) | spectral centroid > adaptive threshold | Indicates spectral balance | 0.15 |
| Centroid Std (`σ_c`) | std of centroid across subframe windows | Temporal instability of broadband noise | 0.15 |
| Insect Periodicity (`IFlag`) | band-limited (3–8 kHz) autocorrelation peak | Targets repetitive cricket/cicada chirp patterns | 0.15 |

Per-subframe noise score:

```
S = 0.25·ZCR̂ + 0.30·Flatnesŝ + 0.15·CFlaĝ + 0.15·σ̂_c + 0.15·IFlag
```

Features are min-max normalised to [0, 1] within each segment before scoring. Majority vote over 6 subframes classifies the segment. Bird Guard override: if harmonic ratio > 0.3 OR spectral peak-to-median ratio > 3.0, the segment is retained in the bird class regardless of the vote.

**Weight sensitivity note (DD-007)**: These weights were set qualitatively based on discriminative power in Bengaluru urban/peri-urban soundscapes. A systematic ablation study (equal-weight, feature-drop, grid-search variants) is the top-priority future experiment (§17). Until that ablation is run, treat the weight vector as an unvalidated hyperparameter, not a tuned result.

### 6.3 MLP Classifier

```
Input: 1024-D BirdNET embedding (L2-normalised at inference)
Architecture:
  Linear(1024, 512) → BatchNorm1d(512) → ReLU → Dropout(0.3)
  Linear(512, 256)  → BatchNorm1d(256) → ReLU → Dropout(0.3)
  Linear(256, 1)    → Sigmoid
Loss: Focal loss, γ=2.0, α=0.25 (down-weights easy majority-class examples)
Optimiser: Adam(lr=1e-3, weight_decay=1e-4)
Scheduler: CosineAnnealingLR(T_max=50)
Early stopping: patience=7 epochs, monitor=val F1
Max epochs: 50
```

The focal loss modulating factor (1 − p)^γ concentrates gradient signal on hard boundary cases — essential for the 38.5:1 bird-to-noise class imbalance in iBC53 without aggressive upsampling that would distort the small noise class.

### 6.4 Bird-Only Autoencoder (OOD Gate)

```
Architecture:
  Encoder: Linear(1024, 512) → ReLU → Linear(512, 128) → ReLU
  Decoder: Linear(128, 512)  → ReLU → Linear(512, 1024)
Loss: MSE reconstruction loss
Training data: bird-class embeddings ONLY (from train split)
τ_AE: P99 of reconstruction MSE over val bird embeddings
       (empirically τ_AE = 0.16975 on iBC53, per paper Table I)
Gating rule: if ε(e) > τ_AE → OOD → route to noise/uncertain band
```

The autoencoder learns the geometry of the in-distribution bird embedding manifold. Segments outside that manifold — novel noise types, atypical recording conditions, heavily reverberant recordings — produce high reconstruction error and are flagged as OOD before reaching the MLP classifier.

### 6.5 Three-Band Router and Hard-Example Mining

```
Band boundaries: τ_low = 0.30, τ_high = 0.70
  p ≥ 0.70        → outputs/bird/       (high-confidence bird)
  0.30 < p < 0.70 → outputs/uncertain/  (≈2% of segments; human review)
  p ≤ 0.30        → outputs/noise/      (high-confidence noise)
  OOD (AE gate)   → outputs/noise/      (routed pre-MLP)

Hard-example mining:
  - Likely FP: noise-class segments where MLP assigns p > 0.5
    (exported to review/likely_fp/ for expert annotation)
  - Likely FN: bird-class segments where MLP assigns p < 0.5 OR AE-rejected
    (exported to review/likely_fn/ for expert annotation)
  - Annotated corrections feed back into training/fine-tuning on next cycle
```

### 6.6 Resource Budget (CPU-only, no GPU)

| Component | Estimated Inference Time (per 3 s segment) | Memory |
|---|---|---|
| Resample + segment | <5 ms | negligible |
| Noise Segregation V2 (FFT-based) | <10 ms | negligible |
| BirdNET V2.4 TFLite embedding extraction | 200–500 ms | ~100 MB (model) |
| MLP forward pass (CPU) | <1 ms | ~8 MB (model) |
| AE forward pass (CPU) | <1 ms | ~4 MB (model) |
| **Total per segment** | **~300–600 ms** | **~115 MB** |

MLP parameters: 1024×512 + 512 + 512×256 + 256 + 256×1 + 1 = **655,617 parameters** (~2.5 MB at float32).
AE parameters: 1024×512 + 512×128 + 128×512 + 512×1024 = **1,182,720 parameters** (~4.5 MB at float32).

All components fit on a Raspberry Pi 4 (4 GB RAM) or an AudioMoth companion compute module with comfortable headroom.

---

## 7. Technology Stack

| Layer | Component | Technology | Justification |
|---|---|---|---|
| Audio I/O | WAV loading, resampling | `librosa` + `soundfile` | De facto standard; handles 48 kHz mono requirement and gracefully resamples any input |
| BirdNET embedding | TFLite runtime inference | `tflite-runtime` (or `tensorflow` lite subset) | Matches the paper's stated method; avoids the full TF dependency on CPU-only hardware |
| Embedding cache | HDF5 storage | `h5py` | Single-file, append-friendly, numpy-compatible; avoids re-running BirdNET inference on every training cycle |
| Feature extraction (V2) | FFT-based spectral statistics | `numpy` + `scipy.signal` | Zero additional dependencies beyond the ML stack; FFT is the only transform needed |
| ML training | PyTorch (CPU wheels) | `torch` (CPU via `download.pytorch.org/whl/cpu`) | MLP and AE are both native PyTorch; ONNX export via `torch.onnx.export` |
| Focal loss | Custom PyTorch loss | inline `FocalLoss` class in `pipeline/losses.py` | Standard implementation; no separate library needed |
| Data loading | Dataset + DataLoader | `torch.utils.data` | Native to PyTorch; handles weighted sampling for class imbalance |
| Evaluation | Sklearn metrics | `scikit-learn` | `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report` |
| Statistical tests | Paired t-test, Wilcoxon | `scipy.stats` | Paper specifies both tests; scipy has both natively |
| Configuration | YAML config | `PyYAML` | Paper specifies a `config.yaml` entry point |
| Plotting | ROC / PR curves, confusion matrices | `matplotlib` | Standard; produces the paper's Fig. 3–8 equivalents |
| Embedding visualisation | PCA, t-SNE | `scikit-learn` (PCA), `sklearn.manifold.TSNE` | Produces the paper's Fig. 7 |
| Feature attribution | Gradient×input + first-layer W | custom `attribution.py` | Produces the paper's Fig. 8; no external XAI library needed |
| Python package management | `uv` (per-service pyproject.toml) | `uv` | Fast resolver, lockfile, no host Python installs required |
| Containerisation | Docker + Docker Compose | Docker | DD-001 (git clone + bootstrap + compose watch = full setup) |
| Python linting/formatting | `ruff` | `ruff` | Replaces flake8+isort+black |
| Type checking | `mypy` strict | `mypy` | Per-service, run via `uv run mypy` |
| Testing | `pytest` + `pytest-cov` | `pytest` | Unit tests per module; coverage gate ≥75% |
| CI | GitHub Actions | `.github/workflows/ci.yml` | Lint, typecheck, test, build, security scan |
| Secret scanning | `gitleaks` | `gitleaks` | Run in CI on full git history |
| Dependency auditing | `pip-audit` | `pip-audit --disable-pip --no-deps` | DD-001-level: free, no external account |

**Python version:** 3.11 (matches the paper's CPU-only commodity hardware target; avoids Python 3.12 breaking changes in some audio libraries).

---

## 8. Novelty and Research Contribution

Per the paper, the primary contributions are:

1. **Noise Segregation V2**: a six-subframe, multi-feature noise scoring module augmented with harmonic structure checking and an MLP re-check pass (Bird Rescue) to recover false negatives before the classifier stage — the first published pipeline of this type for Indian soundscapes.
2. **Focal-loss MLP on BirdNET embeddings**: a binary classifier fine-tuned for Indian bird-versus-noise discrimination at the embedding level, addressing the domain gap without encoder retraining.
3. **Autoencoder OOD gate**: reconstruction-error-based rejection of inputs outside the bird embedding manifold, providing an interpretable safety layer with a single well-defined threshold τ_AE.
4. **Three-band router with hard-example mining**: a calibrated uncertainty band that surfaces ambiguous segments for active-learning annotation cycles.
5. **Full three-way quantitative benchmark on iBC53** with statistical significance testing — the first published end-to-end benchmark of this kind targeting Indian species and noise conditions.

---

## 9. Evaluation Metrics

All metrics are computed on the **held-out test split** (10% of iBC53, ≈1,004 segments, stratified by class) unless explicitly labelled otherwise.

| Component | Metric | Paper Target | Reproduced? |
|---|---|---|---|
| BirdNET Baseline | Accuracy | 0.440 | — |
| BirdNET Baseline | F1 | 0.598 | — |
| BirdNET Baseline | FNR | 57.4% | — |
| BirdNET Baseline | ROC-AUC | 0.724 | — |
| MLP Only | Accuracy | 0.999 | — |
| MLP Only | F1 | 1.000 | — |
| MLP Only | FNR | 0.04% | — |
| MLP Only | ROC-AUC | 0.998 | — |
| MLP Only | PR-AUC | 0.9999 | — |
| MLP + AE Gate | Accuracy | 0.995 | — |
| MLP + AE Gate | F1 | 0.998 | — |
| MLP + AE Gate | FNR | 0.43% | — |
| MLP + AE Gate | ROC-AUC | 0.994 | — |
| AE Gate | τ_AE | 0.16975 (P99 val bird MSE) | — |
| Statistical validation | Baseline → MLP (t-test) | p ≪ 0.001 | — |
| Statistical validation | Baseline → MLP (Wilcoxon) | p ≪ 0.001 | — |
| Statistical validation | MLP → MLP+AE (t-test) | p < 0.05 | — |
| Three-band router | Uncertain band fraction | ≈2% of segments | — |
| Hard-example mining | Likely FP exported to review | count (no target) | — |
| Hard-example mining | Likely FN exported to review | count (no target) | — |

**Reproduction target**: all metric values within ±1% absolute of the paper's Table I. If a metric exceeds this tolerance, document the discrepancy in §26 (Technical Debt) before any code change, not after.

**Evaluation is always held-out**: the test split must never be used for threshold selection (τ_AE is derived from the validation split), hyperparameter tuning, or early stopping decisions. Any metric computed on train or val data must be labelled explicitly in logs and in `design.md`.

---

## 10. Delivery Plan (Phased Build)

| Phase | Key Deliverables | Exit Criteria |
|---|---|---|
| **Phase 1: Engineering foundation** | Docker-first dev environment, `uv`-managed Python project, `ruff`/`mypy`/`pytest` quality gates, GitHub Actions CI, `scripts/bootstrap.sh`, `config.yaml` scaffold, skeleton modules with stubs | `docker compose watch` runs; all CI checks green on a trivial change |
| **Phase 2: Data pipeline** | iBC53 corpus download script, resampling, 3 s segmentation, RMS silence rejection, split manifest CSV, unit tests for each step | `python scripts/prepare_data.py --config config.yaml` produces a valid `data/manifest.csv`; `pytest pipeline/tests/test_data.py` passes |
| **Phase 3: Noise Segregation V2 + Bird Guard + Bird Rescue** | `pipeline/noise_segregation.py`, `pipeline/bird_guard.py`, `pipeline/bird_rescue.py`, unit tests with synthetic signals | V2 correctly classifies a pure sine wave as "bird-like" and white noise as "noise-like"; Bird Guard retains a harmonic test signal that V2 routes to noise |
| **Phase 4: BirdNET embedding extraction + HDF5 cache** | `pipeline/embedding.py`, `pipeline/cache.py`, integration test over 10 real iBC53 segments | HDF5 cache populated; embedding shape (1024,) verified; re-running the script is a no-op (idempotency test) |
| **Phase 5: MLP classifier + AE gate training** | `pipeline/model.py` (MLP + AE), `pipeline/losses.py` (FocalLoss), `pipeline/train.py`, `pipeline/evaluate.py`, training convergence logs | Training completes on CPU without OOM; val F1 ≥ 0.95 within 50 epochs; AE reconstruction histogram matches Fig. 6 shape; τ_AE computed and saved to `artifacts/config.json` |
| **Phase 6: Inference pipeline + three-band router + hard-example mining** | `pipeline/inference.py`, `pipeline/router.py`, `pipeline/mining.py`, `outputs/` directory structure | `python scripts/infer.py --config config.yaml --input data/test/` routes segments correctly; uncertain band ≈2%; review/ populated |
| **Phase 7: Benchmark reproduction + evaluation report** | `scripts/benchmark.py` (three-way comparison), `pipeline/stats.py` (t-test, Wilcoxon), `outputs/evaluation_report.json`, `outputs/plots/` | All §9 metrics within ±1% of paper Table I; statistical significance confirmed (p < 0.05); plots match Fig. 3–8 structure |
| **Phase 8: ONNX export + edge feasibility** | `scripts/export_onnx.py` (MLP + AE), `tests/test_onnx_roundtrip.py` | ONNX model produces identical outputs (within 1e-5 tolerance) to PyTorch model on 10 held-out test embeddings; estimated inference latency documented in §6.6 |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| iBC53 corpus not publicly available at time of build | Medium | Contact corresponding author (per paper §II-J); the paper's described segment-level manifest and labels are "available upon request." Use synthetic augmentation for pipeline development while awaiting access |
| BirdNET V2.4 TFLite weights not available from a stable URL | Low | `birdnet_analyzer` Python package bundles the weights; fallback to the BirdNET-Analyzer GitHub release assets |
| MLP fails to converge on CPU within acceptable wall-clock time | Low | MLP has ~650k parameters; 50 epochs on 7,500 training embeddings (each 4 KB) takes minutes on modern CPU. If slow: reduce to 30 epochs for development, use full 50 for the final benchmark run |
| AE reconstruction threshold τ_AE sensitive to random seed | Medium | Run 5 seeds, report mean ± std of τ_AE; if variance is high, use P95 instead of P99 and document the change as a DD |
| Noise Segregation V2 weights not validated beyond Bengaluru recordings | High (by design) | Document as a known limitation (§15); implement but do not claim generality; ablation study is the first Future Work item (§17) |
| Class imbalance (38.5:1) causes MLP to collapse to majority class | Low | Focal loss with γ=2.0 is the primary mitigation; also use WeightedRandomSampler in DataLoader as a secondary guard; monitor train FNR per epoch |
| Docker image for `tflite-runtime` fails to build on some platforms | Medium | Pin a specific `tflite-runtime` wheel URL; provide a `tensorflow` fallback path with a feature flag in `config.yaml` |

---

## 12. Limitations

The following limitations are explicitly acknowledged (directly from the paper):

- **Corpus scope**: Noise Segregation V2 weights and τ_AE are calibrated on iBC53 (Bengaluru urban/peri-urban, single seasonal window). Performance in the Western Ghats, Indo-Gangetic plain, Himalayan foothills, or monsoon season is unvalidated.
- **Partial synthetic noise**: the noise class in iBC53 includes segments generated from digital noise processes (pink, brown, white, band-limited white). These may not faithfully reproduce the long-tail spectral diversity of real field noise, particularly species-specific insect chirp patterns.
- **Within-corpus evaluation only**: the MLP was evaluated on held-out data from the same corpus and segmentation pipeline as the training data. High within-corpus performance does not guarantee cross-recorder or cross-seasonal generalisation.
- **Binary classification only**: the pipeline produces bird-versus-noise decisions; it does not perform species-level classification.
- **No temporal continuity**: segments are processed independently; temporal context across adjacent segments is not exploited.
- **V2 weight ablation not yet performed**: the feature-weight sensitivity of Noise Segregation V2 is qualitatively reasoned (§6.2 note) but not empirically quantified.

---

## 13. Future Work

1. **V2 weight ablation study**: systematic grid-search over feature weights; equal-weight and feature-drop variants; quantify sensitivity of full pipeline accuracy to deviations from the current weight vector (highest priority, §17).
2. **Cross-dataset and cross-regional validation**: apply the pipeline to independently collected Indian PAM corpora from the Western Ghats, Indo-Gangetic plain, and Himalayan foothills.
3. **Monsoon-season vs. dry-season field trials**: characterise seasonal sensitivity and establish τ_AE recalibration protocols.
4. **Species-level classification extension**: train a multi-class head on the cleaned, BirdNET-embedded bird corpus produced by this pipeline.
5. **Larger Indian bird datasets**: expand the corpus to cover a larger fraction of India's 1,300+ species across additional biogeographic zones.
6. **Real-time edge deployment**: port the full stack to TFLite (MLP + AE) on a Raspberry Pi-class device or AudioMoth companion compute module; validate latency and power profile for sustained off-grid PAM.
7. **Alternative OOD detection methods**: compare reconstruction-based AE gate against Mahalanobis distance and normalising flows on the BirdNET embedding space.

---

## 14. Repository Structure

```
.
├── config.yaml                  Single entry-point config (paths, hyperparameters, thresholds)
├── CLAUDE.md                    Documentation-first workflow rules for AI-assisted development
├── design.md                    This document — canonical engineering memory
├── to_do.md                     Companion action-item checklist (derived from §24)
│
├── scripts/
│   ├── bootstrap.sh             Host validation + local env prep (bash / macOS / Linux / WSL2)
│   ├── prepare_data.py          Download iBC53, resample, segment, silence-reject, write manifest
│   ├── extract_embeddings.py    Run BirdNET TFLite on all segments; populate HDF5 cache
│   ├── train.py                 Train MLP classifier + bird-only AE; save artifacts
│   ├── infer.py                 Run full inference pipeline on a folder of WAV files
│   ├── benchmark.py             Three-way comparison (BirdNET baseline, MLP Only, MLP+AE Gate)
│   └── export_onnx.py           Export trained MLP + AE to ONNX
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py                Pydantic-validated config loader (reads config.yaml)
│   ├── audio.py                 Resample, segment, RMS silence gate, WAV I/O helpers
│   ├── noise_segregation.py     Noise Segregation V2 (6-subframe, 5-feature weighted vote)
│   ├── bird_guard.py            Harmonic ratio + spectral peak prominence override check
│   ├── bird_rescue.py           MLP re-check for segments initially routed to noise class
│   ├── embedding.py             BirdNET V2.4 TFLite inference → 1024-D embedding
│   ├── cache.py                 HDF5 embedding cache (idempotent read/write)
│   ├── dataset.py               PyTorch Dataset over HDF5 cache + manifest CSV
│   ├── model.py                 FocalMLP class + BirdAutoencoder class (PyTorch)
│   ├── losses.py                FocalLoss implementation (γ=2.0, α=0.25)
│   ├── train_loop.py            Training loop, early stopping, checkpointing
│   ├── evaluate.py              Metrics computation (Accuracy, F1, FPR, FNR, ROC-AUC, PR-AUC)
│   ├── stats.py                 Paired t-test + Wilcoxon signed-rank on per-segment correctness
│   ├── router.py                Three-band router (τ_low=0.30, τ_high=0.70)
│   ├── mining.py                Hard-example mining (likely FP/FN export to review/)
│   ├── attribution.py           MLP feature attribution (grad×input + first-layer |W|)
│   └── plots.py                 ROC/PR curves, confusion matrices, AE MSE histogram, PCA/t-SNE
│
├── tests/
│   ├── test_audio.py            Unit: resample, segment, silence gate
│   ├── test_noise_segregation.py Unit: V2 on synthetic signals (sine, white noise, chirp)
│   ├── test_bird_guard.py       Unit: harmonic + peak override on synthetic harmonics
│   ├── test_cache.py            Unit: HDF5 idempotency, shape, dtype
│   ├── test_model.py            Unit: MLP + AE forward pass shapes; FocalLoss value
│   ├── test_evaluate.py         Unit: metric correctness on tiny hand-crafted predictions
│   ├── test_router.py           Unit: all three bands + OOD routing
│   ├── test_mining.py           Unit: FP/FN export logic
│   └── test_onnx_roundtrip.py  Integration: ONNX output matches PyTorch within 1e-5
│
├── data/
│   ├── raw/                     gitignored — raw iBC53 WAV files (downloaded by prepare_data.py)
│   ├── segments/                gitignored — 3 s resampled segments
│   ├── embeddings.h5            gitignored — HDF5 embedding cache
│   └── manifest.csv             gitignored — per-segment record (path, label, split, cached)
│
├── artifacts/
│   ├── mlp_best.pt              gitignored — best MLP checkpoint (val F1)
│   ├── ae_best.pt               gitignored — best AE checkpoint (val MSE)
│   ├── mlp.onnx                 gitignored — ONNX export of MLP
│   ├── ae.onnx                  gitignored — ONNX export of AE
│   └── config.json              τ_AE value, split seed, training hyperparameters used
│
├── outputs/
│   ├── bird/                    gitignored — high-confidence bird routing output
│   ├── uncertain/               gitignored — uncertain-band segments for review
│   ├── noise/                   gitignored — confident noise routing output
│   ├── review/
│   │   ├── likely_fp/           gitignored — hard-example mining: likely false positives
│   │   └── likely_fn/           gitignored — hard-example mining: likely false negatives
│   ├── manifest_results.csv     gitignored — full per-segment inference record
│   ├── evaluation_report.json   committed (small, human-readable benchmark results)
│   └── plots/                   committed — ROC/PR curves, confusion matrices, Fig. 3–8 equivalents
│
├── docs/
│   ├── architecture/            Standalone architecture notes once they outgrow §6
│   ├── adr/                     ADRs promoted from the §15 DD-NNN table
│   └── research/                Research notes, paper references, ablation results
│
├── docker/
│   └── Dockerfile               Single-service image (python:3.11-slim + uv + all deps)
│
├── .github/
│   ├── workflows/ci.yml         lint (ruff), typecheck (mypy), test (pytest), build, security
│   └── dependabot.yml
│
├── .husky/                      pre-commit (lint-staged), commit-msg (commitlint)
├── pyproject.toml               uv-managed project (dependencies, ruff config, mypy config, pytest config)
├── uv.lock                      Lockfile (committed for deterministic installs)
└── .env.example                 Documents all environment variables (no secrets committed)
```

---

## 15. Design Decisions

| ID | Decision | Rationale |
|---|---|---|
| DD-001 | Docker-first onboarding: `git clone && ./scripts/bootstrap.sh && docker compose watch` is the entire setup | No host Python/CUDA/tflite installs required; deterministic across contributor machines; matches the paper's "commodity CPU hardware, cross-platform" claim |
| DD-002 | Single Python service (no microservice split) | This is a research ML pipeline, not a production web service. One `pyproject.toml`, one `uv.lock`, one Docker image; simpler than a polyglot monorepo for a single-language project |
| DD-003 | HDF5 embedding cache (`pipeline/cache.py`) decouples BirdNET inference from classifier training | BirdNET TFLite inference is the dominant per-segment cost (~300–500 ms on CPU); if cache didn't exist, every hyperparameter change would re-run BirdNET over all 10,044 segments — an unnecessary ~1.5–2 hour penalty. Cache is idempotent (re-running `extract_embeddings.py` is a no-op on existing entries). |
| DD-004 | `config.yaml` is the single entry point for all hyperparameters and paths | Paper states "reproducible via a single `config.yaml` entry point." No magic numbers in source code — every threshold (τ_low, τ_high, τ_AE, silence_db, harmonic_ratio_threshold, peak_median_threshold, focal_gamma, etc.) must be declared in `config.yaml` and loaded via `pipeline/config.py`'s Pydantic model. |
| DD-005 | τ_AE is derived from the **validation split** and saved to `artifacts/config.json`, never re-derived from the test split | OOD threshold selection must not touch the test split (would leak information). τ_AE = P99 of val bird MSE; saved alongside model artifacts so inference is fully reproducible from artifacts alone. |
| DD-006 | Focal loss γ=2.0 is the primary class-imbalance mitigation; `WeightedRandomSampler` is secondary | The paper specifies focal loss as the design choice. WeightedRandomSampler is added as a secondary guard to ensure the rare noise class appears in every batch, but the focal loss modulating factor (1−p)^γ is what concentrates gradient signal on hard boundary cases. |
| DD-007 | Noise Segregation V2 feature weights (0.25/0.30/0.15/0.15/0.15) are treated as **unvalidated hyperparameters** until the ablation study (§17 Future Work #1) is completed | The paper sets these qualitatively. This codebase documents them as `config.yaml` entries, not hardcoded constants, so they can be swept in the ablation without touching source code. A systematic ablation is the top-priority future experiment. |
| DD-008 | Stratified split is saved to `data/split.csv` with a fixed seed from `config.yaml` | Reproducibility requires that the exact same 75/15/10 split can be reconstructed from the seed alone. The split CSV is the ground truth; `prepare_data.py` writes it once and never overwrites it unless `--force-resplit` is passed explicitly. |
| DD-009 | `pipeline/embedding.py` supports both `tflite-runtime` and `tensorflow` (lite subset) via a feature flag in `config.yaml` (`embedding.backend: "tflite" | "tensorflow"`) | `tflite-runtime` is the lightweight preferred backend (matching the paper's "TFLite runtime" claim); `tensorflow` is the fallback for platforms where `tflite-runtime` wheels are unavailable. The flag is resolved once at startup, not per-segment. |
| DD-010 | Synthetic noise data (for augmenting the noise class) is generated by a separate `pipeline/synthetic.py` module, opt-in via `config.yaml` (`data.augment_noise: true`), never a default | Matches the policy from CLAUDE.md's ML-specific standards: synthetic data is opt-in, never the default. Must log a clear warning when active. iBC53's existing synthetic noise segments (pink/brown/white/band-limited) are part of the corpus itself and are not covered by this flag. |
| DD-011 | PyTorch CPU-only wheels pinned via `[tool.uv.sources]` in `pyproject.toml` | Training and inference run on CPU-only hardware per §7 and §6.6; CUDA wheels would add multiple GB to the Docker image for no benefit. |
| DD-012 | `scripts/benchmark.py` is a standalone script that reads from `data/manifest.csv` + `artifacts/config.json` and writes to `outputs/evaluation_report.json` | The benchmark is a read-only operation over the data and model artifacts — it must not trigger any training or re-evaluation. Running the benchmark twice must produce identical results. |
| DD-013 | ONNX export covers both MLP and AE; roundtrip test (`tests/test_onnx_roundtrip.py`) verifies output agreement within 1e-5 | Edge deployment (future work, §17 #6) requires ONNX or TFLite format. The roundtrip test must be run as part of CI after every change to `pipeline/model.py`. |
| DD-014 | `outputs/evaluation_report.json` and `outputs/plots/` are committed to the repository; all other `outputs/` subdirectories are gitignored | The benchmark results and figures are the primary deliverables of the research pipeline — they must be version-controlled alongside the code that produced them. Routed audio segments (bird/, noise/, uncertain/) are large and derived artifacts, not source-of-truth. |
| DD-015 | `artifacts/config.json` is committed alongside trained model artifacts' metadata (τ_AE, seed, hyperparameters) but model `.pt` and `.onnx` files are gitignored | Model weights are too large for git; τ_AE and training config are tiny and required for reproducing inference results without re-training. Git LFS is an option for model weights if the team wants to track them; document the choice here when made. |

---

## 16. Dependencies

**Runtime:**
`torch` (CPU, via `[tool.uv.sources]`), `tflite-runtime` (BirdNET embedding; `tensorflow` fallback), `numpy`, `scipy`, `scikit-learn`, `librosa`, `soundfile`, `h5py`, `PyYAML`, `pydantic`, `matplotlib`.

**Dev / test only:**
`pytest`, `pytest-cov`, `mypy`, `ruff`, `onnx`, `onnxruntime` (ONNX roundtrip tests).

**Optional:**
`pandas` (manifest CSV manipulation convenience; `csv` stdlib is sufficient for core pipeline use).

No dependency has been added without a corresponding line item above and, where non-obvious, a DD in §15.

---

## 17. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CONFIG_PATH` | `config.yaml` | Path to the pipeline config file; overrides the default for multi-config experiments |
| `BIRDNET_MODEL_PATH` | `models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite` | Path to the BirdNET V2.4 TFLite model weights (downloaded by `bootstrap.sh`, not committed) |
| `DATA_DIR` | `data/` | Root of the data directory tree |
| `ARTIFACTS_DIR` | `artifacts/` | Root of the model artifacts directory |
| `OUTPUTS_DIR` | `outputs/` | Root of the inference routing output directory |
| `RANDOM_SEED` | `42` | Global random seed for reproducible splits and training |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG` for verbose training output) |

All variables are documented in `.env.example`. No secrets are required for the core pipeline (BirdNET weights are downloaded publicly; iBC53 is requested from the corresponding author).

---

## 18. External Integrations

| Integration | Status | Notes |
|---|---|---|
| iBC53 corpus | Required; request-only at time of writing | Contact corresponding author (Gupta et al., bioRxiv 2025.07.21.665881); segment-level manifest available upon request pending formal data release |
| BirdNET V2.4 TFLite model | Required; publicly available | Via `birdnet_analyzer` Python package (`pip install birdnet`) or direct download from the BirdNET-Analyzer GitHub repository releases |
| GitHub Actions | Active | CI (`ci.yml`), Dependabot (`dependabot.yml`) |
| Docker Hub / base images | Active (pull-only) | `python:3.11-slim` — no images published, only pulled |

---

## 19. Infrastructure & Deployment

**Local development:** `docker compose watch` — single service (`pipeline`) rebuilds/syncs automatically on file changes. No bind-mounted `volumes:` for source code (avoids the host/container `node_modules`-equivalent collision for Python `.venv`). `watch` syncs deltas into the already-built dev image.

**No production deployment in Phase 1** — the pipeline runs as a CLI tool. Future edge deployment (§17 #6) would target a Raspberry Pi-class device running the ONNX-exported models via `onnxruntime` or TFLite; the Docker Compose stack would be replaced by a systemd service or equivalent.

**Health checks:** none for CLI tools. The Docker image's `CMD` is `python scripts/benchmark.py --help` (a no-op that confirms the environment is correct without running training).

---

## 20. Domain-Specific Components

- **Noise Segregation V2** (`pipeline/noise_segregation.py`): the six-subframe weighted-feature scorer. Treats each 3 s segment as six 0.5 s subframes; extracts ZCR, spectral flatness, centroid flag, centroid std, and insect periodicity flag per subframe; computes per-subframe noise score via the weighted sum (Eq. 1 in the paper); majority-votes across six subframes.
- **Bird Guard** (`pipeline/bird_guard.py`): post-vote harmonic override. Accepts a segment that V2 has routed to the noise class; computes harmonic ratio and spectral peak-to-median ratio; if either exceeds its threshold, returns "bird-like" regardless of V2's vote.
- **Bird Rescue** (`pipeline/bird_rescue.py`): MLP-based second-chance classifier. A small binary MLP (128 → 64 → 1, sigmoid) trained on BirdNET embeddings of segments that were initially noise-routed. If P(bird) ≥ τ_rescue (default 0.5, configurable), the segment is rescued back to species folders.
- **BirdNET embedding extractor** (`pipeline/embedding.py`): wraps TFLite runtime inference on BirdNET V2.4; extracts the penultimate global average pooling layer (1024-D); normalises to unit L2 norm at inference time.
- **HDF5 embedding cache** (`pipeline/cache.py`): maps `segment_id` (str) → `embedding` (float32 array of shape (1024,)). Idempotent: re-running `extract_embeddings.py` skips segments already in the cache. Thread-safe read; write is single-writer (no concurrent training runs).
- **FocalMLP** (`pipeline/model.py`): 1024→512→256→1, BatchNorm + ReLU + Dropout(0.3) between each linear layer, sigmoid output. Matches the paper's §III-F specification exactly.
- **BirdAutoencoder** (`pipeline/model.py`): symmetric 1024→512→128→512→1024 bottleneck AE. Trained on bird embeddings only. τ_AE derived from P99 reconstruction MSE on val bird embeddings.
- **Three-band router** (`pipeline/router.py`): routes each (segment, mlp_prob, ood_flag) tuple to bird/, uncertain/, or noise/ output directory. Implements the paper's τ_low=0.30, τ_high=0.70 boundaries.
- **Hard-example miner** (`pipeline/mining.py`): scans the noise/ and uncertain/ output directories; exports likely FPs (noise-labelled segments with MLP prob > 0.5) and likely FNs (bird-labelled segments with prob < 0.5 or AE-rejected) to review/.
- **Benchmark runner** (`scripts/benchmark.py`): computes the three-way comparison (BirdNET baseline / MLP Only / MLP+AE Gate) on the full manifest, calls `pipeline/stats.py` for significance testing, writes `outputs/evaluation_report.json`, and calls `pipeline/plots.py` to produce Figs. 3–8 equivalents.

---

## 21. Current Phase

**Phase 1: Engineering Foundation** — Docker-first dev environment, `uv`-managed Python project, quality gates, CI/CD skeleton, and all module stubs. No application logic implemented yet.

---

## 22. Current Milestone

Setting up the build foundation so that `docker compose watch` runs, all CI checks pass on stubs, and the repository structure matches §14 exactly.

---

## 23. Completed Milestones

_(None yet — this is the initial design document.)_

---

## 24. Pending Tasks

- [ ] Procure the iBC53 corpus (email corresponding author per §18)
- [ ] Download BirdNET V2.4 TFLite model weights (`bootstrap.sh` should handle this automatically)
- [ ] **Phase 1**: engineering foundation — `pyproject.toml`, `uv.lock`, `Dockerfile`, `docker-compose.yml`, `scripts/bootstrap.sh`, `config.yaml`, all skeleton modules, CI pipeline (`ci.yml`)
- [ ] **Phase 2**: data pipeline — `scripts/prepare_data.py`, `pipeline/audio.py`, `data/manifest.csv` generation, `tests/test_audio.py`
- [ ] **Phase 3**: Noise Segregation V2 + Bird Guard + Bird Rescue — `pipeline/noise_segregation.py`, `pipeline/bird_guard.py`, `pipeline/bird_rescue.py`, unit tests on synthetic signals
- [ ] **Phase 4**: BirdNET embedding extraction + HDF5 cache — `pipeline/embedding.py`, `pipeline/cache.py`, `scripts/extract_embeddings.py`, idempotency test
- [ ] **Phase 5**: MLP + AE training — `pipeline/model.py`, `pipeline/losses.py`, `pipeline/train_loop.py`, `scripts/train.py`; convergence confirmed on CPU; τ_AE derived and saved
- [ ] **Phase 6**: inference pipeline + router + mining — `pipeline/inference.py`, `pipeline/router.py`, `pipeline/mining.py`, `scripts/infer.py`
- [ ] **Phase 7**: benchmark reproduction — `scripts/benchmark.py`, `pipeline/stats.py`, `pipeline/plots.py`; §9 metrics all within ±1% of paper Table I
- [ ] **Phase 8**: ONNX export + roundtrip test — `scripts/export_onnx.py`, `tests/test_onnx_roundtrip.py`
- [ ] Commit `outputs/evaluation_report.json` and `outputs/plots/` once benchmark passes
- [ ] Open a PR for each phase so CI validates before merge

---

## 25. Known Issues

_(None yet — pre-implementation.)_

---

## 26. Technical Debt

_(None yet — pre-implementation. This section will track gaps discovered during implementation, particularly any reproduction discrepancies versus paper Table I.)_

---

## 27. Future Improvements

See §13 (Future Work). Priority order:
1. V2 weight ablation study (highest — empirically validates the current weight choice)
2. Cross-dataset/cross-regional validation
3. Monsoon vs. dry-season field trials
4. Species-level classification extension
5. Larger Indian bird dataset integration
6. Real-time edge deployment (Raspberry Pi / AudioMoth)
7. Alternative OOD detection methods (Mahalanobis, normalising flows)

---

## 28. Development Log

### Entry 1 — Pre-implementation (logical project time: 2026-09-19)

**Task completed:** Authored `design.md` (this document) and `CLAUDE.md` based on the research paper "Noise-Aware Pipeline for Indian Bird Sound Classification Using BirdNET Embeddings, Focal-Loss MLP, and Autoencoder Gating" (Kolur et al., DSU). No code written yet — this is the documentation-first foundation required by CLAUDE.md's workflow before any implementation begins.

**Files created:** `CLAUDE.md`, `design.md`.

**Files modified:** None.

**Files deleted:** None.

**Reason for change:** Documentation-first workflow per CLAUDE.md: `design.md` must be created and reflect the full intended design before the first line of application code is written. A new engineer (or AI assistant) should be able to understand what to build, why, and how by reading only this document.

**Architectural decisions:** DD-001 through DD-015 (§15).

**Remaining work:** All of §24 Pending Tasks — starting with Phase 1 (engineering foundation).

**Known issues:** iBC53 corpus access is gated behind a request to the corresponding author (§18). BirdNET V2.4 TFLite weights are publicly available but must be downloaded by `bootstrap.sh` (not committed). Both blockers affect Phase 4 (embedding extraction) onward; Phases 1–3 (audio preprocessing, noise segmentation) can proceed against synthetic signals.

**Recommended next task:** Implement Phase 1 (engineering foundation): `pyproject.toml`, `uv.lock`, `Dockerfile`, `docker-compose.yml`, `scripts/bootstrap.sh`, `config.yaml` scaffold, all module stubs with correct signatures and `NotImplementedError` bodies, and the GitHub Actions CI pipeline. Verify with `docker compose watch` running and all CI checks green before proceeding to Phase 2.

---

## 29. Current Repository State

- **Branch:** `main` (no commits yet — pre-implementation).
- **Files present:** `CLAUDE.md`, `design.md` only.
- **Status:** Pre-implementation. No code exists. This document is the complete specification of what will be built.
- **Next action:** Phase 1 (engineering foundation) — see §24 Pending Tasks.
