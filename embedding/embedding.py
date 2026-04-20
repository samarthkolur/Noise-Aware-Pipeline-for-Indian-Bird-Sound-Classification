"""
embedding.py — Embedding extraction pipeline.

Provides a unified interface for extracting fixed-size audio embeddings
using BirdNET V2.4 (via TFLite). Embeddings are stored in HDF5 (one file
per species) with
a global manifest CSV for fast downstream querying.

Storage layout:
    embeddings_dir/
        manifest.csv                  # global index
        <species>/
            embeddings.h5             # datasets: "embeddings" (N, D), "filenames" (N,)
"""

from __future__ import annotations

import csv
from collections import Counter
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


# ── BirdNET V2.4 (TFLite) — required for embedding.model_name: birdnet ─────
REQUIRED_BIRDNET_VERSION = "V2.4"
REQUIRED_BIRDNET_TFLITE_NAME = "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"


class BirdNETModelNotFoundError(FileNotFoundError):
    """The BirdNET ``.tflite`` file is missing; no fallback encoder is used."""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _birdnet_missing_message(*, searched: List[str], extra: str = "") -> str:
    local_dir = _project_root() / "birdnet_weights"
    expected_file = local_dir / REQUIRED_BIRDNET_TFLITE_NAME
    lines = [
        "BirdNET TFLite model is required but was not found.",
        f"  Required BirdNET version: {REQUIRED_BIRDNET_VERSION}",
        f"  Required filename: {REQUIRED_BIRDNET_TFLITE_NAME}",
        "  Obtain the file by either:",
        "    (1) pip install birdnetlib ai-edge-litert  — model ships under site-packages/birdnetlib/...",
        "    (2) Download official bundle: https://birdnet-team.github.io/BirdNET-Analyzer/models.html",
        "        (V2.4 → BirdNET-Analyzer-V2.4.zip), extract the .tflite above.",
        f"  Recommended project path: {expected_file}",
        "  Then set embedding.birdnet_model_path: auto  OR  an absolute path to the .tflite file.",
        "  Searched locations:",
    ]
    for s in searched:
        lines.append(f"    - {s}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _resolve_birdnet_model_path(cfg: dict) -> Path:
    """Resolve path to BirdNET TFLite; raises BirdNETModelNotFoundError if missing.

    When ``birdnet_model_path`` is ``auto``: (1) project-local
    ``birdnet_weights/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`` if present;
    else (2) the file bundled with ``birdnetlib`` under site-packages.
    """
    emb_cfg = cfg.get("embedding", {})
    raw = emb_cfg.get("birdnet_model_path", None)
    local_weights = _project_root() / "birdnet_weights" / REQUIRED_BIRDNET_TFLITE_NAME

    if raw and str(raw).strip() and str(raw) != "auto":
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise BirdNETModelNotFoundError(
                _birdnet_missing_message(
                    searched=[str(p)],
                    extra="  Config embedding.birdnet_model_path points to a file that does not exist.",
                )
            )
        if p.name != REQUIRED_BIRDNET_TFLITE_NAME:
            logger.warning(
                "BirdNET weights file is named %s; expected %s — continuing anyway.",
                p.name,
                REQUIRED_BIRDNET_TFLITE_NAME,
            )
        print("  [BirdNET] Model source: embedding.birdnet_model_path (explicit)")
        return p

    # auto: prefer project birdnet_weights/, then birdnetlib bundle
    searched: List[str] = [str(local_weights.resolve())]
    if local_weights.is_file():
        print("  [BirdNET] Model source: project birdnet_weights/ (local copy)")
        return local_weights.resolve()

    bundled = _find_birdnet_model()
    if bundled:
        searched.append(f"birdnetlib bundle: {bundled}")
        print("  [BirdNET] Model source: birdnetlib site-packages (bundled .tflite)")
        return Path(bundled).resolve()

    expected_pkg = _birdnet_expected_package_path()
    if expected_pkg:
        searched.append(f"birdnetlib (expected path): {expected_pkg}")

    raise BirdNETModelNotFoundError(_birdnet_missing_message(searched=searched))


# ════════════════════════════════════════════════════════════
#  Data classes
# ════════════════════════════════════════════════════════════

@dataclass
class EmbeddingRecord:
    """Metadata for a single embedding vector."""
    source_file: str
    species: str
    segment_index: int
    embedding_dim: int
    model_name: str
    sample_rate: int
    duration_sec: float


# ════════════════════════════════════════════════════════════
#  Abstract base encoder
# ════════════════════════════════════════════════════════════

class BaseEncoder(ABC):
    """Interface that every embedding backend must implement."""

    @abstractmethod
    def encode(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Return a 1-D embedding vector from a raw waveform.

        Args:
            waveform: 1-D float32 numpy array (mono).
            sr: Sample rate of the waveform.

        Returns:
            np.ndarray of shape (embedding_dim,).
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable encoder name."""


# ════════════════════════════════════════════════════════════
#  BirdNET encoder (TFLite via ai-edge-litert)
# ════════════════════════════════════════════════════════════

def _birdnet_expected_package_path() -> Optional[str]:
    """Path where birdnetlib ships ``BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`` (may not exist)."""
    import importlib.util
    import os

    spec = importlib.util.find_spec("birdnetlib")
    if spec is None or spec.origin is None:
        return None
    pkg_dir = os.path.dirname(spec.origin)
    return os.path.join(
        pkg_dir,
        "models",
        "analyzer",
        REQUIRED_BIRDNET_TFLITE_NAME,
    )


def _find_birdnet_model() -> Optional[str]:
    """Auto-detect the BirdNET TFLite model bundled with birdnetlib."""
    candidate = _birdnet_expected_package_path()
    if candidate is None:
        return None
    import os

    return candidate if os.path.isfile(candidate) else None


class BirdNETEncoder(BaseEncoder):
    """Extract 1024-D embeddings from BirdNET V2.4 (TFLite).

    Uses the ``ai-edge-litert`` runtime to load the official
    ``BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`` model.  Embeddings
    are read from an *internal* tensor (``GLOBAL_AVG_POOL/Mean``,
    index 546) which holds the 1024-D penultimate-layer activations.

    Requirements:
        pip install birdnetlib ai-edge-litert
    """

    _EXPECTED_SR = 48_000
    _EXPECTED_DUR = 3.0
    _EMB_DIM = 1024

    def __init__(self, cfg: dict) -> None:
        resolved = _resolve_birdnet_model_path(cfg)

        print("Loading BirdNET model from:", str(resolved))
        self._model_path = str(resolved)
        self._interpreter = self._load_interpreter(self._model_path)
        self._input_details = self._interpreter.get_input_details()

        # Dynamically find the embedding tensor by name + shape
        self._emb_tensor_index = self._find_embedding_tensor()

        print("Model loaded successfully")
        print(
            "[BirdNET] Resolved model path:",
            self._model_path,
            "| load OK | encoder=BirdNET",
        )
        logger.info(
            f"BirdNET V2.4 loaded ✓  (model: {Path(self._model_path).name}, "
            f"embedding: tensor {self._emb_tensor_index} → {self._EMB_DIM}D)"
        )

        # Collect a few embeddings during extract to verify they vary across segments
        self._emb_debug_buffer: List[np.ndarray] = []
        self._cross_sample_reported: bool = False

    # ── BaseEncoder interface ──────────────────────────────

    def encode(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract 1024-D embedding from a single 3-second waveform."""
        waveform = self._prepare_input(waveform, sr)
        input_data = waveform.reshape(self._input_details[0]["shape"]).astype(
            np.float32
        )
        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        # Read the penultimate-layer embedding (NOT the classification output)
        raw = self._interpreter.get_tensor(self._emb_tensor_index)
        embedding = raw.flatten().astype(np.float32)

        # First segment: shape, sample values, mean/std; must not be all zeros
        if not getattr(self, "_embedding_debug_printed", False):
            self._embedding_debug_printed = True
            raw_arr = np.asarray(raw)
            mu = float(embedding.mean())
            sigma = float(embedding.std())
            all_zero = bool(np.all(embedding == 0))
            print(
                "[BirdNET embedding] raw tensor shape:",
                raw_arr.shape,
                "→ flat shape:",
                embedding.shape,
                embedding.dtype,
            )
            print("[BirdNET embedding] first embedding sample (first 16 values):", embedding[:16])
            print(
                "[BirdNET embedding] mean:",
                mu,
                "std:",
                sigma,
                "min:",
                float(embedding.min()),
                "max:",
                float(embedding.max()),
            )
            print(
                "[BirdNET embedding] all zeros:",
                all_zero,
                "all finite:",
                bool(np.isfinite(embedding).all()),
                "dim_ok:",
                embedding.shape == (self._EMB_DIM,),
            )
            if all_zero or sigma < 1e-12:
                print(
                    "[BirdNET embedding] WARNING: vector is all zeros or ~constant — "
                    "model output may not be valid."
                )

        # Cross-sample check (up to 5 segments)
        if len(self._emb_debug_buffer) < 5:
            self._emb_debug_buffer.append(embedding.copy())
        if len(self._emb_debug_buffer) == 5 and not self._cross_sample_reported:
            self._report_cross_sample_embeddings(5)
            self._cross_sample_reported = True

        return embedding

    def _report_cross_sample_embeddings(self, n: int) -> None:
        """Print whether embeddings differ across the first *n* segments."""
        buf = self._emb_debug_buffer[:n]
        stack = np.stack(buf, axis=0)
        std_per_dim = np.std(stack, axis=0)
        mean_std_across_samples = float(np.mean(std_per_dim))
        l2_between_consecutive = np.linalg.norm(stack[1:] - stack[:-1], axis=1)
        identical = np.all(np.abs(stack - stack[0]) < 1e-7)

        print(
            f"\n[BirdNET embedding] cross-sample check (n={n} segments): "
            f"mean(std per dimension across samples)={mean_std_across_samples:.6f}"
        )
        print(
            "[BirdNET embedding] L2 distance between consecutive segments: "
            f"min={float(l2_between_consecutive.min()):.6f} "
            f"max={float(l2_between_consecutive.max()):.6f} "
            f"mean={float(l2_between_consecutive.mean()):.6f}"
        )
        if identical or mean_std_across_samples < 1e-10:
            print(
                "[BirdNET embedding] ERROR: embeddings are constant across samples — "
                "model is likely not working correctly."
            )
        else:
            print(
                "[BirdNET embedding] OK: embeddings vary across samples "
                "(expected for different audio segments)."
            )

    def flush_cross_sample_embedding_debug(self) -> None:
        """If fewer than 5 segments were processed, still report when n >= 2."""
        if self._cross_sample_reported:
            return
        n = len(self._emb_debug_buffer)
        if n >= 2:
            self._report_cross_sample_embeddings(n)
            self._cross_sample_reported = True

    @property
    def embedding_dim(self) -> int:
        return self._EMB_DIM

    @property
    def name(self) -> str:
        return "birdnet_v2.4"

    # ── Private ────────────────────────────────────────────

    @staticmethod
    def _load_interpreter(model_path: str):
        """Load TFLite interpreter with XNNPACK disabled.
        
        XNNPACK delegates fuse and discard intermediate tensors, which
        prevents us from reading the penultimate embedding layer.
        Disabling it with BUILTIN_WITHOUT_DEFAULT_DELEGATES preserves
        all internal tensor data after invoke().
        """
        try:
            from ai_edge_litert.interpreter import Interpreter
            from ai_edge_litert import interpreter as litert
            interp = Interpreter(
                model_path=model_path,
                num_threads=4,
                experimental_op_resolver_type=litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
                import tflite_runtime.interpreter as tflite_rt
                interp = Interpreter(
                    model_path=model_path,
                    num_threads=4,
                    experimental_op_resolver_type=tflite_rt.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                )
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                    interp = Interpreter(model_path=model_path, num_threads=4)
                except ImportError:
                    raise ImportError(
                        "No TFLite runtime found. Install one of:\n"
                        "  pip install ai-edge-litert   (recommended)\n"
                        "  pip install tflite-runtime\n"
                        "  pip install tensorflow-cpu"
                    )
        interp.allocate_tensors()
        return interp

    def _find_embedding_tensor(self) -> int:
        """Dynamically locate the 1024-D embedding tensor by name + shape.

        Searches for tensors named ``GLOBAL_AVG_POOL`` with shape ``(1, 1024)``.
        Falls back to the last tensor with shape ``(*, 1024)`` if not found.
        """
        details = self._interpreter.get_tensor_details()
        # Priority 1: exact name match
        for d in details:
            if "GLOBAL_AVG_POOL" in d["name"] and tuple(d["shape"])[-1] == self._EMB_DIM:
                return d["index"]
        # Priority 2: any tensor with shape (*, 1024) that isn't the input
        input_indices = {inp["index"] for inp in self._input_details}
        candidates = []
        for d in details:
            shape = tuple(d["shape"])
            if len(shape) == 2 and shape[-1] == self._EMB_DIM and d["index"] not in input_indices:
                candidates.append(d)
        if candidates:
            # Pick the one with the highest index (deepest in the network)
            return max(candidates, key=lambda x: x["index"])["index"]
        raise RuntimeError(
            f"Could not find a {self._EMB_DIM}-D embedding tensor in the "
            "BirdNET TFLite model. The model may be a different version."
        )

    def _prepare_input(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Resample to 48 kHz and pad/trim to exactly 3 seconds."""
        if sr != self._EXPECTED_SR:
            import librosa
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self._EXPECTED_SR
            )
        expected_len = int(self._EXPECTED_SR * self._EXPECTED_DUR)
        if len(waveform) > expected_len:
            waveform = waveform[:expected_len]
        elif len(waveform) < expected_len:
            waveform = np.pad(waveform, (0, expected_len - len(waveform)))
        return waveform.astype(np.float32)


# ════════════════════════════════════════════════════════════
#  Encoder factory
# ════════════════════════════════════════════════════════════

def build_encoder(cfg: dict) -> BaseEncoder:
    """Instantiate the production BirdNET encoder.

    Raises ``BirdNETModelNotFoundError`` if the TFLite file is missing.
    """
    model_name = cfg["embedding"]["model_name"]
    if model_name in ("birdnet", "birdnet_v2.4"):
        return BirdNETEncoder(cfg)
    raise ValueError(
        f"Unsupported embedding model: '{model_name}'. "
        "This production pipeline requires embedding.model_name: 'birdnet'."
    )


# ════════════════════════════════════════════════════════════
#  HDF5 storage helpers
# ════════════════════════════════════════════════════════════

class EmbeddingStore:
    """Manages per-species HDF5 embedding files and a global manifest CSV.

    Storage layout per species:
        embeddings_dir/<species>/embeddings.h5
            datasets:
                embeddings  — float32 (N, D)
                filenames   — variable-length UTF-8 strings (N,)

    Global manifest:
        embeddings_dir/manifest.csv
            columns: species, source_file, segment_index, embedding_dim,
                     model_name, sample_rate, duration_sec, hdf5_path, hdf5_row
    """

    def __init__(self, embeddings_dir: Path, embedding_dim: int) -> None:
        self.root = Path(embeddings_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._manifest_rows: List[Dict] = []
        self._h5_handles: Dict[str, Tuple[h5py.File, int]] = {}

    def add(
        self,
        embedding: np.ndarray,
        record: EmbeddingRecord,
    ) -> None:
        """Append an embedding vector to the species HDF5 file."""
        species = record.species
        species_dir = self.root / species
        species_dir.mkdir(parents=True, exist_ok=True)
        h5_path = species_dir / "embeddings.h5"

        if species not in self._h5_handles:
            self._h5_handles[species] = self._open_h5(h5_path)

        h5f, row_idx = self._h5_handles[species]
        self._append_to_h5(h5f, embedding, record.source_file, row_idx)
        self._h5_handles[species] = (h5f, row_idx + 1)

        self._manifest_rows.append(
            {
                "species": species,
                "source_file": record.source_file,
                "segment_index": record.segment_index,
                "embedding_dim": record.embedding_dim,
                "model_name": record.model_name,
                "sample_rate": record.sample_rate,
                "duration_sec": record.duration_sec,
                "hdf5_path": str(h5_path),
                "hdf5_row": row_idx,
            }
        )

    def flush(self) -> Path:
        """Close all HDF5 files and write the manifest CSV. Returns manifest path."""
        for species, (h5f, _) in self._h5_handles.items():
            h5f.close()
        self._h5_handles.clear()

        manifest_path = self.root / "manifest.csv"
        fieldnames = [
            "species", "source_file", "segment_index", "embedding_dim",
            "model_name", "sample_rate", "duration_sec", "hdf5_path", "hdf5_row",
        ]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._manifest_rows)

        logger.info(
            f"Manifest written → {manifest_path} ({len(self._manifest_rows)} rows)"
        )
        return manifest_path

    # ── Private ────────────────────────────────────────────

    def _open_h5(self, path: Path) -> Tuple[h5py.File, int]:
        """Open or create an HDF5 file with resizable datasets."""
        if path.exists():
            h5f = h5py.File(str(path), "a")
            existing = h5f["embeddings"].shape[0] if "embeddings" in h5f else 0
            return h5f, existing

        h5f = h5py.File(str(path), "w")
        h5f.create_dataset(
            "embeddings",
            shape=(0, self.embedding_dim),
            maxshape=(None, self.embedding_dim),
            dtype="float32",
            chunks=(64, self.embedding_dim),
            compression="gzip",
            compression_opts=4,
        )
        dt = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset(
            "filenames",
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
        )
        return h5f, 0

    @staticmethod
    def _append_to_h5(
        h5f: h5py.File,
        embedding: np.ndarray,
        filename: str,
        row_idx: int,
    ) -> None:
        """Append a single row to the embeddings & filenames datasets."""
        emb_ds = h5f["embeddings"]
        fn_ds = h5f["filenames"]

        new_size = row_idx + 1
        emb_ds.resize(new_size, axis=0)
        fn_ds.resize(new_size, axis=0)

        emb_ds[row_idx] = embedding
        fn_ds[row_idx] = filename


# ════════════════════════════════════════════════════════════
#  Load embeddings back from HDF5
# ════════════════════════════════════════════════════════════

def load_species_embeddings(h5_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load all embeddings for a single species from its HDF5 file.

    Returns:
        (embeddings, filenames) — embeddings shape (N, D), filenames list of N strings.
    """
    with h5py.File(str(h5_path), "r") as h5f:
        embeddings = h5f["embeddings"][:]
        filenames = [fn.decode() if isinstance(fn, bytes) else fn for fn in h5f["filenames"][:]]
    return embeddings, filenames


def load_all_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load embeddings from every species under ``embeddings_dir``.

    Returns:
        (embeddings, species_labels, filenames) — all as flat arrays/lists.
    """
    all_embs, all_species, all_fnames = [], [], []
    for species_dir in sorted(embeddings_dir.iterdir()):
        h5_path = species_dir / "embeddings.h5"
        if not h5_path.exists():
            continue
        embs, fnames = load_species_embeddings(h5_path)
        all_embs.append(embs)
        all_species.extend([species_dir.name] * len(fnames))
        all_fnames.extend(fnames)

    if not all_embs:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.vstack(all_embs), all_species, all_fnames


# ════════════════════════════════════════════════════════════
#  Extraction pipeline
# ════════════════════════════════════════════════════════════

class EmbeddingPipeline:
    """Batch embedding extraction: WAV segments → HDF5 + manifest.

    Reads preprocessed WAV segments from ``processed_dir/<species>/*.wav``,
    runs them through the configured encoder, and writes results to
    ``embeddings_dir/<species>/embeddings.h5`` with a global ``manifest.csv``.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        mn = cfg.get("embedding", {}).get("model_name", "")
        if mn not in ("birdnet", "birdnet_v2.4"):
            raise RuntimeError(
                "Embedding extraction requires embedding.model_name: 'birdnet'."
            )
        self.encoder = build_encoder(cfg)
        self.processed_dir = Path(cfg["data"]["processed_dir"])
        self.embeddings_dir = Path(cfg["data"]["embeddings_dir"])
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.segment_duration = cfg["audio"]["segment_duration_s"]

    def run(self) -> Path:
        """Execute the full extraction pipeline.

        Returns:
            Path to the generated manifest CSV.
        """
        store = EmbeddingStore(self.embeddings_dir, self.encoder.embedding_dim)

        species_dirs = sorted(
            d for d in self.processed_dir.iterdir() if d.is_dir()
        )
        if not species_dirs:
            logger.warning(f"No species dirs in {self.processed_dir}")
            return store.flush()

        total_files = sum(
            len(list(d.glob("*.wav"))) for d in species_dirs
        )
        logger.info(
            f"Extracting {self.encoder.name} embeddings "
            f"({self.encoder.embedding_dim}D) for {total_files} segments "
            f"across {len(species_dirs)} species"
        )

        t0 = time.time()
        processed = 0

        for species_dir in species_dirs:
            species = species_dir.name
            wav_files = sorted(species_dir.glob("*.wav"))
            if not wav_files:
                continue

            for wav_path in tqdm(
                wav_files, desc=f"  {species}", leave=False, unit="seg"
            ):
                try:
                    waveform, sr = self._load_wav(wav_path)
                    embedding = self.encoder.encode(waveform, sr)

                    # Parse segment index from filename convention: *_seg0003.wav
                    seg_idx = self._parse_segment_index(wav_path.stem)

                    record = EmbeddingRecord(
                        source_file=str(wav_path),
                        species=species,
                        segment_index=seg_idx,
                        embedding_dim=self.encoder.embedding_dim,
                        model_name=self.encoder.name,
                        sample_rate=sr,
                        duration_sec=self.segment_duration,
                    )
                    store.add(embedding, record)
                    processed += 1

                except Exception as e:
                    logger.error(f"  FAILED {wav_path.name}: {e}")

        if hasattr(self.encoder, "flush_cross_sample_embedding_debug"):
            self.encoder.flush_cross_sample_embedding_debug()

        elapsed = time.time() - t0
        logger.info(
            f"Embedding extraction complete: {processed}/{total_files} segments "
            f"in {elapsed:.1f}s ({processed / max(elapsed, 0.01):.1f} seg/s)"
        )

        manifest_path = store.flush()
        self._print_embedding_dataset_summary(manifest_path, processed)
        return manifest_path

    def _print_embedding_dataset_summary(self, manifest_path: Path, processed: int) -> None:
        """Print total embedding count and global mean/std over stored vectors."""
        n_manifest = 0
        if manifest_path.is_file():
            with open(manifest_path, newline="", encoding="utf-8") as f:
                n_manifest = sum(1 for _ in csv.DictReader(f))

        print(
            f"\n[Embedding summary] encoder={self.encoder.name!r} | "
            f"segments_written={processed} | manifest_rows={n_manifest}"
        )
        self._log_manifest_species_counts(manifest_path)

        if n_manifest != processed:
            print(
                f"[Embedding summary] WARNING: manifest count ({n_manifest}) != "
                f"processed count ({processed})."
            )

        embs, _, _ = load_all_embeddings(self.embeddings_dir)
        if embs.size == 0:
            print("[Embedding summary] No vectors loaded for mean/std (empty HDF5).")
            return

        g_mean = float(np.mean(embs))
        g_std = float(np.std(embs))
        print(
            f"[Embedding summary] total_embedding_vectors={len(embs)} | "
            f"dim={embs.shape[1]} | global_mean={g_mean:.6f} | global_std={g_std:.6f}"
        )

    @staticmethod
    def _log_manifest_species_counts(manifest_path: Path) -> None:
        """Spot-check manifest for `noise` vs bird species (binary pipeline)."""
        if not manifest_path.is_file():
            return
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        species_col = "species"
        if species_col not in rows[0]:
            return
        counts = Counter(r[species_col] for r in rows)
        n_noise = counts.get("noise", 0)
        n_other = sum(v for k, v in counts.items() if k != "noise")
        print(
            f"[Embedding summary] manifest species: noise={n_noise}, "
            f"bird_and_species={n_other}, total_rows={len(rows)}"
        )
        if n_noise == 0:
            print(
                "[Embedding summary] WARNING: no manifest rows with species='noise'. "
                "Use raw_dir/noise/ (lowercase), then preprocess → embed again."
            )

    # ── Private ────────────────────────────────────────────

    @staticmethod
    def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
        """Load a WAV file as a 1-D float32 numpy array."""
        import soundfile as sf
        waveform, sr = sf.read(str(path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform.astype(np.float32), sr

    @staticmethod
    def _parse_segment_index(stem: str) -> int:
        """Extract segment index from filename like 'recording_seg0003'."""
        parts = stem.rsplit("_seg", 1)
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0


# ════════════════════════════════════════════════════════════
#  Convenience entry-point
# ════════════════════════════════════════════════════════════

def extract_embeddings(cfg: dict) -> Path:
    """Run the full embedding extraction pipeline from config.

    Returns:
        Path to the generated manifest CSV.
    """
    pipeline = EmbeddingPipeline(cfg)
    return pipeline.run()


# ── Device helper ──────────────────────────────────────────

def _resolve_device(cfg: dict) -> torch.device:
    device_str = cfg.get("project", {}).get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
