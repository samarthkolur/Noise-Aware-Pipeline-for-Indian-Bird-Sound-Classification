"""
Stage 2: Deep Embedding Extraction

Extract high-dimensional feature embeddings from 3-second audio segments using
multiple pretrained acoustic models: BirdNET, YAMNet, and OpenL3.
"""

import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Lazy-loaded model singletons ────────────────────────────────────────────
_yamnet_model = None
_yamnet_params = None
_openl3_model = None
_birdnet_model = None


def _load_yamnet():
    """Load YAMNet model from TensorFlow Hub (lazy)."""
    global _yamnet_model, _yamnet_params
    if _yamnet_model is None:
        import tensorflow_hub as hub
        import tensorflow as tf
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _yamnet_model


def _load_openl3():
    """Load OpenL3 model (lazy)."""
    global _openl3_model
    if _openl3_model is None:
        import openl3
        _openl3_model = openl3.models.load_audio_embedding_model(
            input_repr="mel256",
            content_type=config.OPENL3_CONTENT_TYPE,
            embedding_size=config.OPENL3_EMBEDDING_SIZE,
        )
    return _openl3_model


def _load_birdnet():
    """Load BirdNET model (lazy)."""
    global _birdnet_model
    if _birdnet_model is None:
        import birdnet
        _birdnet_model = birdnet.load("acoustic", "2.4", "tf")
    return _birdnet_model


# ─── Embedding Extractors ───────────────────────────────────────────────────

def extract_birdnet_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract embeddings from BirdNET's internal representation.

    Uses BirdNET to predict on a temporary file and extracts the confidence
    vector as a proxy embedding.

    Args:
        audio: Audio array (3 seconds at 48 kHz).
        sr: Sample rate.

    Returns:
        1-D numpy embedding vector.
    """
    import tempfile
    import soundfile as sf

    model = _load_birdnet()

    # Write a temp file since BirdNET takes file paths
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        predictions = model.predict(tmp_path)
        # Use the confidence scores as a feature vector
        if predictions is not None and len(predictions) > 0:
            embedding = predictions.iloc[:, 1].values.astype(np.float32)
            # Pad/truncate to a fixed size for consistency
            target_size = 100
            if len(embedding) < target_size:
                embedding = np.pad(embedding, (0, target_size - len(embedding)))
            else:
                embedding = embedding[:target_size]
        else:
            embedding = np.zeros(100, dtype=np.float32)
    finally:
        os.unlink(tmp_path)

    return embedding


def extract_yamnet_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract embeddings using YAMNet (Google, via TensorFlow Hub).

    YAMNet expects 16 kHz mono audio as input and produces 1024-d embeddings.

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        1-D numpy embedding vector (mean-pooled over frames).
    """
    import tensorflow as tf

    model = _load_yamnet()

    # YAMNet expects 16 kHz input
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio

    audio_16k = audio_16k.astype(np.float32)

    # YAMNet returns: scores, embeddings, spectrogram
    scores, embeddings, spectrogram = model(audio_16k)

    # Mean-pool across time frames to get a single vector
    embedding = np.mean(embeddings.numpy(), axis=0)
    return embedding.astype(np.float32)


def extract_openl3_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract embeddings using OpenL3.

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        1-D numpy embedding vector (mean-pooled over frames).
    """
    import openl3

    model = _load_openl3()

    emb, ts = openl3.get_audio_embedding(
        audio,
        sr,
        model=model,
        hop_size=0.5,
        verbose=False,
    )

    # Mean-pool across time frames
    embedding = np.mean(emb, axis=0)
    return embedding.astype(np.float32)


def extract_embeddings(
    audio: np.ndarray,
    sr: int,
    model_names: list = None,
) -> np.ndarray:
    """
    Extract and optionally fuse embeddings from multiple models.

    Args:
        audio: Audio array (3 seconds at target SR).
        sr: Sample rate.
        model_names: List of model names to use. Defaults to config.EMBEDDING_MODELS.

    Returns:
        1-D numpy array — either single-model embedding or concatenated fusion.
    """
    model_names = model_names or config.EMBEDDING_MODELS

    extractors = {
        "birdnet": extract_birdnet_embedding,
        "yamnet": extract_yamnet_embedding,
        "openl3": extract_openl3_embedding,
    }

    embeddings = []
    for name in model_names:
        if name not in extractors:
            raise ValueError(f"Unknown embedding model: {name}. Choose from {list(extractors.keys())}")
        emb = extractors[name](audio, sr)
        embeddings.append(emb)

    if config.FUSE_EMBEDDINGS or len(embeddings) > 1:
        return np.concatenate(embeddings)
    else:
        return embeddings[0]


def batch_extract_from_directory(
    segment_dir: str = None,
    output_dir: str = None,
    model_names: list = None,
    max_files: int = None,
) -> dict:
    """
    Extract embeddings for all segmented audio files in a directory tree.

    Saves embeddings as .npy files and returns paths.

    Args:
        segment_dir: Root of segmented audio directory.
        output_dir: Directory to save embedding .npy files.
        model_names: Which embedding models to use.
        max_files: Limit the number of files (for testing).

    Returns:
        Dict mapping relative paths to embedding file paths.
    """
    segment_dir = segment_dir or config.SEGMENTED_DIR
    output_dir = output_dir or config.EMBEDDINGS_DIR
    model_names = model_names or config.EMBEDDING_MODELS
    os.makedirs(output_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(segment_dir, "**", "*.wav"), recursive=True)
    if max_files:
        audio_files = audio_files[:max_files]

    if not audio_files:
        print(f"[Stage 2] No segment files found in {segment_dir}")
        return {}

    print(f"[Stage 2] Extracting embeddings for {len(audio_files)} segments using {model_names}")

    results = {}
    all_embeddings = []
    all_labels = []
    all_paths = []

    for file_path in tqdm(audio_files, desc="[Stage 2] Embedding extraction"):
        try:
            y, sr = librosa.load(file_path, sr=config.TARGET_SR)
            emb = extract_embeddings(y, sr, model_names)
            all_embeddings.append(emb)

            # Label = species subdir name
            label = os.path.basename(os.path.dirname(file_path))
            all_labels.append(label)
            all_paths.append(file_path)
        except Exception as e:
            print(f"  [WARN] Failed to process {file_path}: {e}")
            continue

    if all_embeddings:
        embeddings_array = np.stack(all_embeddings)
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_array)
        np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels))
        np.save(os.path.join(output_dir, "paths.npy"), np.array(all_paths))
        print(f"[Stage 2] Saved {embeddings_array.shape} embeddings to {output_dir}")
        results = {
            "embeddings_path": os.path.join(output_dir, "embeddings.npy"),
            "labels_path": os.path.join(output_dir, "labels.npy"),
            "paths_path": os.path.join(output_dir, "paths.npy"),
            "shape": embeddings_array.shape,
        }

    return results


def get_birdnet_confidence(file_path: str) -> float:
    """
    Get the maximum BirdNET confidence score for a single audio file.

    Used for pseudo-labeling segments as bird vs noise.

    Args:
        file_path: Path to a .wav segment.

    Returns:
        Maximum confidence score (0.0–1.0). Returns 0.0 on failure.
    """
    model = _load_birdnet()
    try:
        predictions = model.predict(file_path)
        if predictions is not None and len(predictions) > 0:
            return float(predictions.iloc[:, 1].max())
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    batch_extract_from_directory()
