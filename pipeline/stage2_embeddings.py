"""
Stage 2: BirdNET Embedding Extraction

Extract 1024-dimensional embeddings from 3-second audio segments using
BirdNET's internal representation. Also compute per-segment BirdNET
confidence scores for downstream pseudo-labeling.

Output:
    features/embeddings/embeddings.npy       (N × 1024)
    features/embeddings/labels.npy           (species labels)
    features/embeddings/paths.npy            (file paths)
    features/embeddings/birdnet_confidences.npy  (max confidence per segment)
"""

import os
import glob
import numpy as np
import librosa
import tempfile
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Lazy-loaded model singletons ───────────────────────────────────────────
_birdnet_model = None


def _load_birdnet():
    """Load BirdNET model (lazy)."""
    global _birdnet_model
    if _birdnet_model is None:
        import birdnet
        _birdnet_model = birdnet.load("acoustic", "2.4", "tf")
    return _birdnet_model


# ─── Embedding Extraction ───────────────────────────────────────────────────

def extract_birdnet_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract embeddings from BirdNET's internal representation.

    Uses BirdNET to predict on a temporary file and extracts the species
    probability vector as a proxy embedding (1024-d).

    Args:
        audio: Audio array (3 seconds at 48 kHz).
        sr: Sample rate.

    Returns:
        1-D numpy embedding vector.
    """
    model = _load_birdnet()

    # BirdNET API requires a file path — write temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, sr)

    try:
        # Get 1024-d embedding directly from the model
        res = model.encode(tmp_path)
        embedding = res._embeddings.squeeze()

        # Fallback if empty
        if not embedding.size:
            embedding = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

        # Pad or truncate to expected dimension
        if len(embedding) < config.EMBEDDING_DIM:
            embedding = np.pad(
                embedding,
                (0, config.EMBEDDING_DIM - len(embedding)),
            )
        elif len(embedding) > config.EMBEDDING_DIM:
            embedding = embedding[: config.EMBEDDING_DIM]

        return embedding

    finally:
        os.unlink(tmp_path)


def extract_embeddings(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract BirdNET embeddings from an audio segment.

    Args:
        audio: Audio array (3 seconds at target SR).
        sr: Sample rate.

    Returns:
        1-D numpy array (1024-d BirdNET embedding).
    """
    return extract_birdnet_embedding(audio, sr)


def get_birdnet_confidence(file_path: str) -> float:
    """
    Get the maximum BirdNET confidence score for a single audio file.

    Used for pseudo-labeling segments as bird vs noise.

    Args:
        file_path: Path to a .wav segment.

    Returns:
        Maximum confidence score (0.0–1.0). Returns 0.0 on failure.
    """
    try:
        model = _load_birdnet()
        res = model.predict(file_path)
        if hasattr(res, "_species_probs") and res._species_probs.size > 0:
            return float(res._species_probs.max())
        return 0.0
    except Exception:
        return 0.0


def batch_extract_from_directory(
    segment_dir: str = None,
    output_dir: str = None,
    max_files: int = None,
) -> dict:
    """
    Extract BirdNET embeddings for all segmented audio files in a directory tree.

    Saves embeddings, labels, paths, and BirdNET confidences as .npy files.

    Args:
        segment_dir: Root of segmented audio directory.
        output_dir: Directory to save embeddings.
        max_files: Optional limit on number of files to process.

    Returns:
        Dict with counts of processed files per species.
    """
    segment_dir = segment_dir or config.SEGMENTED_DIR
    output_dir = output_dir or config.EMBEDDINGS_DIR

    # Find all audio segments
    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        audio_files.extend(
            glob.glob(os.path.join(segment_dir, "**", ext), recursive=True)
        )

    if not audio_files:
        print(f"[Stage 2] No audio files found in {segment_dir}")
        return {}

    if max_files:
        audio_files = audio_files[:max_files]

    print(f"[Stage 2] Extracting BirdNET embeddings for {len(audio_files)} segments...")

    all_embeddings = []
    all_labels = []
    all_paths = []
    all_confidences = []
    species_counts = {}

    try:
        for file_path in tqdm(audio_files, desc="[Stage 2] Embedding"):
            species_name = os.path.basename(os.path.dirname(file_path))

            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=config.TARGET_SR)

                # Extract embedding
                embedding = extract_embeddings(y, sr)
                all_embeddings.append(embedding)

                # Get BirdNET confidence
                confidence = get_birdnet_confidence(file_path)
                all_confidences.append(confidence)

                all_labels.append(species_name)
                all_paths.append(file_path)

                species_counts[species_name] = species_counts.get(species_name, 0) + 1

            except Exception as e:
                print(f"  [WARN] Failed on {file_path}: {e}")
                continue
    except KeyboardInterrupt:
        print("\n[Stage 2] Interrupted by user! Saving progress so far...")

    if not all_embeddings:
        print("[Stage 2] No embeddings extracted. Exiting.")
        return {}
    
    # Convert to arrays and save
    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=object)
    paths_arr = np.array(all_paths, dtype=object)
    confs_arr = np.array(all_confidences, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_arr)
    np.save(os.path.join(output_dir, "labels.npy"), labels_arr)
    np.save(os.path.join(output_dir, "paths.npy"), paths_arr)
    np.save(os.path.join(output_dir, "birdnet_confidences.npy"), confs_arr)

    print(f"[Stage 2] Saved {len(embeddings_arr)} embeddings ({embeddings_arr.shape})")
    print(f"[Stage 2] Species: {len(species_counts)}")
    print(f"[Stage 2] BirdNET confidence — mean: {confs_arr.mean():.3f}, "
          f"std: {confs_arr.std():.3f}")

    return species_counts


if __name__ == "__main__":
    batch_extract_from_directory()
