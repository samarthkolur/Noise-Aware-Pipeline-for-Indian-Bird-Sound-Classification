"""
Stage 1: Audio Segmentation & Standardization

Resample audio to 48 kHz, segment into fixed-length windows with optional
overlap, and save segments for downstream processing.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def segment_audio_array(y: np.ndarray, sr: int) -> list:
    """
    Segment a loaded audio array into fixed-length windows.

    Args:
        y: Audio time series.
        sr: Sample rate (must equal config.TARGET_SR).

    Returns:
        List of numpy arrays, each of length segment_samples.
    """
    segment_samples = int(config.SEGMENT_LENGTH * sr)

    if config.USE_OVERLAP:
        hop_samples = int(config.HOP_LENGTH * sr)
    else:
        hop_samples = segment_samples

    segments = []
    start = 0
    while start + segment_samples <= len(y):
        segments.append(y[start : start + segment_samples])
        start += hop_samples

    return segments


def segment_file(file_path: str) -> list:
    """
    Load, resample, and segment a single audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        List of numpy arrays (3-second segments at 48 kHz).
    """
    y, sr = librosa.load(file_path, sr=config.TARGET_SR)
    assert sr == config.TARGET_SR, f"Resampling failed: got {sr} Hz"
    return segment_audio_array(y, sr)


def segment_file_to_disk(file_path: str, output_dir: str, label: str = None) -> list:
    """
    Segment a file and write each segment to disk.

    Args:
        file_path: Input audio file path.
        output_dir: Directory to write segments into.
        label: Optional subdirectory label (e.g., species name).

    Returns:
        List of output file paths.
    """
    segments = segment_file(file_path)

    if label:
        output_dir = os.path.join(output_dir, label)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_paths = []

    for i, seg in enumerate(segments):
        out_file = os.path.join(output_dir, f"{base_name}_seg{i:04d}.wav")
        sf.write(out_file, seg, config.TARGET_SR)
        output_paths.append(out_file)

    return output_paths


def segment_directory(input_dir: str = None, output_dir: str = None) -> dict:
    """
    Batch-segment all audio files under input_dir, preserving subfolder
    structure (species names).

    Args:
        input_dir: Root directory containing species subfolders with .wav/.flac/.mp3 files.
        output_dir: Root directory to write segmented files.

    Returns:
        Dictionary mapping species names to lists of segment file paths.
    """
    input_dir = input_dir or config.RAW_DATA_DIR
    output_dir = output_dir or config.SEGMENTED_DIR

    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        audio_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not audio_files:
        print(f"[Stage 1] No audio files found in {input_dir}")
        return {}

    print(f"[Stage 1] Found {len(audio_files)} audio files in {input_dir}")

    results = {}
    for file_path in tqdm(audio_files, desc="[Stage 1] Segmenting"):
        species_name = os.path.basename(os.path.dirname(file_path))
        out_paths = segment_file_to_disk(file_path, output_dir, label=species_name)

        if species_name not in results:
            results[species_name] = []
        results[species_name].extend(out_paths)

    total_segments = sum(len(v) for v in results.values())
    print(f"[Stage 1] Created {total_segments} segments across {len(results)} species")
    return results


if __name__ == "__main__":
    segment_directory()
