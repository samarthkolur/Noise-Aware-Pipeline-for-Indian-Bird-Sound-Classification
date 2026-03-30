"""
preprocessing.py — Core preprocessing engine.

Converts raw audio to 48 kHz mono WAV, segments into fixed-length clips,
optionally removes silent segments via RMS thresholding, and persists each
segment alongside a JSON metadata sidecar.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Defaults ────────────────────────────────────────────────
TARGET_SR = 48_000
SEGMENT_DURATION_S = 3.0
RMS_THRESHOLD_DB = -40.0  # segments quieter than this are considered silent


# ── Metadata dataclass ──────────────────────────────────────
@dataclass
class SegmentMeta:
    """Metadata sidecar for a single audio segment."""

    source_file: str
    species: str
    segment_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    sample_rate: int
    num_samples: int
    rms_db: float
    is_silent: bool
    output_path: str


# ── Core Preprocessor ──────────────────────────────────────
class Preprocessor:
    """End-to-end audio preprocessor.

    Pipeline per file:
        load → resample to 48 kHz → mono → segment (3 s) →
        [optional] drop silent segments → save WAV + JSON metadata
    """

    def __init__(self, cfg: dict) -> None:
        audio_cfg = cfg.get("audio", {})
        silence_cfg = cfg.get("silence_removal", {})

        self.target_sr: int = audio_cfg.get("sample_rate", TARGET_SR)
        self.segment_duration: float = audio_cfg.get(
            "segment_duration_s", SEGMENT_DURATION_S
        )
        self.mono: bool = audio_cfg.get("mono", True)

        # Silence removal
        self.silence_removal_enabled: bool = silence_cfg.get("enabled", True)
        self.rms_threshold_db: float = silence_cfg.get(
            "rms_threshold_db", RMS_THRESHOLD_DB
        )

        # Derived
        self.segment_samples: int = int(self.target_sr * self.segment_duration)

    # ── Public API ──────────────────────────────────────────

    def process_file(
        self,
        audio_path: Path,
        output_dir: Path,
        species: str = "unknown",
    ) -> List[SegmentMeta]:
        """Process a single audio file → list of saved segments.

        Args:
            audio_path: Path to the source audio file.
            output_dir: Root directory for outputs (species sub-dir is created).
            species: Species / class label for this file.

        Returns:
            List of SegmentMeta for every *kept* segment.
        """
        waveform, sr = self._load_and_normalise(audio_path)
        segments = self._segment(waveform)

        species_dir = output_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)

        kept: List[SegmentMeta] = []
        for idx, seg in enumerate(segments):
            start_sec = idx * self.segment_duration
            end_sec = start_sec + self.segment_duration
            rms_db = self._rms_db(seg)
            is_silent = rms_db < self.rms_threshold_db

            if self.silence_removal_enabled and is_silent:
                logger.debug(
                    f"  Dropping segment {idx} (RMS={rms_db:.1f} dB < "
                    f"{self.rms_threshold_db} dB)"
                )
                continue

            # Save WAV
            stem = audio_path.stem
            seg_filename = f"{stem}_seg{idx:04d}.wav"
            seg_path = species_dir / seg_filename
            self._save_wav(seg, seg_path)

            # Build metadata
            meta = SegmentMeta(
                source_file=str(audio_path),
                species=species,
                segment_index=idx,
                start_sec=round(start_sec, 4),
                end_sec=round(end_sec, 4),
                duration_sec=round(self.segment_duration, 4),
                sample_rate=self.target_sr,
                num_samples=seg.shape[-1],
                rms_db=round(rms_db, 2),
                is_silent=is_silent,
                output_path=str(seg_path),
            )

            # Save JSON sidecar
            meta_path = seg_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(asdict(meta), f, indent=2)

            kept.append(meta)

        logger.info(
            f"  {audio_path.name}: {len(segments)} segments → "
            f"{len(kept)} kept ({len(segments) - len(kept)} silent dropped)"
        )
        return kept

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg"),
    ) -> List[SegmentMeta]:
        """Process every audio file under *input_dir*.

        Expects the layout ``input_dir/<species>/<audio_files>``.
        Mirrors the species sub-directory structure into *output_dir*.

        Returns:
            Flat list of all kept SegmentMeta across every file.
        """
        all_meta: List[SegmentMeta] = []
        species_dirs = sorted(
            [d for d in input_dir.iterdir() if d.is_dir()]
        )

        if not species_dirs:
            logger.warning(f"No species sub-directories found in {input_dir}")
            return all_meta

        logger.info(
            f"Processing {len(species_dirs)} species from {input_dir} → {output_dir}"
        )

        for species_dir in species_dirs:
            species = species_dir.name
            audio_files = sorted(
                f
                for f in species_dir.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            )
            if not audio_files:
                continue

            logger.info(f"[{species}] {len(audio_files)} files")
            for audio_path in audio_files:
                try:
                    metas = self.process_file(audio_path, output_dir, species)
                    all_meta.extend(metas)
                except Exception as e:
                    logger.error(f"  FAILED {audio_path.name}: {e}")

        logger.info(
            f"Preprocessing complete: {len(all_meta)} segments saved to {output_dir}"
        )
        return all_meta

    # ── Private helpers ─────────────────────────────────────

    def _load_and_normalise(self, path: Path) -> Tuple[torch.Tensor, int]:
        """Load audio, convert to mono, resample to target SR."""
        waveform_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # (T, Channels) -> (Channels, T)
        waveform = torch.from_numpy(waveform_np).transpose(0, 1)

        # Mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        return waveform, self.target_sr

    def _segment(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split waveform into non-overlapping fixed-length segments.

        The last segment is zero-padded if shorter than ``segment_samples``
        but longer than half that length; otherwise it is discarded.
        """
        total = waveform.shape[-1]
        segs: List[torch.Tensor] = []
        start = 0

        while start + self.segment_samples <= total:
            segs.append(waveform[:, start : start + self.segment_samples])
            start += self.segment_samples

        # Keep leftover only if > half a segment
        leftover = total - start
        if leftover > self.segment_samples // 2:
            last = waveform[:, start:]
            pad = self.segment_samples - leftover
            last = torch.nn.functional.pad(last, (0, pad))
            segs.append(last)

        return segs

    @staticmethod
    def _rms_db(waveform: torch.Tensor) -> float:
        """Compute RMS level in decibels for a waveform tensor."""
        rms = waveform.float().pow(2).mean().sqrt().item()
        if rms < 1e-10:
            return -100.0
        return 20.0 * math.log10(rms)

    def _save_wav(self, waveform: torch.Tensor, path: Path) -> None:
        """Save a waveform tensor as a 48 kHz mono WAV file."""
        # Convert (Channels, T) to (T, Channels) for soundfile
        waveform_np = waveform.transpose(0, 1).numpy()
        sf.write(str(path), waveform_np, samplerate=self.target_sr, subtype="PCM_16")


# ── Convenience function ────────────────────────────────────
def preprocess_dataset(cfg: dict) -> List[SegmentMeta]:
    """Run the full preprocessing stage using paths from the config.

    Reads ``cfg['data']['raw_dir']`` and writes to ``cfg['data']['processed_dir']``.
    """
    preprocessor = Preprocessor(cfg)
    input_dir = Path(cfg["data"]["raw_dir"])
    output_dir = Path(cfg["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return preprocessor.process_directory(input_dir, output_dir)
