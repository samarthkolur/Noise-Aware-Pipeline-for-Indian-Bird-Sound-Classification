"""
preprocessing.py — Core preprocessing engine.

Converts raw audio to 48 kHz mono WAV, segments into fixed-length clips,
optionally removes silent segments via RMS thresholding, and persists each
segment alongside a JSON metadata sidecar.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from preprocessing.noise_segregation_v2 import NoiseSegregationV2
from utils.logger import get_logger

logger = get_logger(__name__)

NOISE_FOLDER_NAME = "noise"  # must match dataset.NOISE_LABEL_NAME for binary labels

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
    # Pipeline / V2 (optional; baseline mode leaves these None)
    pipeline_mode: str = "baseline"
    v2_label: Optional[str] = None
    v2_mean_score: Optional[float] = None
    v2_noise_subframe_votes: Optional[int] = None
    v2_total_subframes: Optional[int] = None
    source_species: Optional[str] = None


# ── Core Preprocessor ──────────────────────────────────────
class Preprocessor:
    """End-to-end audio preprocessor.

    Pipeline per file (``pipeline.mode: full`` or ``filtered``):
        load → resample to 48 kHz → mono → **segment (3 s)** →
        **[optional] RMS silence drop** → **Noise Segregation V2** (score + vote) →
        **route** to ``processed/<species>/`` (bird) or ``processed/noise/`` (noise in full;
        dropped in filtered) → save WAV + JSON.

    Baseline mode skips V2 and only mirrors species folders (optional manual ``noise/``).
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

        # Cumulative stats for process_directory (silence drops vs raw segments)
        self._stats_raw_segments: int = 0
        self._stats_dropped_silent: int = 0
        self._stats_dropped_v2_noise: int = 0
        self._stats_v2_bird: int = 0
        self._stats_v2_noise: int = 0
        self._v2_mean_S_sum: float = 0.0
        self._v2_mean_S_n: int = 0

        self.cfg = cfg
        self._pipeline_mode: str = cfg.get("pipeline", {}).get("mode", "baseline")
        self._v2: Optional[NoiseSegregationV2] = None
        if self._pipeline_mode in ("filtered", "full"):
            self._v2 = NoiseSegregationV2.from_config(cfg)

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

            stem = audio_path.stem
            source_species = species

            # Manual-only noise folder: no V2 routing (always class `noise`)
            if species == NOISE_FOLDER_NAME:
                out_species = NOISE_FOLDER_NAME
                v2_label = None
                v2_mean = None
                v2_nvotes = None
                v2_nsub = None
                seg_filename = f"{stem}_seg{idx:04d}.wav"
            elif self._pipeline_mode == "baseline" or self._v2 is None:
                out_species = species
                v2_label = None
                v2_mean = None
                v2_nvotes = None
                v2_nsub = None
                seg_filename = f"{stem}_seg{idx:04d}.wav"
            else:
                seg_np = seg[0].detach().cpu().numpy().astype(np.float32)
                v2_res = self._v2.classify_segment(seg_np)
                self._v2_mean_S_sum += float(v2_res.mean_score)
                self._v2_mean_S_n += 1
                if v2_res.label == "bird":
                    self._stats_v2_bird += 1
                else:
                    self._stats_v2_noise += 1
                v2_label = v2_res.label
                v2_mean = round(v2_res.mean_score, 5)
                v2_nvotes = v2_res.noise_votes
                v2_nsub = v2_res.total_subframes

                if self._pipeline_mode == "filtered":
                    if v2_res.label == "noise":
                        self._stats_dropped_v2_noise += 1
                        continue
                    out_species = species
                    seg_filename = f"{stem}_seg{idx:04d}.wav"
                else:  # full
                    if v2_res.label == "noise":
                        out_species = NOISE_FOLDER_NAME
                        seg_filename = f"{source_species}__{stem}_seg{idx:04d}.wav"
                    else:
                        out_species = species
                        seg_filename = f"{stem}_seg{idx:04d}.wav"

            species_dir = output_dir / out_species
            species_dir.mkdir(parents=True, exist_ok=True)
            seg_path = species_dir / seg_filename
            self._save_wav(seg, seg_path)

            meta = SegmentMeta(
                source_file=str(audio_path),
                species=out_species,
                segment_index=idx,
                start_sec=round(start_sec, 4),
                end_sec=round(end_sec, 4),
                duration_sec=round(self.segment_duration, 4),
                sample_rate=self.target_sr,
                num_samples=seg.shape[-1],
                rms_db=round(rms_db, 2),
                is_silent=is_silent,
                output_path=str(seg_path),
                pipeline_mode=self._pipeline_mode,
                v2_label=v2_label,
                v2_mean_score=v2_mean,
                v2_noise_subframe_votes=v2_nvotes,
                v2_total_subframes=v2_nsub,
                source_species=source_species,
            )

            meta_path = seg_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(asdict(meta), f, indent=2)

            kept.append(meta)

        n_raw = len(segments)
        n_drop = n_raw - len(kept)
        self._stats_raw_segments += n_raw
        self._stats_dropped_silent += n_drop
        logger.info(
            f"  {audio_path.name}: {n_raw} segments -> "
            f"{len(kept)} kept ({n_drop} silent dropped)"
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
        self._stats_raw_segments = 0
        self._stats_dropped_silent = 0
        self._stats_dropped_v2_noise = 0
        self._stats_v2_bird = 0
        self._stats_v2_noise = 0
        self._v2_mean_S_sum = 0.0
        self._v2_mean_S_n = 0

        species_dirs = sorted(
            [d for d in input_dir.iterdir() if d.is_dir()]
        )

        if not species_dirs:
            logger.warning(f"No species sub-directories found in {input_dir}")
            return all_meta

        logger.info(
            f"Processing {len(species_dirs)} species from {input_dir} -> {output_dir}"
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
        logger.info(
            f"[Preprocess summary] Raw 3s segments (before silence filter): "
            f"{self._stats_raw_segments}; dropped silent: {self._stats_dropped_silent}; "
            f"kept (written WAVs): {len(all_meta)}"
        )
        if self._v2 is not None:
            logger.info(
                f"[V2 Routing] bird={self._stats_v2_bird}, noise={self._stats_v2_noise}"
            )
            if self._v2_mean_S_n > 0:
                mean_s = self._v2_mean_S_sum / self._v2_mean_S_n
                logger.info(
                    f"[V2 Scores] mean_S_over_segments={mean_s:.4f} (n={self._v2_mean_S_n})"
                )
            else:
                logger.warning("[V2 Scores] no segments scored (check audio / silence gate).")
        if self._pipeline_mode == "full":
            (output_dir / NOISE_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
        if self._pipeline_mode == "filtered" and self._stats_dropped_v2_noise > 0:
            logger.info(
                f"[Preprocess summary] V2 filtered mode: dropped {self._stats_dropped_v2_noise} "
                "noise-like segments (not written)."
            )
        if self.silence_removal_enabled and self._stats_dropped_silent > 0:
            logger.info(
                "[Preprocess summary] If kept count is far below expectations, try "
                "lowering silence_removal.rms_threshold_db (e.g. -50) or "
                "silence_removal.enabled: false in config.yaml."
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


# ── Raw layout verification ─────────────────────────────────

def verify_raw_noise_layout(raw_dir: Path, pipeline_mode: str = "baseline") -> None:
    """Log whether the noise class folder exists and warn on common mistakes.

    For ``pipeline.mode: full``, V2 can populate ``processed_dir/noise/`` without
    a raw ``noise/`` folder, so the missing-folder warning is skipped.
    """
    raw_dir = Path(raw_dir)
    noise_dir = raw_dir / NOISE_FOLDER_NAME
    extensions = {".wav", ".mp3", ".flac", ".ogg"}

    for wrong_name in ("Noise", "NOISE"):
        wrong = raw_dir / wrong_name
        if wrong.is_dir():
            logger.warning(
                f"[Dataset check] Found '{wrong}' but binary labels require the folder "
                f"name exactly '{NOISE_FOLDER_NAME}' (see dataset.NOISE_LABEL_NAME). "
                "Rename to 'noise' or those files will be labeled as bird."
            )

    if noise_dir.is_dir():
        n_audio = sum(
            1
            for f in noise_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        logger.info(
            f"[Dataset check] '{noise_dir}' exists with {n_audio} audio file(s) "
            f"(label 0 in binary mode)."
        )
        if n_audio == 0:
            logger.warning(
                "[Dataset check] noise/ has no audio files — add non-bird clips for the noise class."
            )
    elif pipeline_mode == "full":
        logger.info(
            "[Dataset check] No raw noise/ folder — OK for pipeline.mode=full "
            "(V2 will write noise-like segments to processed_dir/noise/)."
        )
    else:
        logger.warning(
            f"[Dataset check] Missing '{noise_dir}'. For bird vs noise training without "
            f"V2 full routing, add raw_dir/{NOISE_FOLDER_NAME}/ with non-bird audio, "
            "or set pipeline.mode to 'full' to generate noise segments automatically."
        )


def write_segments_manifest(processed_dir: Path, metas: List[SegmentMeta]) -> Optional[Path]:
    """Write ``segments_manifest.csv`` with one row per saved segment (research audit trail)."""
    processed_dir = Path(processed_dir)
    path = processed_dir / "segments_manifest.csv"
    if not metas:
        logger.info("No segments to write to segments_manifest.csv")
        return None
    fieldnames = list(asdict(metas[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for m in metas:
            w.writerow(asdict(m))
    logger.info(f"Wrote segments manifest ({len(metas)} rows): {path}")
    return path


# ── Convenience function ────────────────────────────────────
def preprocess_dataset(cfg: dict) -> List[SegmentMeta]:
    """Run the full preprocessing stage using paths from the config.

    Reads ``cfg['data']['raw_dir']`` and writes to ``cfg['data']['processed_dir']``.
    """
    input_dir = Path(cfg["data"]["raw_dir"])
    pipeline_mode = str(cfg.get("pipeline", {}).get("mode", "baseline"))
    verify_raw_noise_layout(input_dir, pipeline_mode=pipeline_mode)

    preprocessor = Preprocessor(cfg)
    output_dir = Path(cfg["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metas = preprocessor.process_directory(input_dir, output_dir)
    if cfg.get("pipeline", {}).get("write_segments_manifest", True):
        write_segments_manifest(output_dir, metas)
    return metas
