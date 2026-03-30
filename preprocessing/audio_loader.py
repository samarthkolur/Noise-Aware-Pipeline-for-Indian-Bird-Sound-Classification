"""
audio_loader.py — Load, resample, and segment raw audio files.
"""

from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio


class AudioLoader:
    """Handles audio I/O, resampling, and fixed-length segmentation."""

    def __init__(self, cfg: dict) -> None:
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.segment_duration = cfg["audio"]["segment_duration_s"]
        self.overlap = cfg["audio"]["overlap"]
        self.mono = cfg["audio"]["mono"]

    # ── Public API ──────────────────────────────────────────

    def load(self, path: Path) -> Tuple[torch.Tensor, int]:
        """Load an audio file and resample to target sample rate.

        Returns:
            (waveform, sample_rate) where waveform is shape (1, T).
        """
        waveform, sr = torchaudio.load(str(path))

        # Convert to mono if required
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        return waveform, self.sample_rate

    def segment(self, waveform: torch.Tensor, sr: int) -> List[torch.Tensor]:
        """Split waveform into fixed-length, overlapping segments.

        Args:
            waveform: Tensor of shape (1, T).
            sr: Sample rate.

        Returns:
            List of tensors, each of shape (1, segment_samples).
        """
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(segment_samples * (1 - self.overlap))
        total_samples = waveform.shape[-1]

        segments: List[torch.Tensor] = []
        start = 0
        while start + segment_samples <= total_samples:
            segments.append(waveform[:, start : start + segment_samples])
            start += hop_samples

        # Pad last segment if there are leftover samples
        if start < total_samples and total_samples - start > segment_samples // 2:
            last = waveform[:, start:]
            pad_length = segment_samples - last.shape[-1]
            last = torch.nn.functional.pad(last, (0, pad_length))
            segments.append(last)

        return segments
