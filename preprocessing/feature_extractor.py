"""
feature_extractor.py — Mel-spectrogram and MFCC computation.
"""

from pathlib import Path

import torch
import torchaudio


class FeatureExtractor:
    """Compute mel-spectrograms or MFCCs from raw waveforms."""

    def __init__(self, cfg: dict) -> None:
        feat_cfg = cfg["features"]
        self.feature_type = feat_cfg["type"]
        self.n_fft = feat_cfg["n_fft"]
        self.hop_length = feat_cfg["hop_length"]
        self.n_mels = feat_cfg["n_mels"]
        self.f_min = feat_cfg["f_min"]
        self.f_max = feat_cfg["f_max"]
        self.sample_rate = cfg["audio"]["sample_rate"]

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        self._mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f_min": self.f_min,
                "f_max": self.f_max,
            },
        )

    # ── Public API ──────────────────────────────────────────

    def extract(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract features from a waveform segment.

        Args:
            waveform: Tensor of shape (1, T).
            sr: Sample rate (unused, kept for API consistency).

        Returns:
            Feature tensor — shape depends on feature type:
              mel_spectrogram → (1, n_mels, time)
              mfcc            → (1, n_mfcc, time)
        """
        if self.feature_type == "mel_spectrogram":
            spec = self._mel_transform(waveform)
            # Convert to log scale (add small epsilon for numerical stability)
            spec = torch.log(spec + 1e-9)
            return spec
        elif self.feature_type == "mfcc":
            return self._mfcc_transform(waveform)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    @staticmethod
    def save(feature: torch.Tensor, path: Path) -> None:
        """Save a feature tensor to disk."""
        torch.save(feature, str(path))

    @staticmethod
    def load(path: Path) -> torch.Tensor:
        """Load a feature tensor from disk."""
        return torch.load(str(path), weights_only=True)
