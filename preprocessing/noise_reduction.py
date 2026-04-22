"""
noise_reduction.py — Spectral gating and band-pass noise suppression.
"""

import numpy as np
import torch


class NoiseReducer:
    """Apply noise-reduction techniques to audio segments."""

    def __init__(self, cfg: dict) -> None:
        nr_cfg = cfg["noise_reduction"]
        self.method = nr_cfg["method"]
        self.sample_rate = cfg["audio"]["sample_rate"]

        # Spectral gating parameters
        sg = nr_cfg.get("spectral_gating", {})
        self.n_std_thresh = sg.get("n_std_thresh", 1.5)
        self.prop_decrease = sg.get("prop_decrease", 1.0)

        # Band-pass parameters
        bp = nr_cfg.get("bandpass", {})
        self.low_freq = bp.get("low_freq", 150)
        self.high_freq = bp.get("high_freq", 15000)

    # ── Public API ──────────────────────────────────────────

    def reduce(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply noise reduction to a waveform segment.

        Args:
            waveform: Tensor of shape (1, T).
            sr: Sample rate.

        Returns:
            Denoised waveform tensor of same shape.
        """
        if self.method == "spectral_gating":
            return self._spectral_gating(waveform, sr)
        elif self.method == "bandpass":
            return self._bandpass_filter(waveform, sr)
        else:
            raise ValueError(f"Unknown noise reduction method: {self.method}")

    # ── Private ─────────────────────────────────────────────

    def _spectral_gating(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Spectral gating via noisereduce library."""
        try:
            import noisereduce as nr
        except ImportError:
            raise ImportError(
                "noisereduce is required for spectral gating. "
                "Install with: pip install noisereduce"
            )

        audio_np = waveform.squeeze(0).numpy()
        reduced = nr.reduce_noise(
            y=audio_np,
            sr=sr,
            n_std_thresh_stationary=self.n_std_thresh,
            prop_decrease=self.prop_decrease,
        )
        return torch.from_numpy(reduced).unsqueeze(0).float()

    def _bandpass_filter(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Simple FFT-based band-pass filter."""
        audio_np = waveform.squeeze(0).numpy()
        n = len(audio_np)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        fft = np.fft.rfft(audio_np)

        # Zero out frequencies outside the band
        mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
        fft[~mask] = 0

        filtered = np.fft.irfft(fft, n=n)
        return torch.from_numpy(filtered).unsqueeze(0).float()
