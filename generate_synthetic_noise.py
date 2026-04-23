#!/usr/bin/env python3
"""
generate_synthetic_noise.py

Generate diverse synthetic noise WAVs into the configured raw noise folder.
Includes realistic environmental noise types that challenge bird classifiers:

  - white:       Flat-spectrum Gaussian noise (trivial to reject)
  - pink:        1/f noise (closer to natural environmental sounds)
  - brown:       1/f² noise (resembles wind/rain rumble)
  - insects:     Narrow-band tonal chirps in 3-8 kHz (mimics crickets/cicadas)
  - rain:        Filtered noise with exponential amplitude modulation
  - wind:        Low-frequency turbulence with gusts
  - band_limited: White noise filtered to bird-frequency band (1-8 kHz)
  - mixed:       Random mix of the above per file

The non-white types produce spectra that overlap with bird call frequencies,
which is critical for generating non-trivial FPR in baseline comparisons.

Usage:
    python generate_synthetic_noise.py --config config.yaml --n_files 50
    python generate_synthetic_noise.py --config config.yaml --n_files 50 --kind mixed

Writes into: <cfg.data.raw_dir>/noise/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from utils.config import load_config


def _rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    return float(20.0 * np.log10(rms))


def _normalize_rms(x: np.ndarray, target_rms_db: float) -> np.ndarray:
    """Normalize to target RMS and clip to [-1, 1]."""
    cur = _rms_db(x)
    gain = float(10 ** ((target_rms_db - cur) / 20.0))
    x = (x * gain).astype(np.float32)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _bandlimit_fft(x: np.ndarray, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Simple FFT bandlimit."""
    n = x.shape[0]
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    X[~mask] = 0
    y = np.fft.irfft(X, n=n).astype(np.float32)
    return y


def _make_white(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Flat-spectrum Gaussian noise."""
    return rng.standard_normal(n).astype(np.float32)


def _make_pink(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """1/f noise via spectral shaping — resembles natural environmental sound."""
    white = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    freqs[0] = 1.0  # avoid division by zero
    X /= np.sqrt(freqs)  # 1/√f amplitude → 1/f power
    return np.fft.irfft(X, n=n).astype(np.float32)


def _make_brown(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """1/f² (Brownian) noise — mimics wind rumble, distant rain."""
    white = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    freqs[0] = 1.0
    X /= freqs  # 1/f amplitude → 1/f² power
    return np.fft.irfft(X, n=n).astype(np.float32)


def _make_insects(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Simulated insect chorus (crickets/cicadas) — tonal in 3-8 kHz range.

    This is the most important noise type: insect chirps produce harmonic
    content in the bird frequency band that can fool classifiers.
    """
    t = np.arange(n) / float(sr)
    signal = np.zeros(n, dtype=np.float32)

    # 3-6 overlapping chirp frequencies with slight randomness
    n_chirps = rng.integers(3, 7)
    for _ in range(n_chirps):
        freq = rng.uniform(3000, 8000)  # 3-8 kHz range
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.05, 0.3)

        # Amplitude modulation (chirping pattern)
        mod_freq = rng.uniform(5, 30)  # 5-30 Hz modulation = insect chirp rate
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t + rng.uniform(0, 2 * np.pi)))

        signal += amp * mod * np.sin(2 * np.pi * freq * t + phase)

    # Add background broadband noise
    signal += 0.02 * rng.standard_normal(n).astype(np.float32)

    return signal.astype(np.float32)


def _make_rain(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Simulated rain — filtered noise with random amplitude drops."""
    noise = rng.standard_normal(n).astype(np.float32)

    # Rain has energy across broad spectrum but with emphasis 1-6 kHz
    noise = _bandlimit_fft(noise, sr, fmin=200, fmax=8000)

    # Random amplitude modulation (raindrop clusters)
    t = np.arange(n) / float(sr)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 2.0) * t)
    # Add random bursts
    for _ in range(rng.integers(5, 20)):
        center = rng.integers(0, n)
        width = rng.integers(sr // 10, sr)
        burst = np.exp(-0.5 * ((np.arange(n) - center) / width) ** 2)
        mod += rng.uniform(0.3, 1.0) * burst

    return (noise * mod).astype(np.float32)


def _make_wind(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """Simulated wind — low-frequency turbulence with gusts."""
    noise = rng.standard_normal(n).astype(np.float32)

    # Wind is mostly low-frequency (<2 kHz) with some broadband content
    noise = _bandlimit_fft(noise, sr, fmin=20, fmax=3000)

    # Gust modulation (slow amplitude changes)
    t = np.arange(n) / float(sr)
    gust = np.zeros(n, dtype=np.float32)
    for _ in range(rng.integers(2, 6)):
        freq = rng.uniform(0.1, 1.0)  # Very slow modulation
        gust += rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    mod = 0.3 + 0.7 * (gust - gust.min()) / (gust.max() - gust.min() + 1e-8)

    return (noise * mod).astype(np.float32)


def _make_band_limited(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """White noise filtered to bird frequency band (1-8 kHz)."""
    white = rng.standard_normal(n).astype(np.float32)
    return _bandlimit_fft(white, sr, fmin=1000, fmax=8000)


# Generator registry
NOISE_GENERATORS = {
    "white": _make_white,
    "pink": _make_pink,
    "brown": _make_brown,
    "insects": _make_insects,
    "rain": _make_rain,
    "wind": _make_wind,
    "band_limited": _make_band_limited,
}

# Realistic environmental noise mix (weighted towards harder types)
MIXED_WEIGHTS = {
    "insects": 0.30,      # Most challenging for bird classifiers
    "rain": 0.15,
    "wind": 0.15,
    "pink": 0.15,
    "brown": 0.10,
    "band_limited": 0.10,
    "white": 0.05,        # Keep some white noise for completeness
}


def make_noise(
    *,
    kind: str,
    sr: int,
    seconds: float,
    target_rms_db: float,
    seed: int,
) -> np.ndarray:
    """Generate a single noise waveform."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)

    if kind == "mixed":
        # Randomly select a type based on weights
        types = list(MIXED_WEIGHTS.keys())
        probs = np.array([MIXED_WEIGHTS[t] for t in types])
        probs /= probs.sum()
        chosen = rng.choice(types, p=probs)
        gen_fn = NOISE_GENERATORS[chosen]
    elif kind in NOISE_GENERATORS:
        gen_fn = NOISE_GENERATORS[kind]
    else:
        raise ValueError(f"Unknown noise kind: {kind}. Choose from: {list(NOISE_GENERATORS.keys()) + ['mixed']}")

    x = gen_fn(rng, n, sr)
    return _normalize_rms(x, target_rms_db)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate diverse synthetic noise WAVs.")
    ap.add_argument("--config", default="config.yaml", help="Pipeline config.yaml path")
    ap.add_argument("--n_files", type=int, default=50, help="Number of files to generate")
    ap.add_argument("--seconds", type=float, default=15.0, help="Duration per generated file")
    ap.add_argument(
        "--kind",
        choices=list(NOISE_GENERATORS.keys()) + ["mixed"],
        default="mixed",
        help="Noise type (default: mixed = realistic environmental blend)",
    )
    ap.add_argument("--target_rms_db", type=float, default=-18.0, help="Target RMS (dBFS-ish)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument("--clean", action="store_true", help="Remove existing synth_* files first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    sr = int(cfg.get("audio", {}).get("sample_rate", 48000))

    noise_dir = raw_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)

    # Optionally remove old synthetic noise
    if args.clean:
        removed = 0
        for old in noise_dir.glob("synth_*.wav"):
            old.unlink()
            removed += 1
        print(f"Removed {removed} old synth_*.wav files")

    written = 0
    type_counts: dict[str, int] = {}
    for i in range(int(args.n_files)):
        rng_probe = np.random.default_rng(int(args.seed) + i)

        # For mixed mode, determine the actual type for the filename
        if args.kind == "mixed":
            types = list(MIXED_WEIGHTS.keys())
            probs = np.array([MIXED_WEIGHTS[t] for t in types])
            probs /= probs.sum()
            actual_kind = rng_probe.choice(types, p=probs)
        else:
            actual_kind = args.kind

        x = make_noise(
            kind=args.kind,
            sr=sr,
            seconds=float(args.seconds),
            target_rms_db=float(args.target_rms_db),
            seed=int(args.seed) + i,
        )

        out = noise_dir / f"synth_{actual_kind}_{i:04d}.wav"
        sf.write(str(out), x, sr, subtype="PCM_16")
        type_counts[actual_kind] = type_counts.get(actual_kind, 0) + 1
        written += 1

    print(f"Generated {written} noise file(s) in: {noise_dir}")
    print(f"Type distribution: {dict(sorted(type_counts.items()))}")
    print("Next: rerun preprocess → embed → train → evaluate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
