#!/usr/bin/env python3
"""
generate_synthetic_noise.py

Generate synthetic noise WAVs into the configured raw noise folder.
This is meant to improve robustness against "white noise" false positives by
ensuring the noise class includes such examples during training.

Writes into:
  <cfg.data.raw_dir>/noise/

Usage:
  python3 generate_synthetic_noise.py --config config.yaml --n_files 50

Notes:
  - Does NOT modify any model code; it only adds training data.
  - Generated files are short single-channel WAVs at cfg.audio.sample_rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.config import load_config


def _rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    return float(20.0 * np.log10(rms))


def _bandlimit_fft(x: np.ndarray, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Simple FFT bandlimit (no external deps)."""
    n = x.shape[0]
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    X[~mask] = 0
    y = np.fft.irfft(X, n=n).astype(np.float32)
    return y


def make_noise(
    *,
    kind: str,
    sr: int,
    seconds: float,
    target_rms_db: float,
    fmin: float,
    fmax: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    x = rng.standard_normal(n).astype(np.float32)

    if kind == "band_limited":
        x = _bandlimit_fft(x, sr=sr, fmin=fmin, fmax=fmax)
    elif kind == "white":
        pass
    else:
        raise ValueError(f"Unknown noise kind: {kind}")

    # Normalize RMS to target
    cur = _rms_db(x)
    gain_db = float(target_rms_db - cur)
    gain = float(10 ** (gain_db / 20.0))
    x = (x * gain).astype(np.float32)

    # Clip hard bounds (keep within [-1,1])
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic noise WAVs into raw noise/ folder.")
    ap.add_argument("--config", default="config.yaml", help="Pipeline config.yaml path")
    ap.add_argument("--n_files", type=int, default=50, help="Number of files to generate")
    ap.add_argument("--seconds", type=float, default=15.0, help="Duration per generated file")
    ap.add_argument(
        "--kind",
        choices=("white", "band_limited"),
        default="white",
        help="Noise type to generate",
    )
    ap.add_argument("--target_rms_db", type=float, default=-18.0, help="Target RMS (dBFS-ish)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    sr = int(cfg.get("audio", {}).get("sample_rate", 48000))
    fmin = float(cfg.get("features", {}).get("f_min", 150))
    fmax = float(cfg.get("features", {}).get("f_max", 15000))

    noise_dir = raw_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i in range(int(args.n_files)):
        x = make_noise(
            kind=str(args.kind),
            sr=sr,
            seconds=float(args.seconds),
            target_rms_db=float(args.target_rms_db),
            fmin=fmin,
            fmax=fmax,
            seed=int(args.seed) + i,
        )
        out = noise_dir / f"synth_{args.kind}_{i:04d}.wav"
        sf.write(str(out), x, sr, subtype="PCM_16")
        written += 1

    print(f"Generated {written} noise file(s) in: {noise_dir}")
    print("Next: rerun preprocess → embed → train → infer → evaluate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

