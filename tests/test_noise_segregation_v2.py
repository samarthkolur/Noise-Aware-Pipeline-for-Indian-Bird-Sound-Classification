"""Unit tests for Noise Segregation V2."""

import unittest

import numpy as np

from preprocessing.noise_segregation_v2 import NoiseSegregationV2


class TestNoiseSegregationV2(unittest.TestCase):
    def setUp(self) -> None:
        self.sr = 48_000
        self.dur = 3.0
        self.n = int(self.sr * self.dur)
        self.v2 = NoiseSegregationV2(sample_rate=self.sr, segment_duration_s=self.dur)

    def test_segment_length_padding(self) -> None:
        short = np.random.randn(int(self.sr * 0.5)).astype(np.float32) * 0.01
        r = self.v2.classify_segment(short)
        self.assertEqual(r.total_subframes, 6)
        self.assertIn(r.label, ("bird", "noise"))

    def test_pure_tone_tends_bird(self) -> None:
        t = np.arange(self.n) / self.sr
        # ~2 kHz tone — tonal, not flat noise
        x = 0.3 * np.sin(2 * np.pi * 2000.0 * t).astype(np.float32)
        r = self.v2.classify_segment(x)
        self.assertEqual(r.label, "bird")

    def test_white_noise_tends_noise(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self.n).astype(np.float32) * 0.5
        r = self.v2.classify_segment(x)
        self.assertEqual(r.label, "noise")

    def test_subframe_count(self) -> None:
        x = np.zeros(self.n, dtype=np.float32)
        r = self.v2.classify_segment(x)
        self.assertEqual(r.total_subframes, 6)
        self.assertEqual(len(r.subframes), 6)

    def test_norm_ratio(self) -> None:
        from preprocessing.noise_segregation_v2 import _norm_ratio

        self.assertAlmostEqual(_norm_ratio(0.15, 0.30), 0.5)
        self.assertEqual(_norm_ratio(0.40, 0.30), 1.0)


if __name__ == "__main__":
    unittest.main()
