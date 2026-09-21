"""pipeline/inference.py tests. Segment-level processing invokes real BirdNET
embedding extraction whenever Noise Segregation V2 + Bird Guard don't already
route a segment to noise, so these are marked slow."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.inference import InferencePipeline, SegmentRecord, write_manifest_results
from pipeline.model import BirdAutoencoder, FocalMLP

pytestmark = pytest.mark.slow

SR = 48000


@pytest.fixture(scope="module")
def pipeline():
    cfg = load_config()
    mlp = FocalMLP(cfg.embedding.embedding_dim, cfg.mlp)
    ae = BirdAutoencoder(cfg.embedding.embedding_dim, cfg.autoencoder)
    return InferencePipeline(cfg, mlp, ae, tau_ae=1.0)  # generous tau_ae: nothing OOD-rejected


def test_silent_segment_is_rejected_without_calling_birdnet(pipeline):
    silence = np.zeros(SR * 3, dtype=np.float32)
    record = pipeline.process_segment(silence, SR, source_path="silence.wav", segment_index=0)
    assert record.silence_rejected is True
    assert record.final_band is None  # short-circuited before routing


def test_white_noise_segment_routes_to_noise_without_calling_birdnet(pipeline):
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, SR * 3).astype(np.float32)
    white = 0.8 * white / np.max(np.abs(white))
    record = pipeline.process_segment(white, SR, source_path="white.wav", segment_index=0)
    assert record.noise_class == "noise"
    assert record.final_band == "noise"
    assert record.mlp_prob is None  # never reached the embedding/MLP stage


def test_sine_segment_reaches_mlp_and_gets_routed(pipeline):
    t = np.linspace(0, 3, SR * 3, endpoint=False)
    sine = (0.8 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    record = pipeline.process_segment(sine, SR, source_path="sine.wav", segment_index=0)
    assert record.noise_class == "bird"
    assert record.mlp_prob is not None
    assert 0.0 <= record.mlp_prob <= 1.0
    assert record.final_band in ("bird", "uncertain", "noise")


def test_segment_start_end_sec_reflect_segment_index(pipeline):
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, SR * 3).astype(np.float32)
    white = 0.8 * white / np.max(np.abs(white))
    record = pipeline.process_segment(white, SR, source_path="white.wav", segment_index=2)
    assert record.start_sec == pytest.approx(6.0)
    assert record.end_sec == pytest.approx(9.0)


def test_process_file_with_segments_returns_matching_lengths(pipeline, tmp_path):
    from pipeline.audio import write_wav

    rng = np.random.default_rng(0)
    audio = rng.normal(0, 1, SR * 9).astype(np.float32)  # 9s -> 3 non-overlapping segments
    audio = 0.8 * audio / np.max(np.abs(audio))
    wav_path = tmp_path / "clip.wav"
    write_wav(str(wav_path), audio, SR)

    records, segments, sr = pipeline.process_file_with_segments(str(wav_path))

    assert sr == SR
    assert len(records) == len(segments) == 3
    for i, (record, seg) in enumerate(zip(records, segments, strict=True)):
        assert record.segment_index == i
        assert record.start_sec == pytest.approx(i * 3.0)
        assert len(seg) == SR * 3


def test_write_manifest_results_creates_csv(tmp_path):
    records = [
        SegmentRecord(path="a.wav", segment_index=0, final_band="bird", routed_to="outputs/bird/"),
        SegmentRecord(path="b.wav", segment_index=0, silence_rejected=True),
    ]
    out_path = tmp_path / "manifest_results.csv"
    write_manifest_results(records, out_path)
    content = out_path.read_text()
    assert "a.wav" in content
    assert "b.wav" in content
