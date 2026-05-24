import numpy as np
import pytest
import soundfile as sf
import torch
from pathlib import Path
from utils.config import load_config
from inference.predictor import Predictor
from generate_synthetic_noise import _make_white, _make_pink, _make_brown, _make_insects

@pytest.fixture(scope="module")
def predictor():
    cfg = load_config('config.yaml')
    return Predictor(cfg)

@pytest.fixture(scope="module")
def test_audio_dir(tmp_path_factory):
    return tmp_path_factory.mgettemp("test_audio")

def _generate_and_save(generator, filename, test_audio_dir, sr=48000, duration=3.0):
    rng = np.random.default_rng(42)
    audio = generator(rng, int(sr*duration), sr)
    path = test_audio_dir / filename
    sf.write(str(path), audio, sr)
    return path

def test_pure_white_noise(predictor, test_audio_dir):
    path = _generate_and_save(_make_white, 'test_white.wav', test_audio_dir)
    results = predictor.predict_file(path, persist_outputs=False)
    
    # Either it gets dropped by silence/NR gate, or it routes to NOISE
    if results:
        for r in results:
            assert r['decision'] == 'noise', f"White noise misclassified as {r['decision']}"

def test_pink_noise(predictor, test_audio_dir):
    path = _generate_and_save(_make_pink, 'test_pink.wav', test_audio_dir)
    results = predictor.predict_file(path, persist_outputs=False)
    
    if results:
        for r in results:
            assert r['decision'] == 'noise', f"Pink noise misclassified as {r['decision']}"

def test_brown_noise(predictor, test_audio_dir):
    path = _generate_and_save(_make_brown, 'test_brown.wav', test_audio_dir)
    results = predictor.predict_file(path, persist_outputs=False)
    
    if results:
        for r in results:
            assert r['decision'] == 'noise', f"Brown noise misclassified as {r['decision']}"

def test_silence(predictor, test_audio_dir):
    sr = 48000
    duration = 3.0
    sil = np.zeros(int(sr*duration), dtype=np.float32)
    path = test_audio_dir / 'test_silence.wav'
    sf.write(str(path), sil, sr)
    
    results = predictor.predict_file(path, persist_outputs=False)
    # Silence MUST be dropped
    assert len(results) == 0, "Silence was not dropped by the gate"

def test_sine_wave(predictor, test_audio_dir):
    sr = 48000
    duration = 3.0
    t = np.arange(int(sr*duration)) / sr
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    path = test_audio_dir / 'test_sine.wav'
    sf.write(str(path), sine, sr)
    
    results = predictor.predict_file(path, persist_outputs=False)
    if results:
        for r in results:
            assert r['decision'] == 'noise', f"Sine wave misclassified as {r['decision']}"

def test_insects(predictor, test_audio_dir):
    path = _generate_and_save(_make_insects, 'test_insects.wav', test_audio_dir)
    results = predictor.predict_file(path, persist_outputs=False)
    
    if results:
        for r in results:
            assert r['decision'] == 'noise', f"Insects misclassified as {r['decision']}"
