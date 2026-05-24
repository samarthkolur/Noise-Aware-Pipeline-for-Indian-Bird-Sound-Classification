import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from utils.config import load_config
from inference.predictor import Predictor
from generate_synthetic_noise import _make_white, _make_pink, _make_brown, _make_rain, _make_wind, _make_insects

def generate_test_files():
    sr = 48000
    duration = 3.0
    rng = np.random.default_rng(42)
    test_files = {}

    # 1. Pure white noise
    wn = _make_white(rng, int(sr*duration), sr)
    sf.write('test_white.wav', wn, sr)
    test_files['Pure White Noise'] = 'test_white.wav'

    # 2. Pink noise
    pn = _make_pink(rng, int(sr*duration), sr)
    sf.write('test_pink.wav', pn, sr)
    test_files['Pink Noise'] = 'test_pink.wav'

    # 3. Brown noise
    bn = _make_brown(rng, int(sr*duration), sr)
    sf.write('test_brown.wav', bn, sr)
    test_files['Brown Noise'] = 'test_brown.wav'

    # 4. Silence
    sil = np.zeros(int(sr*duration), dtype=np.float32)
    sf.write('test_silence.wav', sil, sr)
    test_files['Silence'] = 'test_silence.wav'

    # 5. Single sine wave (1 kHz)
    t = np.arange(int(sr*duration)) / sr
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    sf.write('test_sine.wav', sine, sr)
    test_files['Sine Wave'] = 'test_sine.wav'

    # Insect audio
    ins = _make_insects(rng, int(sr*duration), sr)
    sf.write('test_insects.wav', ins, sr)
    test_files['Insects'] = 'test_insects.wav'

    return test_files

def run_diagnostics():
    cfg = load_config('config.yaml')
    predictor = Predictor(cfg)
    test_files = generate_test_files()
    
    report = []
    
    for name, path in test_files.items():
        results = predictor.predict_file(Path(path), persist_outputs=False)
        report.append(f"--- TEST: {name} ---")
        if not results:
            report.append("Result: DROPPED (Likely silenced by gate/NR)")
        else:
            for r in results:
                report.append(f"Decision: {r['decision'].upper()}")
                report.append(f"Prob: {r.get('confidence', 0.0):.4f}")
                report.append(f"AE Error: {r.get('recon_error', 0.0):.5f} (Reject: {r.get('recon_error_rejected', False)})")
                report.append(f"Routed By: {r.get('routed_by', 'unknown')}")
        report.append("\n")
        
    print("\n".join(report))

if __name__ == "__main__":
    run_diagnostics()
