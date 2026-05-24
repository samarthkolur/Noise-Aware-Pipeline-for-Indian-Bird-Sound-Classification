import torch
import numpy as np
from inference.predictor import Predictor
from utils.config import load_config
import soundfile as sf
from pathlib import Path

cfg = load_config('config.yaml')
cfg['data']['output_dir'] = './test_output'

sr = 48000
duration = 3.0
noise = np.random.randn(int(sr * duration)).astype(np.float32)

# Scale to -38 dB
current_rms_db = 20 * np.log10(np.sqrt(np.mean(noise**2)))
gain_db = -38.0 - current_rms_db
noise = noise * (10 ** (gain_db / 20.0))

sf.write('test_white_noise_quiet.wav', noise, sr)

predictor = Predictor(cfg)
results = predictor.predict_file(Path('test_white_noise_quiet.wav'), persist_outputs=False)

for r in results:
    print(f"Decision: {r['decision']}, Prob: {r['confidence']:.4f}, AE_Err: {r['recon_error']:.4f}, AE_Reject: {r['recon_error_rejected']}")

