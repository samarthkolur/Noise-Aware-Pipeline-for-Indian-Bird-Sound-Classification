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
noise = np.zeros(int(sr * duration), dtype=np.float32)

sf.write('test_silence.wav', noise, sr)

predictor = Predictor(cfg)
results = predictor.predict_file(Path('test_silence.wav'), persist_outputs=False)

for r in results:
    print(f"Decision: {r['decision']}, Prob: {r['confidence']:.4f}, AE_Err: {r['recon_error']:.4f}, AE_Reject: {r['recon_error_rejected']}")

