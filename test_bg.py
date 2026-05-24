import numpy as np
from preprocessing.noise_segregation_v2 import NoiseSegregationV2
from utils.config import load_config

sr = 48000
duration = 3.0
noise = np.random.randn(int(sr * duration)).astype(np.float32)
noise /= np.max(np.abs(noise))

cfg = load_config('config.yaml')
v2 = NoiseSegregationV2.from_config(cfg)

hr = v2._harmonic_ratio(noise)
sp = v2._spectral_peak_prominence(noise)

print(f"Harmonic ratio: {hr:.4f}")
print(f"Spectral prominence: {sp:.4f}")
