import numpy as np
import noisereduce as nr
from embedding.embedding import build_encoder
from inference.predictor import Predictor
from utils.config import load_config
import torch

sr = 48000
duration = 3.0
noise = np.random.randn(int(sr * duration)).astype(np.float32)

# Scale to -38 dB
current_rms_db = 20 * np.log10(np.sqrt(np.mean(noise**2)))
gain_db = -38.0 - current_rms_db
noise = noise * (10 ** (gain_db / 20.0))

reduced = nr.reduce_noise(
    y=noise,
    sr=sr,
    n_std_thresh_stationary=1.5,
    prop_decrease=1.0,
)

print(f"Reduced mean: {reduced.mean():.6f}, std: {reduced.std():.6f}")

cfg = load_config('config.yaml')
encoder = build_encoder(cfg)
emb = encoder.encode(reduced, sr)

predictor = Predictor(cfg)
emb_pt = torch.from_numpy(emb).unsqueeze(0).to(predictor.device)

ae_model = predictor.autoencoder
reconstructed, _ = ae_model(emb_pt)
ae_err = ((emb_pt - reconstructed)**2).mean(dim=1)

from inference.prediction_api import predict_embeddings_mlp
prob = predict_embeddings_mlp(predictor.classifier, emb_pt, binary=True)

print(f"Reduced Noise - AE Error: {ae_err.item():.5f}, MLP Prob: {prob.item():.5f}")

