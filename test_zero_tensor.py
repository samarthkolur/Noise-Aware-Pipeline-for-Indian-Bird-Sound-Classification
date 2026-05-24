import torch
import numpy as np
from embedding.embedding import build_encoder
from utils.config import load_config
from inference.prediction_api import predict_embeddings_mlp
from inference.predictor import Predictor

cfg = load_config('config.yaml')
encoder = build_encoder(cfg)
predictor = Predictor(cfg)

zeros = np.zeros(int(48000 * 3.0), dtype=np.float32)
emb = encoder.encode(zeros, 48000)

print(f"Zero tensor embedding: mean={emb.mean():.6f}, std={emb.std():.6f}")

emb_pt = torch.from_numpy(emb).unsqueeze(0).to(predictor.device)
prob = predict_embeddings_mlp(predictor.classifier, emb_pt, binary=True)
print(f"MLP Probability for Zero tensor: {prob.item():.4f}")


ae_model = predictor.autoencoder
ae_model.eval()
reconstructed, _ = ae_model(emb_pt)
ae_err = ((emb_pt - reconstructed)**2).mean(dim=1)
print(f"AE Recon error for zero tensor: {ae_err.item():.5f}")

