import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.stage2_embeddings import extract_birdnet_embedding
import config

def test_birdnet_embedding_extraction():
    """Smoke test for BirdNET embedding extraction."""
    pytest.importorskip("birdnet")
    
    print("\n── Test: BirdNET Embedding Extraction ──")
    sr = config.TARGET_SR
    duration = config.SEGMENT_LENGTH
    
    # Generate 3 seconds of synthetic white noise
    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    
    embedding = extract_birdnet_embedding(audio, sr)
    
    # Verify shape
    assert embedding.shape == (config.EMBEDDING_DIM,), f"Expected shape ({config.EMBEDDING_DIM},), got {embedding.shape}"
    
    # Verify it doesn't return all zeros for random noise
    assert not np.all(embedding == 0), "Embedding is unexpectedly all zeros"
    
    print(f"✅ BirdNET embedding extraction: PASSED (Shape: {embedding.shape})")

if __name__ == "__main__":
    test_birdnet_embedding_extraction()
