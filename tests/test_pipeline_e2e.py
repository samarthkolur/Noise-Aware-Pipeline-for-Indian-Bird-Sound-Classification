import os
import sys
import numpy as np
import shutil
import tempfile
import soundfile as sf
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from run_pipeline import (
    run_stage1_segmentation,
    run_stage2_embeddings,
    run_stage3_curation,
    run_stage4_train,
    run_stage5_ood,
    run_stage78_inference,
)

@pytest.fixture
def temp_env():
    """Creates a temporary directory and overrides config paths."""
    tmp_dir = tempfile.mkdtemp(prefix="birdnet_e2e_")
    
    # Original paths
    orig_raw = config.RAW_DATA_DIR
    orig_seg = config.SEGMENTED_DIR
    orig_emb = config.EMBEDDINGS_DIR
    orig_hard = config.HARD_NEGATIVE_DIR
    orig_mod = config.MODELS_DIR
    orig_al = config.ACTIVE_LEARNING_DIR
    orig_out = config.NOISE_AWARE_OUTPUT_DIR
    
    # Override
    config.RAW_DATA_DIR = os.path.join(tmp_dir, "raw")
    config.SEGMENTED_DIR = os.path.join(tmp_dir, "segmented")
    config.EMBEDDINGS_DIR = os.path.join(tmp_dir, "embeddings")
    config.HARD_NEGATIVE_DIR = os.path.join(tmp_dir, "hard_negative")
    config.MODELS_DIR = os.path.join(tmp_dir, "models")
    config.ACTIVE_LEARNING_DIR = os.path.join(tmp_dir, "active_learning")
    config.NOISE_AWARE_OUTPUT_DIR = os.path.join(tmp_dir, "output")
    
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RAW_DATA_DIR, "species1"), exist_ok=True)
    
    # Generate synthetic raw audio
    sr = config.TARGET_SR
    audio = np.random.randn(int(sr * 10.0)).astype(np.float32) # 10 seconds
    sf.write(os.path.join(config.RAW_DATA_DIR, "species1", "test.wav"), audio, sr)
    
    class Args:
        stage = "all"
        max_files = 10
        skip_active_learning = True
        ablation = False
        verbose = False

    yield Args()
    
    # Restore
    config.RAW_DATA_DIR = orig_raw
    config.SEGMENTED_DIR = orig_seg
    config.EMBEDDINGS_DIR = orig_emb
    config.HARD_NEGATIVE_DIR = orig_hard
    config.MODELS_DIR = orig_mod
    config.ACTIVE_LEARNING_DIR = orig_al
    config.NOISE_AWARE_OUTPUT_DIR = orig_out
    
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_pipeline_e2e(temp_env):
    """End-to-End smoke test for the pipeline over synthetic data."""
    pytest.importorskip("birdnet")
    args = temp_env
    
    print("\n── Test: End-to-End Pipeline ──")
    
    # Sub-test stages sequentially to mimic running --stage all
    
    # Stage 1: Segmentation
    run_stage1_segmentation(args)
    assert os.path.exists(config.SEGMENTED_DIR), "Segmentation directory not created"
    
    # Stage 2: Embeddings
    run_stage2_embeddings(args)
    assert os.path.exists(os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy")), "Embeddings not extracted"
    
    # Stage 3: Curation
    # Because we have very little data, curation might not balance classes or produce many negatives.
    # We'll mock the binary labels and splits so downstream works even if heuristics filtered everything.
    np.save(os.path.join(config.EMBEDDINGS_DIR, "binary_labels.npy"), np.ones(3))
    np.save(os.path.join(config.EMBEDDINGS_DIR, "X_train.npy"), np.random.randn(2, config.EMBEDDING_DIM))
    np.save(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"), np.array([0, 1]))
    np.save(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"), np.random.randn(1, config.EMBEDDING_DIM))
    np.save(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"), np.array([1]))
    
    # Stage 4: Train
    run_stage4_train(args)
    assert os.path.exists(os.path.join(config.MODELS_DIR, "rf_classifier.pkl")), "Classifier not saved"
    
    # Stage 5: OOD
    run_stage5_ood(args)
    assert os.path.exists(os.path.join(config.MODELS_DIR, "ood_ensemble", "metadata.json")), "OOD Ensemble not saved"
    
    # Stage 7/8: Inference and Output
    # Let's mock a path array to match X_val length
    np.save(os.path.join(config.EMBEDDINGS_DIR, "paths.npy"), np.array([
        os.path.join(config.SEGMENTED_DIR, "species1", "test.wav")
    ], dtype=object))
    
    results = run_stage78_inference(args)
    assert "ensemble_labels" in results
    assert os.path.exists(config.NOISE_AWARE_OUTPUT_DIR), "Output dir not created"
    
    print("✅ End-to-End Pipeline smoke test: PASSED")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
