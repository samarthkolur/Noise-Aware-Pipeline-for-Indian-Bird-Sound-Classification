import torch

from pipeline.config import load_config
from pipeline.losses import FocalLoss
from pipeline.model import BirdAutoencoder, FocalMLP


def test_focal_mlp_forward_pass_shape():
    cfg = load_config()
    model = FocalMLP(embedding_dim=1024, config=cfg.mlp)
    x = torch.randn(16, 1024)
    out = model(x)
    assert out.shape == (16,)
    assert torch.all((out >= 0) & (out <= 1))


def test_autoencoder_forward_pass_shape():
    cfg = load_config()
    model = BirdAutoencoder(embedding_dim=1024, config=cfg.autoencoder)
    x = torch.randn(16, 1024)
    recon = model(x)
    assert recon.shape == x.shape


def test_autoencoder_reconstruction_error_shape_and_nonnegative():
    cfg = load_config()
    model = BirdAutoencoder(embedding_dim=1024, config=cfg.autoencoder)
    x = torch.randn(16, 1024)
    errors = model.reconstruction_error(x)
    assert errors.shape == (16,)
    assert torch.all(errors >= 0)


def test_focal_loss_zero_when_prediction_matches_target_perfectly():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    probs = torch.tensor([1.0 - 1e-7, 1e-7])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(probs, targets)
    assert loss.item() < 1e-4


def test_focal_loss_positive_for_wrong_predictions():
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    probs = torch.tensor([0.1, 0.9])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(probs, targets)
    assert loss.item() > 0.0


def test_focal_loss_down_weights_easy_examples_vs_bce():
    """The modulating factor (1-p_t)^gamma should make focal loss smaller than
    plain BCE for an easy, well-classified example."""
    focal = FocalLoss(gamma=2.0, alpha=0.5)
    probs = torch.tensor([0.95])
    targets = torch.tensor([1.0])
    focal_loss = focal(probs, targets).item()
    bce_loss = torch.nn.functional.binary_cross_entropy(probs, targets).item()
    assert focal_loss < bce_loss
