import torch

from pipeline.config import AutoencoderConfig, MLPConfig
from pipeline.train_loop import compute_tau_ae, train_autoencoder, train_focal_mlp

MLP_CFG = MLPConfig(
    hidden_dims=[32, 16],
    dropout=0.1,
    focal_gamma=2.0,
    focal_alpha=0.25,
    learning_rate=0.01,
    weight_decay=0.0,
    batch_size=16,
    max_epochs=5,
    early_stop_patience=2,
)

AE_CFG = AutoencoderConfig(bottleneck_dim=8, hidden_dim=32, ae_percentile=99.0)

DIM = 64


def _separable_dataset(n_per_class=40, dim=DIM, seed=0):
    torch.manual_seed(seed)
    bird = torch.randn(n_per_class, dim) + 3.0
    noise = torch.randn(n_per_class, dim) - 3.0
    x = torch.cat([bird, noise])
    y = torch.cat([torch.ones(n_per_class), torch.zeros(n_per_class)])
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


def test_train_focal_mlp_reaches_good_val_f1():
    train_x, train_y = _separable_dataset(40)
    val_x, val_y = _separable_dataset(10)

    model, history = train_focal_mlp(train_x, train_y, val_x, val_y, MLP_CFG, embedding_dim=DIM)

    assert len(history.train_loss) > 0
    assert max(history.val_metric) > 0.8

    model.eval()
    with torch.no_grad():
        preds = (model(val_x) >= 0.5).float()
    accuracy = (preds == val_y).float().mean().item()
    assert accuracy > 0.8


def test_train_autoencoder_reduces_reconstruction_error():
    train_x, train_y = _separable_dataset(40)
    bird_x = train_x[train_y == 1]
    val_x, val_y = _separable_dataset(10)
    val_bird_x = val_x[val_y == 1]

    model, history = train_autoencoder(
        bird_x, val_bird_x, AE_CFG, embedding_dim=DIM, max_epochs=10, patience=10
    )

    assert history.val_metric[-1] <= history.val_metric[0]


def test_compute_tau_ae_is_positive_percentile():
    train_x, train_y = _separable_dataset(40)
    bird_x = train_x[train_y == 1]
    model, _ = train_autoencoder(
        bird_x, bird_x, AE_CFG, embedding_dim=DIM, max_epochs=5, patience=5
    )

    tau = compute_tau_ae(model, bird_x, percentile=99.0)
    assert tau > 0.0

    tau_50 = compute_tau_ae(model, bird_x, percentile=50.0)
    assert tau >= tau_50  # P99 should be >= P50
