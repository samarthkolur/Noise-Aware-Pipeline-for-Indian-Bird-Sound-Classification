import torch

from pipeline.bird_rescue import BirdRescueMLP, rescue_predict, train_bird_rescue
from pipeline.config import load_config


def test_forward_pass_shape():
    cfg = load_config()
    model = BirdRescueMLP(embedding_dim=1024, hidden_dims=cfg.bird_rescue.hidden_dims)
    x = torch.randn(8, 1024)
    out = model(x)
    assert out.shape == (8,)
    assert torch.all((out >= 0) & (out <= 1))


def test_training_reduces_loss_on_separable_synthetic_embeddings():
    cfg = load_config()
    torch.manual_seed(0)

    n = 200
    dim = 1024
    bird_embeddings = torch.randn(n, dim) + 2.0
    noise_embeddings = torch.randn(n, dim) - 2.0
    embeddings = torch.cat([bird_embeddings, noise_embeddings])
    labels = torch.cat([torch.ones(n), torch.zeros(n)])

    model = train_bird_rescue(embeddings, labels, cfg.bird_rescue, embedding_dim=dim, epochs=50)

    with torch.no_grad():
        preds = model(embeddings)
    accuracy = ((preds >= 0.5).float() == labels).float().mean().item()
    assert accuracy > 0.9


def test_rescue_predict_returns_bool_and_prob():
    cfg = load_config()
    model = BirdRescueMLP(embedding_dim=1024, hidden_dims=cfg.bird_rescue.hidden_dims)
    embedding = torch.randn(1024)
    rescued, prob = rescue_predict(embedding, model, cfg.bird_rescue.rescue_threshold)
    assert isinstance(rescued, bool)
    assert 0.0 <= prob <= 1.0
