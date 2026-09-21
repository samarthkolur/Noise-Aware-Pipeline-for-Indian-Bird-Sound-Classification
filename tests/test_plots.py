import numpy as np

from pipeline.plots import (
    plot_ae_mse_histogram,
    plot_confusion_matrix,
    plot_embedding_projection,
    plot_pr_curve,
    plot_roc_curve,
)


def _binary_labels_and_probs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def test_plot_roc_curve_writes_file(tmp_path):
    y_true, _, y_prob = _binary_labels_and_probs()
    out = tmp_path / "roc.png"
    plot_roc_curve(y_true, y_prob, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_pr_curve_writes_file(tmp_path):
    y_true, _, y_prob = _binary_labels_and_probs()
    out = tmp_path / "pr.png"
    plot_pr_curve(y_true, y_prob, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_confusion_matrix_writes_file(tmp_path):
    y_true, y_pred, _ = _binary_labels_and_probs()
    out = tmp_path / "cm.png"
    plot_confusion_matrix(y_true, y_pred, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_ae_mse_histogram_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    mse_values = rng.exponential(0.1, size=200)
    out = tmp_path / "ae_hist.png"
    plot_ae_mse_histogram(mse_values, tau_ae=0.2, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_embedding_projection_pca_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(30, 64))
    labels = rng.integers(0, 2, size=30)
    out = tmp_path / "pca.png"
    plot_embedding_projection(embeddings, labels, out, method="pca")
    assert out.exists() and out.stat().st_size > 0
