"""Phase 8 exit criterion (design.md §10): ONNX output matches PyTorch within 1e-5."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from pipeline.config import load_config
from pipeline.model import BirdAutoencoder, FocalMLP


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _export_and_compare(
    model: torch.nn.Module, onnx_path, input_names, output_names, n_samples=10, dim=1024
):
    model.eval()
    dummy = torch.randn(1, dim)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}},
        opset_version=17,
    )

    test_input = torch.randn(n_samples, dim)
    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {input_names[0]: test_input.numpy()})[0]

    if torch_output.ndim != onnx_output.ndim:
        onnx_output = onnx_output.reshape(torch_output.shape)

    return torch_output, onnx_output


def test_mlp_onnx_roundtrip_matches_pytorch(cfg, tmp_path):
    model = FocalMLP(embedding_dim=1024, config=cfg.mlp)
    torch_out, onnx_out = _export_and_compare(model, tmp_path / "mlp.onnx", ["embedding"], ["prob"])
    np.testing.assert_allclose(torch_out, onnx_out, atol=1e-5)


def test_ae_onnx_roundtrip_matches_pytorch(cfg, tmp_path):
    model = BirdAutoencoder(embedding_dim=1024, config=cfg.autoencoder)
    torch_out, onnx_out = _export_and_compare(
        model, tmp_path / "ae.onnx", ["embedding"], ["reconstruction"]
    )
    np.testing.assert_allclose(torch_out, onnx_out, atol=1e-5)
