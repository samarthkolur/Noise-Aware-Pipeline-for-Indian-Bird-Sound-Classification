#!/usr/bin/env bash
# Host validation + local env prep (DD-001). Cross-platform: bash on macOS/Linux/WSL2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "==> Syncing Python environment (uv sync)..."
uv sync --extra dev

echo "==> Ensuring data/artifacts/outputs directories exist..."
mkdir -p data/raw data/segments artifacts outputs/bird outputs/uncertain outputs/noise outputs/review/likely_fp outputs/review/likely_fn outputs/plots models

BIRDNET_MODEL_PATH="${BIRDNET_MODEL_PATH:-models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite}"
if [ ! -f "$BIRDNET_MODEL_PATH" ]; then
    echo "==> BirdNET V2.4 weights not found at $BIRDNET_MODEL_PATH."
    echo "    This repo uses the 'birdnet' PyPI package, which downloads and caches"
    echo "    model weights automatically on first use (pipeline/embedding.py)."
    echo "    No manual download step is required for the default 'tensorflow' backend."
else
    echo "==> Found BirdNET weights at $BIRDNET_MODEL_PATH"
fi

echo "==> Bootstrap complete. Run 'docker compose watch' or 'uv run pytest' next."
