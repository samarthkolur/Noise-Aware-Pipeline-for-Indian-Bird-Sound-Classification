# Single-service image for the noise-aware bird classification pipeline (DD-001, DD-002).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --extra dev || uv sync --no-install-project --extra dev

COPY . .
RUN uv sync --frozen --extra dev || uv sync --extra dev

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "scripts/benchmark.py", "--help"]
