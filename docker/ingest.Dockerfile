# The ingestion side: Dagster with Docling, torch and LlamaIndex (the `ingest` extra of pyproject.toml).
# Runs dagster-code, dagster-webserver and dagster-daemon. The API and the UI use serve.Dockerfile.
FROM python:3.12-slim

# opencv (pulled in by Docling) needs these at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .
# No NVIDIA GPU here, so install the CPU-only torch wheels first; Docling then finds torch satisfied
# instead of pulling several GB of CUDA libraries.
RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match torch torchvision
# Dependencies only. Source is bind-mounted at /app/src and found through PYTHONPATH.
RUN uv pip install --system -r pyproject.toml --extra ingest

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/dagster_home

RUN mkdir -p /opt/dagster/dagster_home /opt/dagster/storage
COPY docker/dagster.yaml docker/workspace.yaml /opt/dagster/dagster_home/
