# The serving side: the API, the Streamlit UI, the command line and `setup` (the `serve` extra of
# pyproject.toml). No Docling, torch or Dagster, so it is small and starts in seconds.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .
# Dependencies only. Source is bind-mounted at /app/src and found through PYTHONPATH.
RUN uv pip install --system -r pyproject.toml --extra serve

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
