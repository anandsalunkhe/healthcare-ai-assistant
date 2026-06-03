# ── Stage: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim

# Install uv for fast, reproducible package management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the dependency manifest AND lockfile so Docker can cache the install layer.
# Source code changes won't bust this cache unless dependencies change.
COPY pyproject.toml uv.lock ./

# Install declared dependencies without installing the project itself.
# uv creates a .venv inside /app that `uv run` picks up automatically.
RUN uv sync --no-dev --no-install-project

# Copy application source and data
COPY README.md ./
COPY app/ ./app/
COPY data/ ./data/

# HuggingFace cache dir — shared volume keeps models across container restarts
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache

# Ensure data directory exists
RUN mkdir -p data

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# `uv run` executes inside the managed virtual environment
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
