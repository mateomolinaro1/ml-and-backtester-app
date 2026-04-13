FROM python:3.12-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Install dependencies first (better layer caching) ─────────────────────────
# Copy only the files uv needs to resolve the dependency graph.
# The project source is NOT copied yet so this layer is reused on every
# code-only change.
COPY pyproject.toml uv.lock ./

# Stub out the package so uv can install deps without the real source
RUN mkdir -p src/ml_and_backtester_app && \
    touch src/ml_and_backtester_app/__init__.py && \
    uv sync --frozen --no-install-project

# ── Copy source and install the project itself ────────────────────────────────
COPY src/      src/
COPY config/   config/
COPY main.py   ./

RUN uv sync --frozen

# ── Runtime setup ─────────────────────────────────────────────────────────────
# Add the venv to PATH so we can call uvicorn directly (better SIGTERM handling
# than going through `uv run`).
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8050

# Single worker — pipeline_runner.py holds pipeline state in memory, which is
# not safe to share across multiple processes.
CMD ["sh", "-c", "uvicorn ml_and_backtester_app.dashboard.app:api --host 0.0.0.0 --port ${PORT:-8050} --workers 1"]
