FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/      src/
COPY config/   config/
COPY main.py   ./
RUN uv pip install -e . --no-deps

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8050

CMD ["sh", "-c", "uvicorn ml_and_backtester_app.dashboard.app:api --host 0.0.0.0 --port ${PORT:-8050} --workers 1"]
