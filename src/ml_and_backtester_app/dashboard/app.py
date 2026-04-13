"""
Entry point for the ML Backtester Dashboard.

Architecture:
  - Dash (Flask-backed) serves the interactive dashboard at /
  - FastAPI wraps Flask via WSGIMiddleware and adds REST endpoints at /api/

Run with:
    uvicorn ml_and_backtester_app.dashboard.app:api --host 0.0.0.0 --port 8050 --reload

Or directly (dev mode):
    python -m dashboard.app
"""

import os
from dotenv import load_dotenv

# Load .env from the project root before anything else so AWS credentials are
# available for both the analytics pipeline and this dashboard process.
from ml_and_backtester_app.utils.config import Config
load_dotenv()
config = Config()
_ROOT = config.ROOT_DIR

import dash
import dash_bootstrap_components as dbc
from fastapi import FastAPI
from flask import Flask
from a2wsgi import WSGIMiddleware

from ml_and_backtester_app.dashboard.callbacks import register
from ml_and_backtester_app.dashboard.layout import create_layout

# ─── Dash / Flask ─────────────────────────────────────────────────────────────

flask_server = Flask(__name__)

dash_app = dash.Dash(
    __name__,
    server=flask_server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="ML Backtester Dashboard",
)

dash_app.layout = create_layout()
register(dash_app)

# ─── FastAPI wrapper ──────────────────────────────────────────────────────────

api = FastAPI(title="ML Backtester API", version="1.0.0")


@api.get("/api/health", tags=["ops"])
async def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}


# Mount the Dash/Flask app — FastAPI routes take priority, everything else
# falls through to Dash.
api.mount("/", WSGIMiddleware(flask_server))

# ─── Dev entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard.app:api", host="0.0.0.0", port=8050, reload=False)
