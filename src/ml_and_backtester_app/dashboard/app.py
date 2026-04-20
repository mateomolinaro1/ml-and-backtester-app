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
import logging
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv

# Load .env from the project root before anything else so AWS credentials are
# available for both the analytics pipeline and this dashboard process.
from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.dashboard.s3_loader import S3PathManager
load_dotenv()
config = Config()
paths = S3PathManager(config)
_ROOT = config.ROOT_DIR
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{_ROOT / 'mlflow.db'}"))

import pandas as pd # noqa: E402
import dash # noqa: E402
import dash_bootstrap_components as dbc
from fastapi import FastAPI, HTTPException
from flask import Flask
from a2wsgi import WSGIMiddleware
from pydantic import BaseModel

from ml_and_backtester_app.dashboard.callbacks import register
from ml_and_backtester_app.dashboard.layout import create_layout

logger = logging.getLogger(__name__)

MLFLOW_MODEL_NAME = "forecasting_best_model"
MLFLOW_MODEL_ALIAS = "production"


class PredictRequest(BaseModel):
    features: dict
# ─── Dash / Flask ─────────────────────────────────────────────────────────────

# ─── Dash / Flask ─────────────────────────────────────────────────────────────

flask_server = Flask(__name__)

dash_app = dash.Dash(
    __name__,
    server=flask_server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="ML Backtester Dashboard",
)

# PASSAGE DES CHEMINS : On donne l'objet 'paths' au layout et aux callbacks
dash_app.layout = create_layout(paths) 
register(dash_app, paths)

# ─── FastAPI wrapper ──────────────────────────────────────────────────────────

api = FastAPI(title="ML Backtester API", version="1.0.0")


@api.get("/api/health", tags=["ops"])
async def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}

@api.get("/api/model/info", tags=["ml"])
async def model_info() -> dict:
    """Return metadata about the model currently tagged as production."""
    try:
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
        run = client.get_run(version.run_id)
        return {
            "model_name": MLFLOW_MODEL_NAME,
            "alias": MLFLOW_MODEL_ALIAS,
            "version": version.version,
            "run_id": version.run_id,
            "best_model": run.data.params.get("best_model", "unknown"),
            "oos_rmse": run.data.metrics.get("oos_rmse"),
            "sign_accuracy": run.data.metrics.get("sign_accuracy"),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No production model found: {e}")


@api.post("/api/predict", tags=["ml"])
async def predict(request: PredictRequest) -> dict:
    """Load the production model from MLflow registry and return a prediction."""
    try:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not load model: {e}")

    try:
        X = pd.DataFrame([request.features])
        prediction = model.predict(X)
        return {
            "prediction": float(prediction[0]),
            "model_name": MLFLOW_MODEL_NAME,
            "alias": MLFLOW_MODEL_ALIAS,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

# Mount the Dash/Flask app — FastAPI routes take priority, everything else
# falls through to Dash.
api.mount("/", WSGIMiddleware(flask_server))

# ─── Dev entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ml_and_backtester_app.dashboard.app:api", host="0.0.0.0", port=8050, reload=False)
