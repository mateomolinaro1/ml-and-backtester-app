"""
Dashboard-side pipeline interface.

Instead of launching main.py locally, this module:
  - pushes run-jobs to AWS SQS  (picked up by worker.py on your machine)
  - reads status / log output from S3  (written there by worker.py)

Required env vars:
    SQS_JOBS_QUEUE_URL      URL of the pipeline-jobs SQS queue
    AWS_BUCKET_NAME         defaults to ml-and-backtester-app
    AWS_DEFAULT_REGION      defaults to eu-north-1
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
"""

import json
import logging
import os
import uuid

import boto3
from botocore.config import Config as BotocoreConfig
from ml_and_backtester_app.utils.config import Config

logger = logging.getLogger(__name__)

_config = Config()
_ROOT_DIR = _config.ROOT_DIR
CONFIG_PATH = _ROOT_DIR / "config" / "run_pipeline_config.json"

BUCKET = os.getenv("AWS_BUCKET_NAME", "ml-and-backtester-app")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
JOBS_QUEUE_URL = os.getenv("SQS_JOBS_QUEUE_URL", "")

# S3 keys — must match worker.py
STATUS_KEY = "outputs/pipeline/status.json"
LOG_KEY = "outputs/pipeline/pipeline.log"
STOP_KEY = "outputs/pipeline/stop_requested"

# Short timeouts so a missing object or network blip never hangs the UI
_BOTO_CFG = BotocoreConfig(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1})


def _s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=_BOTO_CFG,
    )


def _sqs():
    return boto3.client(
        "sqs",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=_BOTO_CFG,
    )


# ─── Public API (called by callbacks.py) ─────────────────────────────────────


def load_config() -> dict:
    """Load the config from the file baked into the image / local repo."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_status() -> str:
    """Read the current pipeline status from S3. Returns idle/running/done/error."""
    try:
        response = _s3().get_object(Bucket=BUCKET, Key=STATUS_KEY)
        data = json.loads(response["Body"].read())
        return data.get("status", "idle")
    except Exception:
        return "idle"


def get_output() -> str:
    """Read the accumulated pipeline log from S3."""
    try:
        response = _s3().get_object(Bucket=BUCKET, Key=LOG_KEY)
        return response["Body"].read().decode("utf-8", errors="replace")
    except Exception:
        return ""


# Modifie la signature de la fonction pour accepter task_type
def start(config_json: str, task_type: str = "run") -> tuple[bool, str]:
    """
    Validate *config_json* and push a run-job to SQS.
    task_type can be 'run' (default) or 'backtest'.
    """
    if not JOBS_QUEUE_URL:
        return False, "SQS_JOBS_QUEUE_URL is not configured."

    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON — {exc}"

    if get_status() == "running":
        return False, "Pipeline is already running."

    # --- LOGIQUE DYNAMIQUE ICI ---
    # Si task_type est "backtest", on envoie "run_backtest", sinon on envoie "run"
    action = "run_backtest" if task_type == "backtest" else "run"
    
    job_id = str(uuid.uuid4())[:8]
    message = json.dumps({
        "action": action, 
        "config": cfg, 
        "job_id": job_id
    })
    # -----------------------------

    try:
        _sqs().send_message(QueueUrl=JOBS_QUEUE_URL, MessageBody=message)
    except Exception as exc:
        return False, f"Could not send job to SQS: {exc}"

    return True, f"Task '{action}' (Job {job_id}) queued — waiting for the worker."


def stop() -> str:
    """Ask worker.py to terminate the running pipeline via an S3 flag file."""
    try:
        _s3().put_object(Bucket=BUCKET, Key=STOP_KEY, Body=b"")
        return "Stop signal sent to worker."
    except Exception as exc:
        return f"Could not send stop signal: {exc}"
