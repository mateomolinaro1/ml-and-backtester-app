"""
worker.py — runs on your local machine.

Polls an SQS queue for pipeline jobs sent by the dashboard, runs main.py
locally, and streams logs + status back to S3 so the dashboard can display
them in real time.

Setup:
    1. Create an SQS Standard queue named  pipeline-jobs  in your AWS console.
    2. Add  SQS_JOBS_QUEUE_URL=<queue-url>  to your .env file.
    3. Start the worker before (or after) launching the dashboard:

        python worker.py

The worker keeps running until you press Ctrl-C.  It is safe to restart it
at any time — an in-flight pipeline will be re-queued by SQS if the worker
disappears before deleting the message.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv
from ml_and_backtester_app.utils.config import Config
config = Config()
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ROOT_DIR = config.ROOT_DIR
CONFIG_PATH = ROOT_DIR / "config" / "run_pipeline_config.json"

BUCKET = os.getenv("AWS_BUCKET_NAME", "ml-and-backtester-app")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
JOBS_QUEUE_URL = os.getenv("SQS_JOBS_QUEUE_URL", "")

# S3 keys — must match dashboard/pipeline_runner.py
STATUS_KEY = "outputs/pipeline/status.json"
LOG_KEY = "outputs/pipeline/pipeline.log"
STOP_KEY = "outputs/pipeline/stop_requested"

# Write accumulated log to S3 every N lines (keeps S3 API calls low)
LOG_FLUSH_EVERY = 20


# ─── AWS helpers ─────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3", region_name=REGION)


def _sqs():
    return boto3.client("sqs", region_name=REGION)


def _write_status(status: str, job_id: str = "") -> None:
    _s3().put_object(
        Bucket=BUCKET,
        Key=STATUS_KEY,
        Body=json.dumps({"status": status, "job_id": job_id}).encode(),
        ContentType="application/json",
    )


def _flush_log(lines: list[str]) -> None:
    _s3().put_object(
        Bucket=BUCKET,
        Key=LOG_KEY,
        Body="".join(lines).encode("utf-8"),
        ContentType="text/plain",
    )


def _stop_requested() -> bool:
    try:
        _s3().head_object(Bucket=BUCKET, Key=STOP_KEY)
        return True
    except Exception:
        return False


def _clear_stop_signal() -> None:
    try:
        _s3().delete_object(Bucket=BUCKET, Key=STOP_KEY)
    except Exception:
        pass


# ─── Pipeline execution ───────────────────────────────────────────────────────

def _run_pipeline(config: dict, job_id: str) -> None:
    logger.info("Job %s — saving config and launching pipeline...", job_id)

    # Persist the config that the dashboard sent
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # Reset log and status
    _flush_log([])
    _write_status("running", job_id)

    process = subprocess.Popen(
        [sys.executable, str(ROOT_DIR / "main.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT_DIR),
        env=os.environ.copy(),
    )

    log_lines: list[str] = []

    for line in process.stdout:
        print(line, end="", flush=True)   # mirror to local console
        log_lines.append(line)

        if len(log_lines) % LOG_FLUSH_EVERY == 0:
            _flush_log(log_lines)

            # Check for stop signal every flush
            if _stop_requested():
                logger.info("Stop signal received — terminating pipeline.")
                process.terminate()
                _clear_stop_signal()
                log_lines.append("\n[Pipeline terminated by user.]\n")
                break

    process.wait()

    # Final flush
    _flush_log(log_lines)

    if process.returncode == 0:
        status = "done"
    elif process.returncode is None:
        status = "idle"   # terminated cleanly
    else:
        status = "error"

    _write_status(status, job_id)
    logger.info("Job %s finished — status: %s (exit code %s)", job_id, status, process.returncode)


# ─── Main polling loop ────────────────────────────────────────────────────────

def poll() -> None:
    if not JOBS_QUEUE_URL:
        logger.error("SQS_JOBS_QUEUE_URL is not set in .env — exiting.")
        sys.exit(1)

    logger.info("Worker ready. Polling queue: %s", JOBS_QUEUE_URL)

    while True:
        try:
            response = _sqs().receive_message(
                QueueUrl=JOBS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,   # long polling — cheaper and faster than short polling
            )
        except Exception as exc:
            logger.error("SQS receive_message failed: %s — retrying in 5 s", exc)
            time.sleep(5)
            continue

        for msg in response.get("Messages", []):
            # Delete immediately so another worker won't pick it up
            _sqs().delete_message(
                QueueUrl=JOBS_QUEUE_URL,
                ReceiptHandle=msg["ReceiptHandle"],
            )

            try:
                body = json.loads(msg["Body"])
            except json.JSONDecodeError:
                logger.warning("Received malformed SQS message — skipping.")
                continue

            if body.get("action") == "run":
                _run_pipeline(body["config"], body.get("job_id", "unknown"))
            else:
                logger.warning("Unknown action '%s' — skipping.", body.get("action"))


if __name__ == "__main__":
    poll()
