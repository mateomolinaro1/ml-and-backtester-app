"""
Manages pipeline execution state for the dashboard.

Runs main.py in a background subprocess, streams stdout/stderr into an
in-memory buffer, and exposes live output + status to Dash callbacks.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from ml_and_backtester_app.utils.config import Config
config = Config()

# src/ml_and_backtester_app/dashboard/ → 3 parents → src/ → 1 more → project root
ROOT_DIR = config.ROOT_DIR
CONFIG_PATH = ROOT_DIR / "config" / "run_pipeline_config.json"

_lock = threading.Lock()
_process: subprocess.Popen | None = None
_output_lines: list[str] = []
_status: str = "idle"  # idle | running | done | error


# ─── Public API ───────────────────────────────────────────────────────────────
def load_config() -> dict:
    """Read the current config JSON from disk."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_status() -> str:
    """Return the pipeline status, refreshing if the process just finished."""
    global _status, _process
    if _status == "running" and _process is not None and _process.poll() is not None:
        with _lock:
            _status = "done" if _process.returncode == 0 else "error"
    return _status


def get_output() -> str:
    """Return all captured stdout/stderr so far as a single string."""
    with _lock:
        return "".join(_output_lines)


def start(config_json: str) -> tuple[bool, str]:
    """
    Validate *config_json*, write it to disk, then launch main.py.

    Returns (success, message).
    """
    global _process, _output_lines, _status

    # Validate JSON before touching the file
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON — {exc}"

    with _lock:
        if _status == "running" and _process is not None and _process.poll() is None:
            return False, "Pipeline is already running."

        # Persist updated config
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
        except OSError as exc:
            return False, f"Could not save config: {exc}"

        _output_lines = []
        _status = "running"
        _process = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "main.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT_DIR),
            env=os.environ.copy(),
        )

    def _reader() -> None:
        global _status
        for line in _process.stdout:
            with _lock:
                _output_lines.append(line)
        _process.wait()
        with _lock:
            _status = "done" if _process.returncode == 0 else "error"

    threading.Thread(target=_reader, daemon=True).start()
    return True, "Pipeline started."


def stop() -> str:
    """Terminate the running pipeline process."""
    global _status, _process
    with _lock:
        if _process is not None and _process.poll() is None:
            _process.terminate()
            _status = "idle"
            return "Pipeline terminated by user."
        return "No pipeline currently running."
