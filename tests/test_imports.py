import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_import_package():
    import ml_and_backtester_app

    assert ml_and_backtester_app is not None


def test_import_utils_config():
    from ml_and_backtester_app.utils.config import Config

    assert Config is not None