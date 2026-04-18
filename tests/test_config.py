import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ml_and_backtester_app.utils.config import Config


def test_config_instantiates():
    config = Config()
    assert config is not None


def test_root_dir_exists():
    config = Config()
    assert Path(config.ROOT_DIR).exists()


def test_config_files_exist():
    config = Config()
    root = Path(config.ROOT_DIR)

    assert (root / "config" / "run_pipeline_config.json").exists()
    assert (root / "config" / "backtest_config.json").exists()


def test_models_attribute_exists():
    config = Config()
    assert hasattr(config, "models")