"""
S3 data loading utilities for the dashboard.

Uses boto3 directly (presigned URLs for PNGs, get_object for parquets).
All S3 keys are relative to the bucket root (no bucket prefix).
"""

import io
import logging
import os

import boto3
import pandas as pd
from botocore.config import Config as BotocoreConfig

logger = logging.getLogger(__name__)

BUCKET = os.getenv("AWS_BUCKET_NAME", "ml-and-backtester-app")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

# Fail fast: don't let boto3 hang the Dash callback for 60 s on network issues
_BOTO_CFG = BotocoreConfig(connect_timeout=5, read_timeout=15, retries={"max_attempts": 1})


def _s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=_BOTO_CFG,
    )


def presigned_url(key: str, expires: int = 3600) -> str | None:
    """Return a presigned GET URL for *key*, or None on any failure."""
    try:
        return _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expires,
        )
    except Exception as exc:
        logger.warning("Could not generate presigned URL for %s: %s", key, exc)
        return None


def load_parquet(key: str) -> pd.DataFrame | None:
    """Download *key* from S3 and return it as a DataFrame, or None on any failure."""
    try:
        response = _s3().get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(response["Body"].read()))
    except Exception as exc:
        logger.warning("Could not load parquet %s: %s", key, exc)
        return None


# On récupère la méthode depuis la config (assure-toi que 'method' est défini)
# ─── S3 Dynamic Path Manager ────────────────────────────────────────────────

class S3PathManager:
    def __init__(self, config):
        """
        Génère les chemins S3 dynamiquement selon la config.
        """
        # On récupère les valeurs depuis l'objet config
        self.method = str(config.estimation_method).lower()
        self.base = config.outputs_path  # Ex: "outputs"
        self.bucket = config.aws_bucket_name

        # 1. Dossiers de base par module
        self.fmp_root = f"{self.base}/fmp/{self.method}"
        self.forecast_root = f"{self.base}/forecasting/{self.method}"
        self.alloc_root = f"{self.base}/dynamic_allocation/{self.method}"
        self.backtest_root = f"{self.base}/backtest"

    @property
    def FMP_FIGURES(self) -> dict:
        return {
            "Betas Distribution": f"{self.fmp_root}/figures/bayesian_betas_distribution.png",
            "Betas Over Time": f"{self.fmp_root}/figures/bayesian_betas_overtime.png",
            "R² Over Time": f"{self.fmp_root}/figures/rsquared_overtime.png",
            "Significance Proportion": f"{self.fmp_root}/figures/proportion_significant_bayesian_betas_overtime.png",
            "Betas Summary": f"{self.fmp_root}/figures/bayesian_vs_non_bayesian_betas_summary.png",
            "R² Summary": f"{self.fmp_root}/figures/rsquared_summary.png",
            "Equity Curves (static)": f"{self.fmp_root}/figures/fmp_equity_curves.png",
            "Performance Summary (static)": f"{self.fmp_root}/figures/fmp_performance_summary.png",
        }

    @property
    def FORECASTING_FIGURES(self) -> dict:
        return {
            "Features Sample": f"{self.forecast_root}/figures/features_df_short.png",
            "Best Val Score (static)": f"{self.forecast_root}/figures/best_val_score_all_models_overtime.png",
            "Best Hyperparams": f"{self.forecast_root}/figures/best_hyperparams_all_models_overtime.png",
            "Model Parameters": f"{self.forecast_root}/figures/best_parameters_all_models_overtime.png",
            "Selected Features": f"{self.forecast_root}/figures/proportion_selected_features.png",
            "Mean Parameters": f"{self.forecast_root}/figures/mean_parameters.png",
            "OOS RMSE Overtime (static)": f"{self.forecast_root}/figures/oos_rmse_all_models_overtime.png",
            "OOS RMSE Table (static)": f"{self.forecast_root}/figures/oos_rmse_all_models.png",
            "Sign Accuracy (static)": f"{self.forecast_root}/figures/oos_sign_accuracy_all_models.png",
        }

    @property
    def DYNAMIC_ALLOC_FIGURES(self) -> dict:
        return {
            "Cumulative Returns (static)": f"{self.alloc_root}/figures/dynamic_allocation_cum_returns.png",
            "Performance Table (static)": f"{self.alloc_root}/figures/performance_table.png",
        }

    @property
    def DATA(self) -> dict:
        """Chemins vers les fichiers PARQUET pour les graphiques interactifs"""
        return {
            "fmp_equity_curves": f"{self.fmp_root}/data/fmp_equity_curves.parquet",
            "fmp_performance": f"{self.fmp_root}/data/fmp_performance_table.parquet",
            "best_val_score": f"{self.forecast_root}/data/best_val_score_overtime.parquet",
            "oos_rmse_overtime": f"{self.forecast_root}/data/oos_rmse_overtime.parquet",
            "oos_rmse_table": f"{self.forecast_root}/data/oos_rmse_table.parquet",
            "oos_sign_accuracy": f"{self.forecast_root}/data/oos_sign_accuracy.parquet",
            "dynamic_alloc_cum_returns": f"{self.alloc_root}/data/dynamic_allocation_cum_returns.parquet",
            "dynamic_alloc_performance": f"{self.alloc_root}/data/dynamic_allocation_performance_table.parquet",
        }
    
    @property
    def BACKTEST_FIGURES(self) -> dict:
        """Chemins vers les images générées par le backtester"""
        return {
            "Cumulative Performance": f"{self.backtest_root}/figures/cumulative_performance.png",
        }