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
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

BUCKET = os.getenv("AWS_BUCKET_NAME", "ml-and-backtester-app")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")


def _s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def presigned_url(key: str, expires: int = 3600) -> str | None:
    """Return a presigned GET URL for *key*, or None if the object is missing."""
    try:
        return _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expires,
        )
    except ClientError as exc:
        logger.warning("Could not generate presigned URL for %s: %s", key, exc)
        return None


def load_parquet(key: str) -> pd.DataFrame | None:
    """Download *key* from S3 and return it as a DataFrame, or None on failure."""
    try:
        response = _s3().get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(response["Body"].read()))
    except ClientError as exc:
        logger.warning("Could not load parquet %s: %s", key, exc)
        return None


# ─── S3 keys: PNG figures ────────────────────────────────────────────────────

FMP_FIGURES: dict[str, str] = {
    "Betas Distribution": "outputs/figures/bayesian_betas_distribution.png",
    "Betas Over Time": "outputs/figures/bayesian_betas_overtime.png",
    "R\u00b2 Over Time": "outputs/figures/rsquared_overtime.png",
    "Significance Proportion": "outputs/figures/proportion_significant_bayesian_betas_overtime.png",
    "Betas Summary": "outputs/figures/bayesian_vs_non_bayesian_betas_summary.png",
    "R\u00b2 Summary": "outputs/figures/rsquared_summary.png",
    "Equity Curves (static)": "outputs/figures/fmp_equity_curves.png",
    "Performance Summary (static)": "outputs/figures/fmp_performance_summary.png",
}

FORECASTING_FIGURES: dict[str, str] = {
    "Features Sample": "outputs/figures/features_df_short.png",
    "Best Val Score (static)": "outputs/figures/best_val_score_all_models_overtime.png",
    "Best Hyperparams": "outputs/figures/best_hyperparams_all_models_overtime.png",
    "Model Parameters": "outputs/figures/best_parameters_all_models_overtime.png",
    "Selected Features": "outputs/figures/proportion_selected_features.png",
    "Mean Parameters": "outputs/figures/mean_parameters.png",
    "OOS RMSE Overtime (static)": "outputs/figures/oos_rmse_all_models_overtime.png",
    "OOS RMSE Table (static)": "outputs/figures/oos_rmse_all_models.png",
    "Sign Accuracy (static)": "outputs/figures/oos_sign_accuracy_all_models.png",
}

DYNAMIC_ALLOC_FIGURES: dict[str, str] = {
    "Cumulative Returns (static)": "outputs/figures/dynamic_allocation_cum_returns.png",
    "Performance Table (static)": "outputs/figures/performance_table.png",
}

# ─── S3 keys: parquet data (for interactive charts) ──────────────────────────

DATA: dict[str, str] = {
    "fmp_equity_curves": "outputs/figures/fmp_equity_curves.parquet",
    "fmp_performance": "outputs/figures/fmp_performance_table.parquet",
    "best_val_score": "outputs/figures/best_val_score_overtime.parquet",
    "oos_rmse_overtime": "outputs/figures/oos_rmse_overtime.parquet",
    "oos_rmse_table": "outputs/figures/oos_rmse_table.parquet",
    "oos_sign_accuracy": "outputs/figures/oos_sign_accuracy.parquet",
    "dynamic_alloc_cum_returns": "outputs/figures/dynamic_allocation_cum_returns.parquet",
    "dynamic_alloc_performance": "outputs/figures/dynamic_allocation_performance_table.parquet",
}
