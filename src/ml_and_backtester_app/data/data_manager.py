"""
Data manager module.

This module provides the DataManager class responsible for loading
and managing financial datasets stored in S3.
"""

from __future__ import annotations
from typing import Dict
import pandas as pd
from better_aws import AWS
from ml_and_backtester_app.utils.config import Config


class DataManager:
    """
    Load and process financial datasets stored in S3.

    :ivar config: Configuration object containing paths and AWS settings.
    :ivar fred_data: Cleaned FRED data.
    :ivar returns_data: Return data.
    :ivar code_transfo: Mapping of FRED variable names to transformation codes.
    :ivar aws: AWS client used to interact with S3.
    """

    def __init__(self, config: Config):
        """
        Initialize the data manager.

        :param config: Configuration object containing settings for data loading.
        """
        self.config = config

        self.fred_data: pd.DataFrame | None = None
        self.returns_data: pd.DataFrame | None = None
        self.code_transfo: Dict[str, int | float] | None = None
        self.aws: AWS | None = None

        self.load()

    # ---------- Public API ---------- #
    def load(self) -> None:
        """
        Load and process data from S3.
        """
        self._init_s3()
        raw = self._fetch_from_s3()
        self._process(raw)

    # ---------- Internal helpers ---------- #
    def _init_s3(self) -> None:
        """
        Initialize the AWS client and configure S3 defaults.
        """
        self.aws = AWS(region=self.config.aws_default_region, verbose=True)
        # Optional sanity check
        self.aws.identity(print_info=True)
        # 2) Configure S3 defaults
        self.aws.s3.config(
            bucket=self.config.aws_bucket_name,
            output_type="pandas",  # tabular loads -> pandas (or "polars")
            file_type="parquet",  # default tabular format for dataframe uploads without extension
            overwrite=True,
            allow_unsafe_serialization=True,  # Allow loading of potentially unsafe data (e.g., pkl)
        )

    def _fetch_from_s3(self) -> dict[str, pd.DataFrame]:
        """
        Fetch raw data from S3.

        :return: Dictionary containing raw dataframes loaded from S3.
        """
        return {
            "fred": self.aws.s3.load(key=self.config.fred_path),
            "codes": self.aws.s3.load(key=self.config.codes_path),
            "prices": self.aws.s3.load(key=self.config.prices_path),
        }

    def _process(self, raw: dict[str, pd.DataFrame]) -> None:
        """
        Process raw data loaded from S3.

        :param raw: Dictionary containing the raw dataframes.
        """
        # self.code_transfo = self._extract_fred_transform_codes(raw["fred"])
        self.code_transfo = raw["codes"]
        # self.fred_data = self._clean_fred(raw["fred"])
        self.fred_data = raw["fred"]
        self.returns_data = raw["prices"]

    # ---------- FRED-specific logic ---------- #
    @staticmethod
    def _extract_fred_transform_codes(fred: pd.DataFrame) -> dict[str, float]:
        """
        Extract transformation codes from the raw FRED dataframe.

        :param fred: Raw FRED dataframe.
        :return: Dictionary mapping variable names to transformation codes.
        """
        res = dict(zip(fred.columns, fred.loc["Transform:", :]))
        res = dict(sorted(res.items()))
        return res

    @staticmethod
    def _clean_fred(fred: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw FRED dataframe.

        :param fred: Raw FRED dataframe.
        :return: Cleaned FRED dataframe.
        """
        fred = fred.iloc[1:].copy()
        fred.index = pd.to_datetime(fred.index, format="%m/%d/%Y")
        fred = fred.sort_index(axis=1)
        return fred
