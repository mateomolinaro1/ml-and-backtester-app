from __future__ import annotations
import pandas as pd
from typing import Dict
from ml_and_backtester_app.utils.config import Config
from better_aws import AWS


class DataManager:
    def __init__(self, config: Config):
        self.config = config

        self.fred_data: pd.DataFrame | None = None
        self.returns_data: pd.DataFrame | None = None
        self.code_transfo: Dict[str, int|float] | None = None
        self.aws: AWS | None = None

        self.load()

    # ---------- Public API ---------- #
    def load(self) -> None:
        self._init_s3()
        raw = self._fetch_from_s3()
        self._process(raw)

    # ---------- Internal helpers ---------- #
    def _init_s3(self) -> None:
        self.aws = AWS(
            region=self.config.aws_default_region,
            verbose=True
        )
        # Optional sanity check
        self.aws.identity(print_info=True)
        # 2) Configure S3 defaults
        self.aws.s3.config(
            bucket=self.config.aws_bucket_name,
            output_type="pandas",  # tabular loads -> pandas (or "polars")
            file_type="parquet",  # default tabular format for dataframe uploads without extension
            overwrite=True,
        )
    def _fetch_from_s3(self) -> dict[str, pd.DataFrame]:
        return {
            "fred": self.aws.s3.load(
                key=self.config.fred_path
            ),
            "codes": self.aws.s3.load(
                key=self.config.codes_path
            ),
            "prices": self.aws.s3.load(
                key=self.config.prices_path
            ),
        }

    def _process(self, raw: dict[str, pd.DataFrame]) -> None:
        # self.code_transfo = self._extract_fred_transform_codes(raw["fred"])
        self.code_transfo = raw["codes"]
        # self.fred_data = self._clean_fred(raw["fred"])
        self.fred_data = raw["fred"]
        self.returns_data = raw["prices"]

    # ---------- FRED-specific logic ---------- #
    @staticmethod
    def _extract_fred_transform_codes(fred: pd.DataFrame) -> dict[str, float]:
        res = dict(
            zip(fred.columns, fred.loc["Transform:", :])
        )
        res = dict(sorted(res.items()))
        return res

    @staticmethod
    def _clean_fred(fred: pd.DataFrame) -> pd.DataFrame:
        fred = fred.iloc[1:].copy()
        fred.index = pd.to_datetime(fred.index, format="%m/%d/%Y")
        fred = fred.sort_index(axis=1)
        return fred
