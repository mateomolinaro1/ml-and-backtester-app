import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
from pandas.tseries.offsets import BDay
from typing import List, Union, Dict
import logging
import wrds
from ib_insync import IB, Stock, util
import boto3
from botocore.client import BaseClient
from pathlib import Path
import io
from ml_and_backtester_app.utils.s3_utils import s3Utils

logger = logging.getLogger(__name__)

class DataHandler:
    """
    A class to handle data fetching and processing from WRDS and Interactive Brokers (IB).
    It requires to have the IB Gateway or TWS running and configured to accept API connections (right host
    and port, which you can find in the IB Gateway settings).
    More info: https://www.interactivebrokers.eu/campus/ibkr-api-page/twsapi-doc/#api-introduction
    Attributes:
    - wrds_username: WRDS username for database connection.
    - ib_host: Host address for IB API connection.
    - ib_port: Port number for IB API connection.
    - ib_client_id: Client ID for IB API connection.
    - pause: Pause duration between IB API requests to avoid rate limiting.
    - retries: Number of retries for IB API requests in case of failure.
    - reconnect_wait: Wait time before reconnecting to IB API after a disconnection.
    -
    Methods:
    """
    def __init__(self,
                 data_path: Path,
                 wrds_username:str,
                 wrds_password:str,
                 bucket_name:str,
                 ib_host:str = '127.0.0.1',
                 ib_port:int = 4002,
                 ib_client_id:int = 1,
                 pause = 0.40,
                 retries = 3,
                 reconnect_wait = 2,
    )->None:
        """
        Initializes the DataHandler with WRDS and IB connection parameters.
        :parameters:
        - data_path: path of the data folder.
        - wrds_username: WRDS username for database connection.
        - ib_host: Host address for IB API connection.
        - ib_port: Port number for IB API connection.
        - ib_client_id: Client ID for IB API connection.
        - pause: Pause duration between IB API requests to avoid rate limiting.
        - retries: Number of retries for IB API requests in case of failure.
        - reconnect_wait: Wait time before reconnecting to IB API after a disconnection.
        - wrds_db: WRDS database connection object.
        - ib: IB API connection object.
        - wrds_gross_query: Raw WRDS query result DataFrame.
        - crsp_to_ib_mapping_tickers: Dictionary mapping CRSP tickers to IB tickers.
        - wrds_universe: Processed WRDS universe DataFrame.
        - wrds_universe_last_date: Last date in the WRDS universe DataFrame.
        - universe_prices_wrds: WRDS universe prices DataFrame.
        - universe_returns_wrds: WRDS universe returns DataFrame.
        - fields_wrds_to_keep_for_universe: List of fields to keep in the WRDS universe DataFrame.
        - crsp_to_ib_exchange: Dictionary mapping CRSP exchange codes to IB exchange names.
        - crps_exchcd_to_currency: Dictionary mapping CRSP exchange codes to currency codes.
        - tickers_across_dates: List of unique tickers across all dates in the WRDS universe.
        - ib_tickers: List of IB tickers corresponding to CRSP tickers.
        - dates: List of unique dates in the WRDS universe.
        - universe_prices_ib: IB universe prices DataFrame.
        - universe_returns_ib: IB universe returns DataFrame.
        - valid_tickers_per_ib_date: Series of valid tickers per IB date.
        - valid_permnos_per_ib_date: Series of valid PERMNOs per IB date.
        """
        self.data_path = data_path
        self.wrds_username = wrds_username
        self.wrds_password = wrds_password
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_client_id = ib_client_id
        self.pause = pause
        self.retries = retries
        self.reconnect_wait = reconnect_wait

        self.wrds_db = None
        self.ib = None
        self._manual_disconnect = False
        self._shutting_down = False

        self.wrds_gross_query = None
        self.crsp_to_ib_mapping_tickers = None
        self.wrds_universe = None
        self.wrds_universe_last_date = None
        self.universe_prices_wrds = None
        self.universe_returns_wrds = None
        self.fields_wrds_to_keep_for_universe = ['ticker',
                                                 'exchcd',
                                                 'cusip',
                                                 'ncusip',
                                                 'comnam',
                                                 'permno',
                                                 'permco',
                                                 'namedt',
                                                 'nameendt',
                                                 'date']
        self.crsp_to_ib_exchange = {
            1: "NYSE",
            2: "AMEX",
            3: "NASDAQ",
            4: "ARCA",
            11: "BATS",
            31: "NASDAQ",
            32: "NASDAQ",
            33: "NASDAQ"
        }
        self.crps_exchcd_to_currency = {
            1:'USD',
            2:'USD',
            3:'USD'
        }
        self.tickers_across_dates = None
        self.ib_tickers = None
        self.dates = None

        self.universe_prices_ib = None
        self.universe_returns_ib = None
        self.valid_tickers_per_ib_date = None
        self.valid_permnos_per_ib_date = None

        self.aws_credentials = None
        self.s3 = None
        self.file_paths_and_s3_object_names = {
            self.data_path / "ib_tickers.pkl":"data/ib_tickers.pkl",
            self.data_path / "crsp_to_ib_mapping_tickers.pkl":"data/crsp_to_ib_mapping_tickers.pkl",
            self.data_path / "wrds_universe.parquet":"data/wrds_universe.parquet",
            self.data_path / "wrds_gross_query.parquet":"data/wrds_gross_query.parquet",
            self.data_path / "ib_historical_prices.parquet":"data/ib_historical_prices.parquet",
            self.data_path / "tickers_across_dates.pkl":"data/tickers_across_dates.pkl",
            self.data_path / "dates.pkl":"data/dates.pkl",
            self.data_path / "portfolio_value_historical.parquet":"paper_trading/portfolio_value_historical.parquet"
        }
        self.bucket_name = bucket_name
        self.s3_files_downloaded = None

    def connect_wrds(self):
        """Establishes a connection to the WRDS database using the provided username."""
        self.wrds_db = wrds.Connection(wrds_username=self.wrds_username,
                                       wrds_password=self.wrds_password)

    def logout_wrds(self):
        """Logs out from the WRDS database connection."""
        if self.wrds_db is not None:
            self.wrds_db.close()
            self.wrds_db = None

    def connect_ib(self):
        if self.ib is None:
            self.ib = IB()

        if not self.ib.isConnected():
            self._manual_disconnect = False
            self._shutting_down = False
            self.ib.connect(
                host=self.ib_host,
                port=self.ib_port,
                clientId=self.ib_client_id
            )
            self.ib.disconnectedEvent += lambda *args: self.on_ib_disconnected(*args)
            self.ib.sleep(1)
            logger.info("Connected to IB.")

    # 'Manual' check. Will be called before requesting
    def ensure_ib_connection(self):
        if not self.ib.isConnected():
            logger.warning("IB disconnected — reconnecting...")
            self.connect_ib()
            self.ib.sleep(2)

    # As between the above check and the moment ib request, ib can still disconnect, we need a
    # method that is automatically triggered on event, when ib disconnect. This is what the below
    # Function does.
    def on_ib_disconnected(self, *args):
        if self._shutting_down:
            logger.info("IB disconnect during shutdown — ignoring.")
            return

        if self._manual_disconnect:
            logger.info("IB disconnected manually — no reconnection needed.")
            return

        logger.warning("IB disconnected unexpectedly — reconnecting...")

        try:
            self.ib.connect(
                host=self.ib_host,
                port=self.ib_port,
                clientId=self.ib_client_id
            )

            # Give IB time
            self.ib.sleep(2)

            logger.info("IB reconnected successfully.")

        except Exception as e:
            logger.error(f"IB reconnection failed: {e}")

    def logout_ib(self):
        """Disconnects from the Interactive Brokers API."""
        if self.ib is not None and self.ib.isConnected():
            logger.info("Disconnecting from IB (manual).")
            self._shutting_down = True
            self._manual_disconnect = True
            self.ib.disconnect()
            self.ib = None

    def load_data(self):
        """
        Loads previously saved data from disk.
        Loads WRDS universe, CRSP to IB ticker mapping, and IB historical prices from disk.
        Requires the data files: 'wrds_universe.parquet', 'crsp_to_ib_mapping_tickers.pkl', and
        'ib_historical_prices.parquet' in 'data' folder.
        To get these files, run the get_data_first_time.py script in 'scripts' folder.
        """
        # Load WRDS universe
        if self.wrds_universe is None:
            try:
                self.wrds_universe = pd.read_parquet(self.data_path / "wrds_universe.parquet")
                self.format_wrds_historical_universe(from_cloud=False)
            except Exception as e:
                logger.error(f"Error reading WRDS universe: {e}")
                raise ValueError("wrds_universe data is not loaded. Please fetch it first.")

        # Load CRSP to IB mapping
        if self.crsp_to_ib_mapping_tickers is None:
            try:
                with open(self.data_path / "crsp_to_ib_mapping_tickers.pkl", "rb") as f:
                    self.crsp_to_ib_mapping_tickers = pickle.load(f)
            except Exception as e:
                logger.error(f"Error reading crsp_to_ib_mapping_tickers.pkl: {e}")
                raise ValueError("crsp_to_ib_mapping_tickers not loaded.")

        # Load ib_historical_prices
        if self.universe_prices_ib is None:
            try:
                self.universe_prices_ib = pd.read_parquet(self.data_path / "ib_historical_prices.parquet")
                self.trim_data_survivorship_free_ib()
            except Exception as e:
                logger.error(f"Error reading IB historical prices: {e}")
                raise ValueError("IB historical prices not loaded. Please fetch it first.")

        if self.valid_permnos_per_ib_date is None or self.valid_tickers_per_ib_date is None:
            # WRDS dates as a proper DataFrame with a column
            wrds_dates = (
                pd.DataFrame({"wrds_date": pd.to_datetime(self.wrds_universe.index.unique())})
                .sort_values("wrds_date")
            )

            # IB dates also as a column
            ib_dates = (
                pd.DataFrame({"ib_date": pd.to_datetime(self.universe_prices_ib.index)})
                .sort_values("ib_date")
            )

            # Perform nearest-past merge
            aligned_dates = pd.merge_asof(
                ib_dates,
                wrds_dates,
                left_on="ib_date",
                right_on="wrds_date",
                direction="backward"
            ).set_index("ib_date")

            if 'ticker_ib' not in self.wrds_universe.columns:
                self.format_wrds_historical_universe(from_cloud=False)

            if self.valid_tickers_per_ib_date is None:
                tickers_by_date = self.wrds_universe.groupby(level=0)["ticker_ib"].apply(set)
                # For each IB date, get the set of tickers valid at the corresponding WRDS date
                self.valid_tickers_per_ib_date = aligned_dates["wrds_date"].apply(
                    lambda d: tickers_by_date.get(d, set())
                )
            if self.valid_permnos_per_ib_date is None:
                permnos_by_date = self.wrds_universe.groupby(level=0)["permno"].apply(set)
                self.valid_permnos_per_ib_date = aligned_dates["wrds_date"].apply(
                    lambda d: permnos_by_date.get(d, set())
                )

    def crsp_ticker_to_ib_ticker(self,
                                 save_mapping_locally:bool=True,
                                 save_ib_tickers_to_cloud:bool=False
                                 ) -> None:
        """Maps CRSP ticker to IB ticker safely with throttling + retries."""
        if self.crsp_to_ib_mapping_tickers is None:
            self.crsp_to_ib_mapping_tickers = {}

        if self.ib is None or not self.ib.isConnected():
            self.connect_ib()

        if self.tickers_across_dates is None:
            try:
                with open(self.data_path / "tickers_across_dates.pkl", "rb") as f:
                    self.tickers_across_dates = pickle.load(f)
            except Exception as e:
                logger.error(f"Error reading tickers_across_dates.pkl: {e}")
                raise ValueError("Tickers across dates not loaded.")

        self.ib_tickers = []

        def safe_ib_match(query: str):
            """Safe wrapper around reqMatchingSymbols with throttling + retry."""
            if not isinstance(query, str):
                logger.error(f"Query must be a string. Got {type(query)} instead.")
                raise ValueError("Query must be a string.")
            for attempt in range(self.retries):
                try:
                    if not self.ib.isConnected():
                        logger.warning("IB disconnected — reconnecting...")
                        self.ib.disconnect()
                        time.sleep(1)
                        self.connect_ib()
                        time.sleep(self.reconnect_wait)

                    result = self.ib.reqMatchingSymbols(query)
                    time.sleep(self.pause)
                    return result

                except Exception as exc:
                    logger.error(
                        f"Error querying IB for {query}: {exc} "
                        f"(attempt {attempt + 1}/{self.retries})"
                    )
                    time.sleep(1.5)

            return []

        # ---- MAIN LOOP ----
        for i, crsp_ticker in enumerate(self.tickers_across_dates):

            if i % 50 == 0:
                logger.info(f"matching CRSP ticker {i + 1}/{len(self.tickers_across_dates)}")

            matches = safe_ib_match(crsp_ticker)

            # --- No match ---
            if not matches or matches[0].contract.symbol in (None, ""):
                logger.info(f"No IB match for CRSP ticker {crsp_ticker}")
                print(f"No IB match for CRSP ticker {crsp_ticker}")
                continue

            ib_ticker = matches[0].contract.symbol

            self.ib_tickers.append(ib_ticker)
            self.crsp_to_ib_mapping_tickers[crsp_ticker] = ib_ticker

        if save_mapping_locally:
            # --- Save mapping ---
            with open(self.data_path / "crsp_to_ib_mapping_tickers.pkl", "wb") as f:
                pickle.dump(self.crsp_to_ib_mapping_tickers, f)

        if save_ib_tickers_to_cloud:
            # --- Save ib_tickers to cloud ---
            if self.s3 is None:
                self.connect_aws_s3()
            ib_tickers_bytes = pickle.dumps(self.ib_tickers)
            self.s3.put_object(Bucket=self.bucket_name,
                               Key='data/ib_tickers.pkl',
                               Body=ib_tickers_bytes)

        logger.info("CRSP -> IB ticker mapping completed successfully.")

    def fetch_wrds_historical_universe(self,
                                       wrds_request:str,
                                       starting_date:str,
                                       date_cols:List[str],
                                       saving_config:dict,
                                       save_tickers_across_dates:bool=True,
                                       save_dates:bool=True,
                                       return_bool:bool=False,
                                       crsp_to_ib_mapping_tickers_from_cloud:bool=False,
                                       update_crsp_to_ib_mapping_tickers:bool=False,
                                       save_ib_tickers_to_cloud:bool=False
                                       )->Union[None,dict]:
        """
        Fetches historical universe from WRDS based on the provided SQL request. It saves wrds_gross_query
        and wrds_universe to disk if specified in saving_config.
        :parameters:
        - wrds_request: SQL query string to fetch data from WRDS.
        - date_cols: List of columns in the query result that should be parsed as dates.
        - saving_config: Dictionary specifying saving paths and formats for gross query and universe.
        - return_bool: If True, returns the fetched universe DataFrame.
        """
        # Check input data types
        if not isinstance(wrds_request, str):
            logger.error("wrds_request must be a string.")
            raise ValueError("wrds_request must be a string containing the SQL query.")
        if not isinstance(starting_date, str):
            logger.error("starting_date must be a string.")
            raise ValueError("starting_date must be a string in 'YYYY-MM-DD' format.")
        if not isinstance(date_cols, list):
            logger.error("date_cols must be a list of strings.")
            raise ValueError("date_cols must be a list of strings.")
        for col in date_cols:
            if not isinstance(col, str):
                logger.error("All elements in date_cols must be strings.")
                raise ValueError("All elements in date_cols must be strings.")
        if not isinstance(saving_config, dict):
            logger.error("saving_config must be a dictionary.")
            raise ValueError("saving_config must be a dictionary.")
        if not isinstance(return_bool, bool):
            logger.error("return_bool must be a boolean.")
            raise ValueError("return_bool must be a boolean.")

        # Ensure connection to WRDS
        if self.wrds_db is None:
            self.connect_wrds()

        # Query WRDS database
        wrds_request = wrds_request.format(starting_date=starting_date)
        self.wrds_gross_query = self.wrds_db.raw_sql(sql=wrds_request,
                                                     date_cols=date_cols)

        # As unique identifiers of WRDS/CRSP are PERMNO and not ticker but for IB it is ticker,
        # We have to ensure (date, permno) are unique, and then we will have to create a mapping
        # between CRSP tickers and IB tickers
        self.wrds_gross_query = self.wrds_gross_query.drop_duplicates(subset=['date', 'permno'],
                                                                      keep='last')
        # Sort for checking
        self.wrds_gross_query = self.wrds_gross_query.sort_values(by=['date'],
                                                                  ascending=True).reset_index(drop=True)
        self.tickers_across_dates = list(self.wrds_gross_query['ticker'].unique())
        if save_tickers_across_dates:
            with open(self.data_path / "tickers_across_dates.pkl", "wb") as f:
                pickle.dump(self.tickers_across_dates, f)

        self.dates = list(self.wrds_gross_query['date'].unique())
        if save_dates:
            with open(self.data_path / "dates.pkl", "wb") as f:
                pickle.dump(self.dates, f)

        # Save gross query if specified
        if 'gross_query' in saving_config:
            if saving_config['gross_query']['extension'] == 'parquet':
                self.wrds_gross_query.to_parquet(saving_config['gross_query']['path'],
                                                index=False)
            else:
                logger.error("Unsupported file extension for gross query.")
                raise ValueError("Unsupported file extension for gross query. Use 'parquet'.")


        universe = self.wrds_gross_query.copy()
        universe.index = universe['date']
        self.wrds_universe = universe

        # Format the universe DataFrame
        self.format_wrds_historical_universe(from_cloud=crsp_to_ib_mapping_tickers_from_cloud,
                                             update_crsp_to_ib_mapping=update_crsp_to_ib_mapping_tickers,
                                             save_ib_tickers_to_cloud=save_ib_tickers_to_cloud)

        # Save to file if a saving path is provided
        if 'universe' in saving_config:
            if saving_config['universe']['extension'] == 'parquet':
                self.wrds_universe.to_parquet(saving_config['universe']['path'],
                                              index=True)
            else:
                logger.error("Unsupported file extension for universe.")
                raise ValueError("Unsupported file extension for universe. Use 'parquet'.")

        if return_bool:
            return {'wrds_gross_query':self.wrds_gross_query,
                    'wrds_universe':self.wrds_universe}

    def build_crsp_to_ib_ticker_mapping(self)->None:
        self.crsp_ticker_to_ib_ticker()
        # Saving
        with open(self.data_path / "ib_tickers.pkl", "wb") as f:
            pickle.dump(self.ib_tickers, f)

    def format_wrds_historical_universe(self,
                                        from_cloud:bool=False,
                                        update_crsp_to_ib_mapping:bool=False,
                                        save_ib_tickers_to_cloud:bool=False
                                        )->None:
        """Formats the WRDS historical universe DataFrame."""
        if self.wrds_universe is None:
            try:
                self.wrds_universe = pd.read_parquet(self.data_path / "wrds_universe.parquet")
            except Exception as e:
                logger.error(f"Error reading WRDS universe: {e}")
                raise ValueError("WRDS universe data is not loaded. Please fetch it first.")

        if self.crsp_to_ib_mapping_tickers is None:
            if not from_cloud:
                try:
                    with open(self.data_path / "crsp_to_ib_mapping_tickers.pkl", "rb") as f:
                        self.crsp_to_ib_mapping_tickers = pickle.load(f)
                except FileNotFoundError as _:
                    try:
                        self.build_crsp_to_ib_ticker_mapping()
                    except Exception as e:
                        logger.error(f"Error reading crsp_to_ib_mapping_tickers.pkl: {e}")
                        raise ValueError("crsp_to_ib_mapping_tickers not loaded.")
            elif from_cloud:
                try:
                    # Connect to s3
                    if self.s3 is not None:
                        pass
                    else:
                        self.connect_aws_s3()
                    # Get the file on s3
                    crsp_to_ib_mapping_tickers = self.get_file_from_s3(s3=self.s3,
                                                                       bucket_name=self.bucket_name,
                                                                       s3_object_name='data/crsp_to_ib_mapping_tickers.pkl')
                    self.crsp_to_ib_mapping_tickers = crsp_to_ib_mapping_tickers

                except Exception as e:
                    logger.error(f"Error reading crsp_to_ib_mapping_tickers from cloud: {e}")
                    raise ValueError("crsp_to_ib_mapping_tickers not loaded from cloud.")
            else:
                logger.error("wrong value for from_cloud entered. Must be  boolean.")
                raise ValueError("wrong value for from_cloud entered. Must be  boolean.")

        wrds_universe = self.wrds_universe.copy()
        fields_wrds_to_keep_for_universe = self.fields_wrds_to_keep_for_universe.copy()
        if 'date' not in wrds_universe.columns and 'date' in fields_wrds_to_keep_for_universe:
            fields_wrds_to_keep_for_universe.remove('date')
        self.wrds_universe = wrds_universe[fields_wrds_to_keep_for_universe].copy()
        self.wrds_universe['exchange'] = (
            self.wrds_universe['exchcd']
            .map(self.crsp_to_ib_exchange)
            .fillna("UNKNOWN")
        )
        self.wrds_universe['currency'] = (
            self.wrds_universe['exchcd']
            .map(self.crps_exchcd_to_currency)
            .fillna("UNKNOWN")
        )
        self.wrds_universe['exchange_ib'] = "SMART"  # Default to SMART for IB

        if update_crsp_to_ib_mapping:
            # As we are updating our universe, we need to update the mapping too
            # To update the mapping, we will do the following: compare the current wrds tickers in
            # crsp_to_ib_mapping_tickers and the wrds tickers that are in the current wrds universe.
            # if there are new tickers in the wrds universe that are not in the mapping, we will
            # fetch their ib mapping and update the mapping dictionary.
            current_wrds_tickers = set(self.wrds_universe['ticker'].unique())
            mapped_wrds_tickers = set(self.crsp_to_ib_mapping_tickers.keys())
            new_wrds_tickers = list(current_wrds_tickers - mapped_wrds_tickers)
            if len(new_wrds_tickers) > 0:
                logger.info(f"Found {len(new_wrds_tickers)} new WRDS tickers to map to IB.")
                # Temporarily set tickers_across_dates to new tickers only
                original_tickers_across_dates = self.tickers_across_dates
                self.tickers_across_dates = new_wrds_tickers
                # Build mapping for new tickers
                self.crsp_ticker_to_ib_ticker(save_mapping_locally=False,
                                              save_ib_tickers_to_cloud=save_ib_tickers_to_cloud)
                # Now in self.crsp_to_ib_mapping_tickers we have the new mappings added
                # Restore original tickers_across_dates
                self.tickers_across_dates = original_tickers_across_dates
            else:
                logger.info("No new WRDS tickers found for mapping to IB.")

        self.wrds_universe['ticker_ib'] = self.wrds_universe['ticker'].map(
            self.crsp_to_ib_mapping_tickers
        )
        # Sort
        self.wrds_universe = self.wrds_universe.sort_index(ascending=True)

    def get_wrds_historical_prices(self,
                                   saving_config:dict,
                                   return_bool:bool=False) -> Union[None, pd.DataFrame]:
        """
        Format self.wrds_gross_query to have a nice prices df.
        :parameters:
        - saving_config: Dictionary specifying saving paths and formats for prices.
        - return_bool: If True, returns the prices DataFrame.
        It either saves the prices DataFrame to disk or returns it based on the parameters.
        """
        if self.wrds_gross_query is None:
            try:
                self.wrds_gross_query = pd.read_parquet(self.data_path / "wrds_gross_query.parquet")
            except Exception as e:
                logger.error(f"Error reading WRDS gross query: {e}")
                raise ValueError("WRDS universe data is not loaded. Please fetch it first.")

        prices = self.wrds_gross_query.pivot(values='prc',
                                             index='date',
                                             columns='permno')
        self.universe_prices_wrds = prices

        if 'prices' in saving_config:
            if saving_config['prices']['extension'] == 'parquet':
                prices.to_parquet(saving_config['prices']['path'],
                              index=True)
            else:
                raise ValueError("Unsupported file extension for prices. Use 'parquet'.")

        if return_bool:
            return prices

    def get_wrds_returns(self,
                         return_bool:bool=False) -> Union[None, pd.DataFrame]:
        """
        Compute returns DataFrame from universe prices
        :parameters:
        - return_bool: If True, returns the returns DataFrame.
        """
        if self.universe_prices_wrds is None:
            raise ValueError("Universe prices data is not loaded. Please fetch it first.")
        returns = self.universe_prices_wrds.pct_change(fill_method=None)
        self.universe_returns_wrds = returns
        if return_bool:
            return returns

    def fetch_ib_historical_prices(self,
                                   end_date:str='',
                                   past_period:str='1 M',
                                   frequency:str='1 day',
                                   data_prices:str='ADJUSTED_LAST',
                                   use_rth:bool=True,
                                   format_date:int=1,
                                   save_prices:bool=True,
                                   load_from_cloud:bool=False,
                                   updating_procedure:bool=False,
                                   return_bool:bool=False)->Union[None, pd.DataFrame]:
        """
        Fetch historical prices from IB for all tickers in 'tickers_across_dates'
        :parameters:
        - end_date: The end date for the historical data in 'YYYYMMDD HH:MM:SS' format.
        - past_period: The duration of historical data to fetch (e.g., '1 M' for 1 month).
            S for seconds, D for days, W for weeks, M for months, Y for years.
        - frequency: The bar size for the historical data (e.g., '1 day', '1 hour').
        - data_prices: The type of data to fetch (e.g., 'ADJUSTED_LAST', 'TRADES', 'BID', 'ASK', 'BID_ASK',
            'MIDPOINT',...)
            see:https://www.interactivebrokers.eu/campus/ibkr-api-page/twsapi-doc/#historical-bars.
        - use_rth: Whether to use regular trading hours only (True) or all hours (False).
        - format_date: The format of the returned date (1 for 'YYYYMMDD HH:MM:SS', 2 for epoch time).
        - save_prices: If True, saves the fetched prices (not yet nicely formatted) to disk.
        - return_bool: If True, returns the fetched prices DataFrame.
        """
        logger.info("Starting IB historical fetch...")
        # --------------------------------------------------------
        # 0. Ensure IB connection
        # --------------------------------------------------------
        if self.ib is None or not self.ib.isConnected():
            logger.info("IB connection not found — attempting to reconnect.")
            self.connect_ib()
            logger.info("Connected to IB.")

        if not load_from_cloud:
            logger.info("Loading data from local disk.")
            # --------------------------------------------------------
            # 1. Load WRDS universe if needed
            # --------------------------------------------------------
            if self.wrds_universe is None:
                file_path = r".\data\wrds_universe.parquet"
                if os.path.exists(file_path):
                    logger.info(f"Loading WRDS universe from {file_path}")
                    self.wrds_universe = pd.read_parquet(file_path,
                                                         index_col='date',
                                                         parse_dates=['date'])
                else:
                    logger.error("WRDS universe file not found.")
                    raise ValueError("WRDS universe data is not loaded. Please fetch it first.")

            # --------------------------------------------------------
            # 2. Load tickers_across_dates if needed
            # --------------------------------------------------------
            if self.tickers_across_dates is None:
                file_path = r".\data\tickers_across_dates.pkl"
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    logger.info(f"Loading tickers from {file_path}")
                    try:
                        with open(file_path, 'rb') as f:
                            self.tickers_across_dates = pickle.load(f)
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        raise ValueError("Failed to load tickers_across_dates from pickle file.")
                else:
                    logger.error("Tickers file missing or empty.")
                    raise ValueError("Tickers across dates data is not loaded. Please fetch WRDS universe first.")

            # --------------------------------------------------------
            # 2.2 Load ib_tickers if needed
            # --------------------------------------------------------
            if self.crsp_to_ib_mapping_tickers is None:
                file_path = r".\data\crsp_to_ib_mapping_tickers.pkl"
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    logger.info(f"Loading tickers from {file_path}")
                    try:
                        with open(file_path, 'rb') as f:
                            self.crsp_to_ib_mapping_tickers = pickle.load(f)
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        raise ValueError("Failed to load crsp_to_ib_mapping_tickers from pickle file.")
                else:
                    logger.error("Tickers file missing or empty.")
                    raise ValueError("crsp_to_ib_mapping_tickers not loaded.")
        else:
            logger.info("Loading data from cloud storage.")
            self.get_files_from_s3()
            self.wrds_universe = self.s3_files_downloaded['wrds_universe']
            self.tickers_across_dates = self.s3_files_downloaded['tickers_across_dates']
            self.crsp_to_ib_mapping_tickers = self.s3_files_downloaded['crsp_to_ib_mapping_tickers']

        # --------------------------------------------------------
        # 3. Prepare dict for results
        # --------------------------------------------------------
        ib_prices_dct = {}
        wrds_df = self.wrds_universe.sort_index(ascending=False)

        # --------------------------------------------------------
        # 4. Helper: safe IB request with pacing retry
        # --------------------------------------------------------
        def safe_req(contract_ib):
            """
            Safe wrapper around IB reqHistoricalData with pacing violation handling.
            :param contract_ib: IB Contract object.
            """
            while True:
                try:
                    self.ensure_ib_connection()
                    return self.ib.reqHistoricalData(
                        contract=contract_ib,
                        endDateTime=end_date,
                        durationStr=past_period,
                        barSizeSetting=frequency,
                        whatToShow=data_prices,
                        useRTH=use_rth,
                        formatDate=format_date
                    )
                except Exception as exc:
                    msg = str(exc).lower()
                    if "pacing" in msg or "rate" in msg or "violation" in msg:
                        print("! Pacing violation -> waiting 30 seconds before retrying...")
                        logger.warning(f"Pacing violation for {contract_ib.symbol}, retrying...")
                        self.ib.sleep(30)
                    else:
                        raise

        # --------------------------------------------------------
        # 4. Loop over tickers
        # --------------------------------------------------------
        if updating_procedure:
            # If we are in the updating procedure, there is no need to loop across all tickers across time,
            # it is sufficient to loop only on the tickers that compose our universe at the last available date.
            last_date = self.wrds_universe.index.max()
            subset_last_date = self.wrds_universe.loc[last_date]
            tickers_to_loop = subset_last_date['ticker'].unique().tolist()
        else:
            tickers_to_loop = self.tickers_across_dates

        for i,ticker in enumerate(tickers_to_loop):
            logger.info(f"Fetching ticker: {ticker} ({i+1}/{len(tickers_to_loop)})")
            print(f"Fetching ticker: {ticker} ({i+1}/{len(tickers_to_loop)})")

            # Extract the most recent data
            subset = wrds_df[wrds_df["ticker"] == ticker]
            if subset.empty:
                print(f"! WARNING: Ticker {ticker} not found in WRDS universe. Skipping.")
                continue
            exchange_ib = subset.iloc[0]["exchange_ib"]
            currency = subset.iloc[0]["currency"]
            primary_exchange = subset.iloc[0]["exchange"]

            # --------------------------------------------------------
            # IB Contract
            # --------------------------------------------------------
            ib_ticker = self.crsp_to_ib_mapping_tickers.get(ticker, np.nan)
            if ib_ticker is np.nan:
                print(f"!! IB ticker is {ib_ticker}. Skipping.")
                logger.warning(f"IB ticker is {ib_ticker}. Skipping.")
                continue
            contract = Stock(symbol=ib_ticker,
                             exchange=exchange_ib,
                             primaryExchange=primary_exchange,
                             currency=currency
                             )

            # --------------------------------------------------------
            # Validate contract
            # --------------------------------------------------------
            try:
                self.ensure_ib_connection()
                details = self.ib.reqContractDetails(contract)
                if not details:
                    print(f"!! IB cannot identify contract for ticker {ib_ticker}. Skipping.")
                    logger.warning(f"Invalid contract for {ib_ticker}. Skipping.")
                    continue
            except Exception as e:
                print(f"!! Error requesting contract details for {ib_ticker}: {e}")
                logger.error(f"Contract details error for {ib_ticker}: {e}")
                continue

            # Safe API call
            try:
                bars = safe_req(contract)
            except Exception as e:
                logger.error(f"Error fetching ticker {ib_ticker}: {e}")
                print(f"!! Error fetching IB data for {ib_ticker}: {e}")
                continue

            df = util.df(bars)

            # Handle edge cases
            if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
                logger.warning(f"No IB data returned for {ib_ticker}.")
                print(f"! WARNING: No data returned for {ib_ticker}. Skipping.")
                continue

            ib_prices_dct[ib_ticker] = df.loc[:,['date', 'close']].set_index('date').rename(columns={'close':ib_ticker})

            # anti rate limit sleep
            self.ib.sleep(1.2)

        # --------------------------------------------------------
        # 5. Format into a single dataframe
        # --------------------------------------------------------
        self.universe_prices_ib = self.format_ib_historical_prices(ib_prices_dct=ib_prices_dct)
        self.universe_prices_ib.sort_index(inplace=True)

        # --------------------------------------------------------
        # 5.2 Trim the data to actually keep universe_prices_ib survivorship-bias free!!
        # --------------------------------------------------------
        if updating_procedure:
            pass # No need to trim if updating because we only fetched the last date tickers
        else:
            logger.info("Trimming IB prices to match WRDS universe tickers per date...")
            self.trim_data_survivorship_free_ib()
            logger.info("Trimming completed.")

        # --------------------------------------------------------
        # 6. Save results
        # --------------------------------------------------------
        if save_prices:
            with open(self.data_path / "ib_historical_prices_dct.pkl", "wb") as f:
                pickle.dump(ib_prices_dct, f)
            self.universe_prices_ib.to_parquet(self.data_path / "ib_historical_prices.parquet",
                                               index=True)
            logger.info("Saved historical prices to disk.")

        logger.info("Finished IB historical fetch successfully.")
        # --------------------------------------------------------
        # 7. Optionally return a value
        # --------------------------------------------------------
        if return_bool:
            return self.universe_prices_ib

    def fetch_portfolio_value(self,
                              saving_path_pv:str|None,
                              return_bool:int=False)->pd.DataFrame|None:
        if self.ib is None:
            self.connect_ib()
        values = self.ib.accountValues()
        portfolio_value = float(next(v.value for v in values if v.tag == 'NetLiquidation'))
        df = pd.DataFrame(index=pd.Series(pd.Timestamp.today().normalize()),
                          data=portfolio_value,
                          columns=["portfolio_value"])

        if return_bool:
            return df

        if saving_path_pv is not None:
            df.to_parquet(path=saving_path_pv)
        # else:
        #     logger.info("ib must be connected before requesting portfolio value")
        #     print("ib must be connected before requesting portfolio value")
        #     return None

    @staticmethod
    def format_ib_historical_prices(ib_prices_dct: dict) -> pd.DataFrame:
        """ Given a dict of DataFrames from IB, format them into a single DataFrame."""
        aligned_df = pd.concat(
            ib_prices_dct.values(),
            axis=1,
            join="outer"  # forces union of all dates
        )
        aligned_df.index = pd.to_datetime(aligned_df.index)
        return aligned_df


    def trim_data_survivorship_free_ib(self)->None:
        if 'ticker_ib' not in self.wrds_universe.columns:
            self.wrds_universe['ticker_ib'] = self.wrds_universe['ticker'].map(
                self.crsp_to_ib_mapping_tickers
            )
        wrds_dates = (
            pd.DataFrame({"wrds_date": pd.to_datetime(self.wrds_universe.index.unique())})
            .sort_values("wrds_date")
        )
        ib_dates = (
            pd.DataFrame({"ib_date": pd.to_datetime(self.universe_prices_ib.index)})
            .sort_values("ib_date")
        )
        # Perform nearest-past merge
        aligned_dates = pd.merge_asof(
            ib_dates,
            wrds_dates,
            left_on="ib_date",
            right_on="wrds_date",
            direction="backward"
        ).set_index("ib_date")
        tickers_by_date = self.wrds_universe.groupby(level=0)["ticker_ib"].apply(set)
        # For each IB date, get the set of tickers valid at the corresponding WRDS date
        valid_tickers_per_ib_date = aligned_dates["wrds_date"].apply(
            lambda d: tickers_by_date.get(d, set())
        )
        permnos_by_date = self.wrds_universe.groupby(level=0)["permno"].apply(set)
        valid_permnos_per_ib_date = aligned_dates["wrds_date"].apply(
            lambda d: permnos_by_date.get(d, set())
        )
        self.valid_permnos_per_ib_date = valid_permnos_per_ib_date
        self.valid_tickers_per_ib_date = valid_tickers_per_ib_date
        mask = pd.DataFrame(
            {col: [col in valid_tickers_per_ib_date[date]
                   for date in self.universe_prices_ib.index]
             for col in self.universe_prices_ib.columns},
            index=self.universe_prices_ib.index
        )
        self.universe_prices_ib = self.universe_prices_ib.where(mask)

    def compute_coverage(self)->None:
        """ Compute coverage of IB prices over WRDS universe prices."""

        if self.valid_tickers_per_ib_date is None or self.valid_permnos_per_ib_date is None:
            self.load_data()

        if self.valid_tickers_per_ib_date.shape[0] != self.universe_prices_ib.shape[0]:
            raise ValueError("Mismatch between dates in valid_tickers_per_ib_date and universe_prices_ib.")

        coverage_crsp_tickers_to_ib_tickers = self.valid_tickers_per_ib_date.apply(len)/self.valid_permnos_per_ib_date.apply(len)
        coverage_crsp_tickers_to_ib_tickers.name = 'convertion_rate_crsp_tickers_to_ib_tickers'

        plt.figure()
        plt.plot(coverage_crsp_tickers_to_ib_tickers)
        plt.title("Convertion rate of CRSP tickers to IB tickers over time")
        # Save fig
        plt.savefig(self.data_path.parent / "outputs" / "figures" / "convertion_rate_crsp_tickers_to_ib_tickers.png")
        plt.close()

        # Now compute prices coverage (ib compared to crsp)
        prices_coverage = self.universe_prices_ib.notna().sum(axis=1)/self.valid_permnos_per_ib_date.apply(len)
        prices_coverage.name = 'prices_coverage_ib_over_crsp_universe'
        plt.figure()
        plt.plot(prices_coverage)
        plt.title("Prices coverage of IB over CRSP universe over time")
        # Save fig
        plt.savefig(self.data_path.parent / "outputs" / "figures" / "prices_coverage_ib_over_crsp_universe.png")
        plt.close()

    def get_credentials(self,
                        return_bool:bool=False
                        )->Dict[str, str]:
        """
        Load credentials from .env .
        :param:
        - return_bool: If True, returns the credentials dict.
        :return: a dict containing the credentials.
        """
        creds = {}
        for key in ["KEY", "SECRET_KEY","REGION","OUTPUT_FORMAT"]:
            val = os.getenv(key)
            if val is not None:
                creds[key] = val

        required = ["KEY", "SECRET_KEY","REGION","OUTPUT_FORMAT"]
        missing = [k for k in required if k not in creds]

        if missing:
            logger.error(f"Missing credentials: {missing}")
            raise RuntimeError(f"Missing credentials: {missing}")

        if self.aws_credentials is None:
            self.aws_credentials = creds

        if return_bool:
            return creds

    def connect_aws_s3(self) -> None:
        """
        Connect to AWS S3 using environment-based credentials.
        Credentials are automatically resolved by boto3 from:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_SESSION_TOKEN (optional)
        - AWS_DEFAULT_REGION
        """
        try:
            self.s3 = boto3.client("s3")
            logger.info("Successfully connected to AWS S3.")
        except Exception as e:
            logger.error(f"Failed to connect to AWS S3: {e}")
            raise RuntimeError("Could not connect to AWS S3. Check AWS credentials and region.")


    def upload_file_to_s3(self,
                          file_paths_and_s3_object_names:dict,
                          bucket_name:str
                          )->None:
        """
        Upload files to AWS S3 bucket.
        :param file_paths_and_s3_object_names: dict with local file paths as keys and S3 object names as values.
            if values are None or "" the file name will be used as S3 object name.
        :param bucket_name: s3 bucket name.
        :return: None
        """
        if self.s3 is None:
            try:
                self.connect_aws_s3()
            except Exception as e:
                logger.error(f"Error connecting to AWS S3: {e}")
                raise ValueError("AWS S3 connection not established. Please check credentials.")

        for file_path, s3_object_name in file_paths_and_s3_object_names.items():
            logger.info(f"Uploading {file_path} to S3 bucket {bucket_name} as {s3_object_name}...")
            print(f"Uploading {file_path} to S3 bucket {bucket_name} as {s3_object_name}...")
            if not os.path.isfile(file_path):
                logger.error(f"File {file_path} does not exist.")
                raise ValueError(f"File {file_path} does not exist.")

            if s3_object_name in (None, ""):
                s3_object_name = os.path.basename(file_path)

            self.s3.upload_file(
                Filename=file_path,
                Bucket=bucket_name,
                Key=s3_object_name
            )

    def check_files_on_s3(self)->None:
        """
        Check if the required files are present on S3.
        :return:
        """
        if self.s3 is None:
            try:
                self.connect_aws_s3()
            except Exception as e:
                logger.error(f"Error connecting to AWS S3: {e}")
                raise ValueError("AWS S3 connection not established. Please check credentials.")

        for _, s3_object_name in self.file_paths_and_s3_object_names.items():
            logger.info(f"Checking existence of {s3_object_name} in S3 bucket {self.bucket_name}...")
            print(f"Checking existence of {s3_object_name} in S3 bucket {self.bucket_name}...")
            try:
                self.s3.head_object(Bucket=self.bucket_name, Key=s3_object_name)
                logger.info(f"File {s3_object_name} exists in S3 bucket {self.bucket_name}.")
            except Exception as e:
                logger.error(f"File {s3_object_name} does not exist in S3 bucket {self.bucket_name}: {e}")
                raise ValueError(f"File {s3_object_name} does not exist in S3 bucket {self.bucket_name}.")

    @staticmethod
    def get_file_from_s3(s3:BaseClient,
                         bucket_name:str,
                         s3_object_name:str
                         )->bytes:
        """
        Get a file from S3 and return its content as originally stored format.
        :param s3: boto3 S3 client.
        :param bucket_name: S3 bucket name.
        :param s3_object_name: S3 object name (file key).
        :return: file content as original object (.parquet and .pkl supported for now).
        """
        response = s3.get_object(
            Bucket=bucket_name,
            Key=s3_object_name
        )
        file_content = response['Body'].read()
        if 'pkl' in s3_object_name.split('.')[-1]:
            file_content = pickle.loads(file_content)
        elif 'parquet' in s3_object_name.split('.')[-1]:
            file_content = pd.read_parquet(io.BytesIO(file_content))
        else:
            logger.error(f"Unsupported file extension for {s3_object_name}.")
            raise ValueError(f"Unsupported file extension for {s3_object_name}.")

        return file_content

    def get_files_from_s3(self)->None:
        """
        Download required files from S3.
        :return:
        """
        if self.s3 is None:
            try:
                self.connect_aws_s3()
            except Exception as e:
                logger.error(f"Error connecting to AWS S3: {e}")
                raise ValueError("AWS S3 connection not established. Please check credentials.")

        if self.file_paths_and_s3_object_names is not None and self.s3_files_downloaded is None:
            self.s3_files_downloaded = {
                Path(file_name).stem:None
                for file_name in self.file_paths_and_s3_object_names.keys()
            }
            for _, s3_object_name in self.file_paths_and_s3_object_names.items():
                logger.info(f"Downloading {s3_object_name} from S3 bucket {self.bucket_name}")
                print(f"Downloading {s3_object_name} from S3 bucket {self.bucket_name}")

                response = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_object_name
                )
                file_content = response['Body'].read()

                if 'pkl' in s3_object_name.split('.')[-1]:
                    file_content = pickle.loads(file_content)
                elif 'parquet' in s3_object_name.split('.')[-1]:
                    file_content = pd.read_parquet(io.BytesIO(file_content))
                else:
                    logger.error(f"Unsupported file extension for {s3_object_name}.")
                    raise ValueError(f"Unsupported file extension for {s3_object_name}.")

                self.s3_files_downloaded[Path(s3_object_name).stem] = file_content

    def update_wrds_data(self,
                         wrds_request: str,
                         date_cols: List[str],
                         saving_config: dict,
                         return_bool: bool = False
                         )->dict:
        """
        Update the wrds_gross_query.parquet file with the latest data from WRDS.
        :return:
        """
        if self.wrds_db is None:
            self.connect_wrds()
        if self.s3_files_downloaded is None or 'wrds_gross_query' not in self.s3_files_downloaded:
            logger.error("wrds_gross_query data not downloaded from S3.")
            raise ValueError("wrds_gross_query data not downloaded from S3.")
        current_wrds_gross_query = self.s3_files_downloaded['wrds_gross_query']
        current_wrds_gross_query = current_wrds_gross_query.sort_values('date', ascending=True).reset_index(drop=True)
        latest_date_in_current = pd.to_datetime(current_wrds_gross_query['date'].max())

        # Request new data from WRDS
        starting_date_for_update = str(pd.to_datetime(latest_date_in_current) + pd.Timedelta(days=1))

        # The below function will save in self.wrds_universe the new universe
        self.fetch_wrds_historical_universe(wrds_request=wrds_request,
                                            date_cols=date_cols,
                                            starting_date=starting_date_for_update,
                                            saving_config=saving_config,
                                            save_tickers_across_dates=False,
                                            save_dates=False,
                                            return_bool=return_bool,
                                            crsp_to_ib_mapping_tickers_from_cloud=True,
                                            update_crsp_to_ib_mapping_tickers=True,
                                            save_ib_tickers_to_cloud=True
                                            )
        # Now we have in self.wrds_gross_query the new data from starting_date_for_update to today
        # In self.wrds_universe we have the new universe from starting_date_for_update to today
        # In self.tickers_across_dates we have the tickers across dates from starting_date_for_update to today
        # In self.crsp_to_ib_mapping_tickers we have the updated mapping (ancient + new tickers if any)
        # In self.dates we have the new dates from starting_date_for_update to today
        # We now need to concatenate the current data with the new data
        # Check that we only concatenate the new data by looking if the first date of our new query is
        # after the latest_date_in_current
        first_date_new_query = pd.to_datetime(self.wrds_gross_query['date'].min())
        if pd.isna(first_date_new_query) or first_date_new_query <= latest_date_in_current:
            logger.info("The new WRDS query does not contain new data beyond the current latest date.")
            # Nothing to update
            return {}

        if self.wrds_gross_query.empty:
            logger.info("The new WRDS query returned no new data.")
            return {}

        # Concatenate the dataframes
        # First sort both dataframes by date ascending
        updated_wrds_gross_query = pd.concat([current_wrds_gross_query,self.wrds_gross_query],
                                             axis=0,
                                             ignore_index=True)
        # Do the same with self.wrds_universe
        current_wrds_universe = self.s3_files_downloaded['wrds_universe']
        current_wrds_universe = current_wrds_universe.sort_index(ascending=True)
        # Check the dates
        first_date_new_universe = self.wrds_universe.index.min()
        last_date_current_universe = current_wrds_universe.index.max()

        # Send phone alerts to monitor when away from computer
        msg = "last date old wrds universe " + str(last_date_current_universe)
        logger.info(msg)

        msg = "first date new wrds universe " + str(first_date_new_universe)
        logger.info(msg)

        if first_date_new_universe <= last_date_current_universe:
            logger.info("The new WRDS universe does not contain new data beyond the current latest date.")
            return {}
        updated_wrds_universe = pd.concat([current_wrds_universe,self.wrds_universe],
                                         axis=0,
                                         ignore_index=False)
        current_tickers_across_dates = self.s3_files_downloaded['tickers_across_dates']
        new_tickers_across_dates = self.tickers_across_dates
        updated_tickers_across_dates = list(set(current_tickers_across_dates + new_tickers_across_dates))
        current_dates = self.s3_files_downloaded['dates']
        updated_dates = current_dates + self.dates
        updated_crsp_to_ib_mapping_tickers = self.crsp_to_ib_mapping_tickers

        return {"data/wrds_gross_query.parquet":updated_wrds_gross_query,
                "data/wrds_universe.parquet":updated_wrds_universe,
                "data/tickers_across_dates.pkl":updated_tickers_across_dates,
                "data/dates.pkl":updated_dates,
                "data/crsp_to_ib_mapping_tickers.pkl":updated_crsp_to_ib_mapping_tickers
                }

    def update_ib_data(self):
        """
        Update the IB historical prices data with the latest data from IB.
        :return:
        """
        if self.ib is None or not self.ib.isConnected():
            self.connect_ib()
        if self.s3_files_downloaded is None or 'ib_historical_prices' not in self.s3_files_downloaded or 'ib_tickers' not in self.s3_files_downloaded:
            logger.error("IB data not downloaded from S3.")
            raise ValueError("IB data not downloaded from S3.")

        current_ib_historical_prices = self.s3_files_downloaded['ib_historical_prices']
        current_ib_tickers = self.s3_files_downloaded['ib_tickers']
        last_date_in_current = current_ib_historical_prices.index.max()
        current_date = pd.to_datetime('today').normalize()

        # Last US business day
        last_trading_day = current_date - BDay(1) if current_date.weekday() >= 5 else current_date

        delta_days = (last_trading_day - last_date_in_current).days

        if delta_days <= 0:
            logger.info("IB historical prices data is already up to date.")
            return
        past_period = f"{delta_days+2} D" # +2 as security
        # Fetch new data from IB
        new_ib_prices = self.fetch_ib_historical_prices(end_date='',
                                                        past_period=past_period,
                                                        frequency='1 day',
                                                        data_prices='ADJUSTED_LAST',
                                                        use_rth=True,
                                                        format_date=1,
                                                        load_from_cloud=True,
                                                        updating_procedure=True,
                                                        save_prices=False,
                                                        return_bool=True
                                                        )
        # As we might have retrieved 'too much' days, crop only the new
        flg = new_ib_prices.index > last_date_in_current
        new_ib_prices = new_ib_prices.loc[flg,:]

        # Check if there is no negative prices, if any put nan
        mask = new_ib_prices < 0
        new_ib_prices = new_ib_prices.mask(mask, np.nan)
        # Now we have to "merge" the current_ib_historical_prices with new_ib_prices. We must to it with care
        # because: in the new_ib_prices we only have the tickers that were in the universe at the last date,
        # so we cannot just do a concat. We have to update only the relevant tickers. If there are new tickers
        # in new_ib_prices that were not in current_ib_historical_prices, we have to add a new column for them
        # and make their previous values NaN. For the others tickers (not new and not in the last universe) we
        # just fill their values with NaN for the new dates.
        updated_ib_historical_prices = current_ib_historical_prices.copy()
        # add new rows for new dates. Before, check that the first new date is after the last date in current
        first_new_date = new_ib_prices.index.min()

        # Send phone alerts to monitor when away from computer
        msg = "last date old ib prices " + str(last_date_in_current)
        logger.info(msg)

        msg = "first date new ib prices " + str(first_new_date)
        logger.info(msg)

        if first_new_date <= last_date_in_current:
            logger.info("The new IB prices do not contain new data beyond the current latest date.")
            return
        # add new dates
        updated_ib_historical_prices = pd.concat([updated_ib_historical_prices,
                                                 pd.DataFrame(index=new_ib_prices.index,
                                                              columns=updated_ib_historical_prices.columns,
                                                              data=np.nan)
                                                  ],
                                                axis=0,
                                                ignore_index=False)
        # Now update the relevant tickers
        for ticker in new_ib_prices.columns:
            if ticker in updated_ib_historical_prices.columns:
                # update existing column
                updated_ib_historical_prices.loc[new_ib_prices.index, ticker] = new_ib_prices[ticker]
            else:
                # add new column
                updated_ib_historical_prices[ticker] = np.nan
                updated_ib_historical_prices.loc[new_ib_prices.index, ticker] = new_ib_prices[ticker]

        # re sort for double safety
        updated_ib_historical_prices.sort_index(inplace=True, ascending=True)

        # Finally, update the ib_tickers list
        updated_ib_tickers = list(set(current_ib_tickers + new_ib_prices.columns.tolist()))

        return {"data/ib_historical_prices.parquet":updated_ib_historical_prices,
                "data/ib_tickers.pkl":updated_ib_tickers
                }

    def update_portfolio_value(self):
        old_pv = self.s3_files_downloaded["portfolio_value_historical"]
        old_pv = old_pv.sort_index(ascending=True)
        new_pv = self.fetch_portfolio_value(saving_path_pv=None, return_bool=True)

        # Check the date is new
        last_date_old = old_pv.index[-1]
        first_date_new = new_pv.index[-1] #-1 or 0 is the same as it only contains 1 value
        if first_date_new<=last_date_old:
            logger.info("No new value for portfolio_value")
            print("No new value for portfolio_value")
            return
        else:
            # Include all dates even non-business days, but we would have to be careful
            # when manipulating with others df -> do a merge
            missing_dates = pd.date_range(start=last_date_old,
                                          end=first_date_new,
                                          inclusive="neither")
            if missing_dates.empty:
                # just concatenate
                updated_pv = pd.concat([old_pv,new_pv], axis=0)
                updated_pv = updated_pv.sort_index(ascending=True)
            else:
                df_missing = pd.DataFrame(index=missing_dates,
                                          data=np.nan,
                                          columns=["portfolio_value"])
                updated_pv = pd.concat([old_pv,df_missing,new_pv], axis=0)
                updated_pv = updated_pv.sort_index(ascending=True)

        updated_pv = updated_pv[~updated_pv.index.duplicated(keep="last")]

        return {"paper_trading/portfolio_value_historical.parquet":updated_pv}


    def update_data(self,
                    wrds_request:str,
                    date_cols: List[str],
                    saving_config: dict,
                    return_bool: bool = False,
                    coverage_threshold: float = 0.9,
                    max_tries: int = 3
                    )->None:
        """
        Update the WRDS and IB data to the latest available.
        More precisely, it aims at updating the files:
        - wrds_gross_query.parquet
        - wrds_universe.parquet
        - ib_historical_prices.parquet
        - tickers_across_dates.pkl
        - dates.pkl
        - crsp_to_ib_mapping_tickers.pkl
        - ib_tickers.pkl
        - coverage_threshold: nb of not nan prices at the moment we update the data compared to the
            previous date. If we recover less than coverage_threshold for this update, we will redo it
            max_tries times.
        - max_tries: see above
        It will download these current files from s3, then update them with the latest data from WRDS and IB,
        and finally re-upload the updated files to s3.
        :return: None
        """
        # Step 0.1: Check data types
        if not isinstance(wrds_request, str):
            logger.error("wrds_request must be a string.")
            raise ValueError("wrds_request must be a string containing the SQL query.")
        if not isinstance(date_cols, list):
            logger.error("date_cols must be a list of strings.")
            raise ValueError("date_cols must be a list of strings.")
        for col in date_cols:
            if not isinstance(col, str):
                logger.error("All elements in date_cols must be strings.")
                raise ValueError("All elements in date_cols must be strings.")
        if not isinstance(saving_config, dict):
            logger.error("saving_config must be a dictionary.")
            raise ValueError("saving_config must be a dictionary.")
        if not isinstance(return_bool, bool):
            logger.error("return_bool must be a boolean.")
            raise ValueError("return_bool must be a boolean.")

        logger.info("Updating data.")

        # Step 0.2: Connect to AWS S3 if not already connected
        if self.s3 is None:
            try:
                self.connect_aws_s3()
            except Exception as e:
                logger.error(f"Error connecting to AWS S3: {e}")
                raise ValueError("AWS S3 connection not established. Please check credentials.")

        # Step 1: Check if the current files are on s3
        self.check_files_on_s3()

        # Step 2: Download the current files from s3
        self.get_files_from_s3()

        # Step 3: update wrds files
        updated_wrds_objects = self.update_wrds_data(wrds_request=wrds_request,
                                                     date_cols=date_cols,
                                                     saving_config=saving_config,
                                                     return_bool=return_bool
                                                     )
        # Step 3.1: upload updated wrds files to s3
        if updated_wrds_objects:
            logger.info("New WRDS data to update.")
            s3Utils.replace_existing_files_in_s3(s3=self.s3,
                                              bucket_name=self.bucket_name,
                                              files_dct=updated_wrds_objects
                                              )
            logger.info("Uploaded updated WRDS files to S3.")
        else:
            logger.info("No new WRDS data to update.")

        # Wait a bit to ensure the files are available on s3
        time.sleep(10)

        # Step 4.0 update portfolio_value
        # For the orders df we also need to store the historical portfolio value
        updated_pv = self.update_portfolio_value()
        if updated_pv:
            logger.info("New portfolio value data to update.")
            s3Utils.replace_existing_files_in_s3(s3=self.s3,
                                                 bucket_name=self.bucket_name,
                                                 files_dct=updated_pv
                                                 )
            logger.info("Uploaded updated portfolio value file to S3.")
        else:
            logger.info("No new portfolio value data to update.")

        # Step 4: update ib files
        updated_ib_objects = self.update_ib_data()
        if not updated_ib_objects:
            logger.info("No new IB data to update.")
            print("No new IB data to update.")
            return

        # Checking that the coverage we get for prices is correct, if not redo the update
        notna = updated_ib_objects["data/ib_historical_prices.parquet"].notna().sum(axis=1)
        notna_prevdate = notna.iloc[-2]
        notna_current = notna.iloc[-1]
        tries = 0

        while (notna_current/notna_prevdate<coverage_threshold) and (tries<=max_tries):
            updated_ib_objects = self.update_ib_data()
            tries+=1
            logger.info(f"We got poor coverage of non nan prices while updating. Redoing the update for the {tries} time.")
            print(f"We got poor coverage of non nan prices while updating. Redoing the update for the {tries} time.")

        # Step 4.1: upload updated ib files to s3
        s3Utils.replace_existing_files_in_s3(s3=self.s3,
                                          bucket_name=self.bucket_name,
                                          files_dct=updated_ib_objects
                                          )

        self.logout_wrds()
        self.logout_ib()
        # s3 automatically closes as it uses short live https requests

        return





