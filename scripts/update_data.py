from ml_and_backtester_app.data.data_handler import DataHandler
from dotenv import load_dotenv
import os
import logging
import sys

def main(
    data_path,
    bucket_name,
    wrds_request,
    date_cols,
    saving_config,
    return_bool,
):
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    dh = DataHandler(
        data_path=data_path,
        wrds_username=os.getenv("WRDS_USERNAME"),
        wrds_password=os.getenv("WRDS_PASSWORD"),
        ib_host=os.getenv("IB_HOST"),
        ib_port=int(os.getenv("IB_PORT")),
        ib_client_id=int(os.getenv("IB_CLIENT_ID")),
        bucket_name=bucket_name,
    )

    dh.update_data(
        wrds_request=wrds_request,
        date_cols=date_cols,
        saving_config=saving_config,
        return_bool=return_bool
    )

    logger.info("Data update completed.")

if __name__ == "__main__":
    from config.config_update_data import CONFIG
    main(**CONFIG)

from ml_and_backtester_app.utils.s3_utils import s3Utils
import boto3
from dotenv import load_dotenv
import os
load_dotenv()
s3u = s3Utils()
res = s3u.pull_parquet_files_from_s3(
    paths=[
        "ml-and-backtester-app/data/wrds_gross_query.parquet",
        "ml-and-backtester-app/data/wrds_universe.parquet",
        "ml-and-backtester-app/data/ib_historical_prices.parquet"
    ]
)
notna = res["ib_historical_prices"].notna().sum(axis=1)
# notna_prevdate = notna.iloc[-2]
# notna_current = notna.iloc[-1]
# notna_current/notna_prevdate
#
# new_ib_prices = res["ib_historical_prices"].loc[res["ib_historical_prices"].index<"2026-01-06",:]
# updated_ib_objects = {
#     "data/ib_historical_prices.parquet": new_ib_prices
# }
# s3Utils.replace_existing_files_in_s3(s3=boto3.client("s3"),
#                                      bucket_name=os.getenv("BUCKET_NAME"),
#                                      files_dct=updated_ib_objects
#                                      )
#
# paths = [
#     "data/wrds_gross_query.parquet",
#     "data/wrds_universe.parquet",
#     "data/tickers_across_dates.pkl",
#     "data/dates.pkl",
#     "data/crsp_to_ib_mapping_tickers.pkl"
# ]