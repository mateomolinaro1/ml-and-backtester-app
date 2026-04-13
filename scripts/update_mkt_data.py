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
        ib_host=config.ib_host,
        ib_port=config.ib_port,
        ib_client_id=config.ib_client_id,
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
    from config.config_update_data import WRDS_REQUEST, DATE_COLS, SAVING_CONFIG_UNIVERSE, RETURN_BOOL_UNIVERSE
    from ml_and_backtester_app.utils.config import Config
    from dotenv import load_dotenv
    load_dotenv()
    config = Config()

    main(
        data_path=config.ROOT_DIR / "data",
        bucket_name=config.aws_bucket_name,
        wrds_request=WRDS_REQUEST,
        date_cols=DATE_COLS,
        saving_config=SAVING_CONFIG_UNIVERSE,
        return_bool=RETURN_BOOL_UNIVERSE
    )
