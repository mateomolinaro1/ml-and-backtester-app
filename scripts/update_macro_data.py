import pandas as pd
from ml_and_backtester_app.utils.s3_utils import s3Utils
from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager
from dotenv import load_dotenv
import sys
import logging
load_dotenv()

# Config
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
config = Config()

# Data
data_manager = DataManager(config=config)

df = pd.read_excel(config.ROOT_DIR/"data"/"US_montlhy_epu_index.xlsx")
df = df.iloc[0:-1, :].copy()
df = (
    df.assign(date=pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1)))
      .set_index("date")
      .drop(columns=["Year", "Month"])
)
df.rename(columns={"News_Based_Policy_Uncert_Index": "monthly_epu_index"}, inplace=True)
s3Utils.upload_df_with_index(df=df, bucket=config.aws_bucket_name, path="data/monthly_epu_index.parquet")
dff=data_manager.aws.s3.load(
    bucket=config.aws_bucket_name,
    key="data/monthly_epu_index.parquet"
)