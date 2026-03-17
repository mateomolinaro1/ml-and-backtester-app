import pandas as pd
from better_aws import AWS
from dotenv import load_dotenv
load_dotenv()

# 1) Create a session (boto3 will use the default credential chain unless you add other auth modes)
aws = AWS(region="eu-north-1", verbose=True)
# Optional sanity check
aws.identity(print_info=True)

# 2) Configure S3 defaults
aws.s3.config(
    bucket="ml-and-backtester-app",
    output_type="pandas",      # tabular loads -> pandas (or "polars")
    file_type="parquet",       # default tabular format for dataframe uploads without extension
    overwrite=True,
)

# 3) Upload the parquet file to S3
df = pd.read_parquet(r"C:\Users\mateo\Downloads\FRED-MD-2026-02.parquet")
aws.s3.upload(src=df, key="data/FRED-MD-2026-02.parquet")

# 4) Load it
df_loaded = aws.s3.load("data/FRED-MD-2026-02.parquet")