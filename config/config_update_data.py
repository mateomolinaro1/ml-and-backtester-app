from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "logger.log"

BUCKET_NAME = "systematic-trading-infra-storage"

# At least, the query must retrieve the following columns:
# ['ticker','exchcd','cusip','ncusip','comnam','permno','permco','namedt','nameendt','date']
WRDS_REQUEST = """
WITH base AS (
    SELECT
        a.ticker, a.exchcd,
        a.comnam, a.cusip, a.ncusip,
        a.permno, a.permco,
        a.namedt, a.nameendt,
        b.date, b.ret, b.prc, b.shrout, b.vol,
        ABS(b.prc) * b.shrout * 1000 AS market_cap
    FROM crsp.msenames AS a
    JOIN crsp.dsf AS b
      ON a.permno = b.permno
     AND b.date BETWEEN a.namedt AND a.nameendt
    WHERE a.exchcd IN (1, 2, 3)          -- NYSE, AMEX, NASDAQ
      AND a.shrcd IN (10, 11)            -- Common shares only
      AND b.date >= '{starting_date}'
      AND b.prc IS NOT NULL              -- ensure valid price
      AND b.vol IS NOT NULL              -- ensure valid volume
      AND b.prc != 0                     -- avoid zero-price issues
      AND ABS(b.prc) * b.vol >= 10000000 -- Dollar volume ≥ $10M
)
SELECT *
FROM (
    SELECT *,
           RANK() OVER (PARTITION BY date ORDER BY market_cap DESC) AS mcap_rank
    FROM base
) ranked
WHERE mcap_rank <= 1000
ORDER BY date, mcap_rank;
"""

DATE_COLS = [
    'namedt',
    'nameendt',
    'date'
]

SAVING_CONFIG_UNIVERSE = {}
RETURN_BOOL_UNIVERSE = False

CONFIG = {
    "data_path": DATA_PATH,
    "bucket_name": BUCKET_NAME,
    "wrds_request": WRDS_REQUEST,
    "date_cols": DATE_COLS,
    "saving_config": SAVING_CONFIG_UNIVERSE,
    "return_bool": RETURN_BOOL_UNIVERSE,
}
