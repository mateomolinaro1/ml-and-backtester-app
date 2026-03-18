from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager
from dotenv import load_dotenv
# from ml_and_backtester_app.fmp.fmp import FactorMimickingPortfolio
# from ml_and_backtester_app.forecasting.features_engineering import FeaturesEngineering
# from ml_and_backtester_app.forecasting.schemes.expanding import ExpandingWindowScheme
# from ml_and_backtester_app.dynamic_allocation.dynamic_allocation import DynamicAllocation
# from ml_and_backtester_app.analytics.analytics import AnalyticsFMP, AnalyticsForecasting, \
#     AnalyticsDynamicAllocation
import sys
import logging
load_dotenv()

logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
config = Config()

# Data
data_manager = DataManager(config=config)

# FMP
fmp = FactorMimickingPortfolio(
    config=config,
    data=data_manager,
    market_returns=None,
    rf=None
)
fmp.build_macro_portfolios()