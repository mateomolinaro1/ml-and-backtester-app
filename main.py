from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager
from dotenv import load_dotenv
from ml_and_backtester_app.fmp.fmp import FactorMimickingPortfolio
from ml_and_backtester_app.machine_learning.features_engineering import FeaturesEngineering
from ml_and_backtester_app.machine_learning.schemes.expanding import ExpandingWindowScheme
from ml_and_backtester_app.dynamic_allocation.dynamic_allocation import DynamicAllocation
from ml_and_backtester_app.analytics.analytics import AnalyticsFMP, AnalyticsForecasting, \
    AnalyticsDynamicAllocation
import sys
import logging
load_dotenv()

logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)
config = Config()

# Data
logger.info("Loading data...")
data_manager = DataManager(config=config)

# FMP
logger.info("Building Factor Mimicking Portfolios...")
fmp = FactorMimickingPortfolio(
    config=config,
    data=data_manager,
    market_returns=None,
    rf=None
)
fmp.build_macro_portfolios()

# Analytics FMP
logger.info("Getting analytics on FMP...")
analytics_fmp = AnalyticsFMP(
    config=config,
    dm=data_manager,
    fmp=fmp
)
analytics_fmp.get_analytics()

# Feature Engineering
logger.info("Getting features...")
fe = FeaturesEngineering(config=config, data=data_manager)
fe.get_features()

# Expanding Window Scheme
logger.info("Running Expanding Window Scheme...")
exp_window = ExpandingWindowScheme(
    config=config,
    dm=data_manager,
    x=fe.x,
    y=fe.y,
    forecast_horizon=config.forecast_horizon,
    validation_window=config.validation_window,
    min_nb_periods_required=config.min_nb_periods_required
)
exp_window.run(
    models=config.models,
    hyperparams_grid=config.hyperparams_grid
)

# Analytics Forecasting
logger.info("Getting analytics on forecasting...")
analytics_forecasting = AnalyticsForecasting(
    config=config,
    dm=data_manager,
    exp_window=exp_window
)
analytics_forecasting.get_analytics()

# Dynamic Allocation
logger.info("Running Dynamic Allocation backtest...")
dynamic_alloc = DynamicAllocation(
    config=config,
    predictions=exp_window.oos_predictions,
    long_leg_fmp=fmp.positive_betas_fmp_returns,
    short_leg_fmp=fmp.negative_betas_fmp_returns,
    benchmark_ptf=fmp.benchmark_returns
)
dynamic_alloc.run_backtest()

# Analytics Dynamic Allocation
logger.info("Getting analytics on Dynamic Allocation backtest...")
analytics_dynamic_alloc = AnalyticsDynamicAllocation(
    config=config,
    dm=data_manager,
    dynamic_alloc=dynamic_alloc,
    fmp=fmp
)
analytics_dynamic_alloc.get_analytics()
