from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager
from dotenv import load_dotenv
from ml_and_backtester_app.fmp.fmp import FactorMimickingPortfolio
from ml_and_backtester_app.machine_learning.features_engineering import FeaturesEngineering
from ml_and_backtester_app.machine_learning.schemes.expanding import ExpandingWindowScheme
from ml_and_backtester_app.machine_learning.schemes.rolling import RollingWindowScheme
from ml_and_backtester_app.dynamic_allocation.dynamic_allocation import DynamicAllocation
from ml_and_backtester_app.analytics.analytics import AnalyticsFMP, AnalyticsForecasting, \
    AnalyticsDynamicAllocation
import sys
import logging
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

config = Config()
# print("--- DEBUG CONFIG ---")
# print(f"Attributs disponibles : {vars(config).keys()}")
# print(f"Valeur trouvée pour method : {getattr(config, 'estimation_method', 'RIEN TROUVÉ')}")


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

# Analytics FMP
analytics_fmp = AnalyticsFMP(
    config=config,
    dm=data_manager,
    fmp=fmp
)
analytics_fmp.get_analytics()

# Feature Engineering
fe = FeaturesEngineering(config=config, data=data_manager)
fe.get_features()

# --- CHOIX DU SCHÉMA (Basé sur ta config) ---
# On vérifie si tu as mis "rolling" dans ton JSON, sinon par défaut on fait "expanding"
estimation_method = getattr(config, "estimation_method", "expanding")

if estimation_method == "rolling":
    logger.info("### MODE SÉLECTIONNÉ : ROLLING WINDOW ###")
    scheme = RollingWindowScheme(
        config=config,
        dm=data_manager,
        x=fe.x,
        y=fe.y,
        forecast_horizon=config.forecast_horizon,
        validation_window=config.validation_window,
        min_nb_periods_required=config.min_nb_periods_required
    )
else:
    logger.info("### MODE SÉLECTIONNÉ : EXPANDING WINDOW ###")
    scheme = ExpandingWindowScheme(
        config=config,
        dm=data_manager,
        x=fe.x,
        y=fe.y,
        forecast_horizon=config.forecast_horizon,
        validation_window=config.validation_window,
        min_nb_periods_required=config.min_nb_periods_required
    )

# On lance l'entraînement (le code est le même pour les deux !)
scheme.run(
    models=config.models,
    hyperparams_grid=config.hyperparams_grid
)

# Analytics Forecasting
analytics_forecasting = AnalyticsForecasting(
    config=config,
    dm=data_manager,
    exp_window=scheme
)
analytics_forecasting.get_analytics()

# Dynamic Allocation
dynamic_alloc = DynamicAllocation(
    config=config,
    predictions=scheme.oos_predictions,
    long_leg_fmp=fmp.positive_betas_fmp_returns,
    short_leg_fmp=fmp.negative_betas_fmp_returns,
    benchmark_ptf=fmp.benchmark_returns
)
dynamic_alloc.run_backtest()

# Analytics Dynamic Allocation
analytics_dynamic_alloc = AnalyticsDynamicAllocation(
    config=config,
    dm=data_manager,
    dynamic_alloc=dynamic_alloc,
    fmp=fmp
)
analytics_dynamic_alloc.get_analytics()
