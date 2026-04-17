import json
import os
import sys
import pandas as pd
import numpy as np
from ml_and_backtester_app.backtester import data, signal_utilities, strategies, portfolio, backtest_pandas, analysis
from ml_and_backtester_app.utils.config import Config


def run():
    config = Config()

    from ml_and_backtester_app.data.data_manager import DataManager as AWSDataManager

    aws_manager = AWSDataManager(config=config)
    
    # 1. Charger la config envoyée par le dashboard
    # On s'assure que le fichier existe
    config_file = "config/backtest_config.json"
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found.")
        return

    with open(config_file, "r") as f:
        bt_params = json.load(f)

    print(f"Starting Backtest: {bt_params.get('strategy_name', 'CSMOM')}...")

    # 2. Récupération des données
    ds = data.AmazonS3(
        bucket_name=config.aws_bucket_name, 
        s3_object_name="data/wrds_gross_query.parquet"
    )
    dm = data.DataManager(
        data_source=ds, 
        rebase_prices=False, 
        already_returns=True
    )
    dm.get_data(format_date="%Y-%m-%d", crop_lookback_period=0)

    # 3. Setup de la Stratégie (Momentum)
    strategy = strategies.CrossSectionalPercentiles(
        returns=dm.returns,
        signal_function=signal_utilities.Momentum.rolling_momentum,
        signal_function_inputs={
            "df": dm.cleaned_data,
            "nb_period": bt_params.get("nb_period", 252),
            "nb_period_to_exclude": bt_params.get("nb_period_to_exclude", 22),
            "exclude_last_period": True
        }
    )
    strategy.compute_signals_values()
    
    # On récupère les percentiles du JSON
    p_low, p_high = bt_params.get("percentiles", [10, 90])
    strategy.compute_signals(percentiles_portfolios=(p_low, p_high))

    # 4. Benchmark (Buy & Hold)
    bench = strategies.BuyAndHold(returns=dm.returns)
    bench.compute_signals_values()
    bench.compute_signals()

    # 5. Construction du Portefeuille (Equal Weighting)
    ptf = portfolio.EqualWeightingScheme(
        returns=dm.returns,
        signals=strategy.signals,
        rebal_periods=22,
        portfolio_type="long_only"
    )
    ptf.compute_weights()
    ptf.rebalance_portfolio()

    ptf_bench = portfolio.EqualWeightingScheme(
        returns=dm.returns,
        signals=bench.signals,
        rebal_periods=22,
        portfolio_type="long_only"
    )
    ptf_bench.compute_weights()
    ptf_bench.rebalance_portfolio()

    # 6. Exécution du Backtest
    costs = bt_params.get("transaction_costs", 10)
    
    backtester = backtest_pandas.Backtest(
        returns=dm.returns,
        weights=ptf.rebalanced_weights.shift(1),
        turnover=ptf.turnover,
        transaction_costs=costs,
        strategy_name=bt_params.get("strategy_name", "CSMOM")
    )
    backtester.run_backtest()

    backtester_bench = backtest_pandas.Backtest(
        returns=dm.returns,
        weights=ptf_bench.rebalanced_weights.shift(1),
        turnover=ptf_bench.turnover,
        transaction_costs=costs,
        strategy_name="BUY_AND_HOLD"
    )
    backtester_bench.run_backtest()

    # 7. Analyse des Performances (L'ANALYZER EST LÀ !)
    analyzer = analysis.PerformanceAnalyser(
        portfolio_returns=backtester.cropped_portfolio_net_returns,
        bench_returns=backtester_bench.portfolio_net_returns.loc[backtester.start_date:, :],
        freq="d",
        percentiles=str((p_low, p_high)),
        industries="Cross_Industries",
        rebal_freq="22 d"
    )
    analyzer.compute_metrics()

    # 8. Export S3
    s3_path = f"{config.outputs_path}/backtest/figures/cumulative_performance.png"
    local_img = "tmp_cum_perf.png"
    
    # 4. On crée l'image
    print("Generating performance plot...")
    analyzer.plot_cumulative_performance(saving_path=local_img)
    
    # 5. On utilise l'AWSDataManager pour l'upload
    print(f"Uploading to S3: {s3_path}...")
    aws_manager.aws.s3.upload(src=local_img, key=s3_path)
    
    # 6. Ménage
    if os.path.exists(local_img):
        os.remove(local_img)
        
    print("Done! Backtest complete.")

if __name__ == "__main__":
    run()