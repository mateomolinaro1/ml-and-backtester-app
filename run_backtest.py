import json
import os
import pandas as pd
from ml_and_backtester_app.backtester import data, signal_utilities, strategies, portfolio, backtest_pandas, analysis
from ml_and_backtester_app.utils.config import Config

def run():
    config = Config()
    from ml_and_backtester_app.data.data_manager import DataManager as AWSDataManager
    aws_manager = AWSDataManager(config=config)
    
    # 1. Charger la config envoyée par le dashboard
    config_file = "config/backtest_config.json"
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found.")
        return

    with open(config_file, "r") as f:
        bt_params = json.load(f)

    # --- RÉCUPÉRATION DES PARAMÈTRES DYNAMIQUES ---
    chosen_ratio = bt_params.get("ratio_name", "Momentum")
    start_date = bt_params.get("start_date", "2010-01-01")
    rebal = bt_params.get("nb_period_to_exclude", 22)
    costs = bt_params.get("transaction_costs", 10)
    is_ascending = bt_params.get("ascending", False) 

    # --- LE MINIMUM POUR LA FRÉQUENCE ---
    if chosen_ratio == "Momentum":
        freq, ann, actual_rebal = "d", 252, rebal
    else:
        freq, ann, actual_rebal = "m", 12, 1 # 1 mois si on est en fondamental

    print(f"Starting Backtest: {chosen_ratio} | Direction: {'Buy Low' if is_ascending else 'Buy High'}")

# 1. Sélection du fichier S3
    if chosen_ratio == "Momentum":
        s3_file = "data/wrds_gross_query.parquet"
    else:
        s3_file = "data/wrds_funda_gross_query.parquet"

    # 2. Récupération et "Patch" des données
    # On charge les données manuellement pour corriger les noms avant le DataManager
    print(f"Loading and normalizing: {s3_file}...")
    raw_df = aws_manager.aws.s3.load(key=s3_file)
    
    # Remplacements magiques pour que le backtester soit content
    rename_map = {
        "public_date": "date",  # Le fichier funda utilise public_date
        "ret_crsp": "ret"       # Le fichier funda utilise ret_crsp
    }
# 2. Récupération et "Patch" des données    
    rename_map = {"public_date": "date", "ret_crsp": "ret"}
    raw_df = raw_df.rename(columns={k: v for k, v in rename_map.items() if k in raw_df.columns})
    raw_df = raw_df.dropna(subset=['date', 'permno'])
    raw_df = raw_df.drop_duplicates(subset=['date', 'permno'], keep='last')

    # --- ÉTAPE CRUCIALE : On filtre la date TOUT DE SUITE ---
    raw_df = raw_df[raw_df['date'] >= start_date]

    class LocalSource:
        def __init__(self, df): self.df = df
        def fetch_data(self): return self.df

    dm = data.DataManager(data_source=LocalSource(raw_df), rebase_prices=False, already_returns=True)
    dm.get_data(format_date="%Y-%m-%d", crop_lookback_period=0)

    # On définit notre base de temps officielle
    returns_cut = dm.returns 

    # 3. Setup de la fonction de Signal
    if chosen_ratio == "Momentum":
        base_func = signal_utilities.Momentum.rolling_momentum
        func = lambda df, **kwargs: base_func(df, **kwargs) * (-1 if is_ascending else 1)
        inputs = {"df": dm.cleaned_data, "nb_period": 252, "nb_period_to_exclude": rebal}
    else:
        print(f"Pivoting factor: {chosen_ratio}...")
        signal_wide = raw_df.pivot(index="date", columns="permno", values=chosen_ratio)
        
        # --- SÉCURITÉ : On aligne le signal sur les rendements ---
        # On ne garde que les dates et les tickers qui existent dans les deux
        signal_wide = signal_wide.reindex(index=returns_cut.index, columns=returns_cut.columns)
        
        if is_ascending:
            signal_wide = -signal_wide
        
        func = lambda df, wide_data: wide_data
        inputs = {"df": dm.cleaned_data, "wide_data": signal_wide}

    # 4. Initialisation de la Stratégie
    strategy = strategies.CrossSectionalPercentiles(
        returns=returns_cut, 
        signal_function=func,
        signal_function_inputs=inputs
    )
    strategy.compute_signals_values()

    
    p_low, p_high = bt_params.get("percentiles", [10, 90])
    # ON APPLIQUE LA DIRECTION ICI :
    strategy.compute_signals(percentiles_portfolios=(p_low, p_high))

    # 5. Benchmark & Portefeuilles
    bench = strategies.BuyAndHold(returns=returns_cut)
    bench.compute_signals_values()
    bench.compute_signals()

    ptf = portfolio.EqualWeightingScheme(
        returns=returns_cut,
        signals=strategy.signals,
        rebal_periods=actual_rebal,
        portfolio_type="long_only"
    )
    ptf.compute_weights()
    ptf.rebalance_portfolio()

    ptf_bench = portfolio.EqualWeightingScheme(
        returns=returns_cut,
        signals=bench.signals,
        rebal_periods=actual_rebal,
        portfolio_type="long_only"
    )
    ptf_bench.compute_weights()
    ptf_bench.rebalance_portfolio()

    # 6. Exécution du Backtest
    backtester = backtest_pandas.Backtest(
        returns=returns_cut,
        weights=ptf.rebalanced_weights.shift(1),
        turnover=ptf.turnover,
        transaction_costs=costs,
        strategy_name=bt_params.get("strategy_name", chosen_ratio)
    )
    backtester.run_backtest()

    backtester_bench = backtest_pandas.Backtest(
        returns=returns_cut,
        weights=ptf_bench.rebalanced_weights.shift(1),
        turnover=ptf_bench.turnover,
        transaction_costs=costs,
        strategy_name="BUY_AND_HOLD"
    )
    backtester_bench.run_backtest()

    # 7. Analyse des Performances
    analyzer = analysis.PerformanceAnalyser(
        portfolio_returns=backtester.cropped_portfolio_net_returns,
        bench_returns=backtester_bench.portfolio_net_returns.loc[backtester.start_date:, :],
        freq=freq,
        percentiles=str((p_low, p_high)),
        industries="Cross_Industries",
        rebal_freq=f"{actual_rebal} {freq}"
    )
    analyzer.compute_metrics()

    # 8. Export S3
    s3_path = f"{config.outputs_path}/backtest/figures/cumulative_performance.png"
    local_img = "tmp_cum_perf.png"
    
    print("Generating performance plot...")
    analyzer.plot_cumulative_performance(saving_path=local_img)
    
    print(f"Uploading to S3: {s3_path}...")
    aws_manager.aws.s3.upload(src=local_img, key=s3_path)
    
    if os.path.exists(local_img):
        os.remove(local_img)
        
    print("Done! Backtest complete.")

if __name__ == "__main__":
    run()