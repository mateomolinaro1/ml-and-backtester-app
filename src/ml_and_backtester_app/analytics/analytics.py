import pandas as pd
import numpy as np
import matplotlib
from ml_and_backtester_app.dynamic_allocation.dynamic_allocation import DynamicAllocation
import matplotlib.pyplot as plt
import logging
from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager
from ml_and_backtester_app.fmp.fmp import FactorMimickingPortfolio
from ml_and_backtester_app.machine_learning.schemes.expanding import ExpandingWindowScheme
from ml_and_backtester_app.utils.s3_utils import s3Utils


matplotlib.use("Agg")  # non-GUI backend
plt.style.use("seaborn-v0_8-whitegrid")
logger = logging.getLogger(__name__)

class AnalyticsFMP:
    def __init__(self, config: Config, dm: DataManager, fmp: FactorMimickingPortfolio):
        self.config = config
        self.fmp = fmp
        self.dm = dm
        
        # Récupération dynamique de la méthode (rolling ou expanding)
        self.method = str(self.config.estimation_method).lower()
        
        # Définition des dossiers de sortie propres
        self.base_path = f"{self.config.outputs_path}/fmp/{self.method}"
        self.fig_path = f"{self.base_path}/figures"
        self.data_path = f"{self.base_path}/data"

    def _render_styled_table(self, df: pd.DataFrame, title: str, figsize: tuple = (10, 4)) -> plt.Figure:
        """Rendu de tableau stylisé 'Finance Pro'"""
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        ax.axis("off")
        table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc="center", loc="center")
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style visuel
        header_color, row_colors, edge_color = '#1f4e78', ['#f2f2f2', 'white'], '#d9d9d9'
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(edge_color)
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(header_color)
            elif col == -1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e6e6e6')
            else:
                cell.set_facecolor(row_colors[row % len(row_colors)])

        ax.set_title(title, pad=20, fontsize=12, weight='bold')
        fig.tight_layout()
        return fig

    def get_analytics(self) -> None:
        self._get_bayesian_betas_distribution()
        self._get_bayesian_betas_overtime()
        self._get_table_comparison_between_bayesian_and_non_bayesian_betas()
        self._get_summary_statistics_rsquared()
        self._get_cross_sectional_statistics_rsquared()
        self._plot_significant_proportion_bayesian_betas_overtime()
        self._plot_equity_curves_fmp()
        self._export_fmp_performance_table()

    def _get_bayesian_betas_distribution(self) -> None:
        betas = self.fmp.bayesian_betas.stack().dropna()
        if betas.empty:
            return

        start_date, end_date = betas.index[0][0], betas.index[-1][0]
        lo, hi = betas.quantile([0.01, 0.99])
        betas_clip = betas.clip(lo, hi)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(betas_clip, bins=40, density=True, alpha=0.6, color="steelblue")
        betas_clip.plot(kind="kde", color="darkred", linewidth=2, ax=ax)
        ax.set_title(f"Bayesian Betas Distribution (1–99%)\n{start_date.date()} – {end_date.date()}")
        
        path_name = f"{self.fig_path}/bayesian_betas_distribution.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _get_bayesian_betas_overtime(self) -> None:
        betas = self.fmp.bayesian_betas
        if betas.empty: 
            return
        
        avg, p10, p90 = betas.mean(axis=1), betas.quantile(0.10, axis=1), betas.quantile(0.90, axis=1)
        avg_non_na = avg.dropna()
        if avg_non_na.empty:
            return
        start_date, end_date = avg_non_na.index[[0, -1]]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(avg.index, avg, label="Cross-sectional mean", linewidth=2)
        ax.fill_between(p10.index, p10, p90, alpha=0.3, label="10–90 percentile band")
        
        ylo, yhi = np.nanpercentile(betas.values, [2, 98])
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"Bayesian Betas Dynamics\n{start_date.date()} – {end_date.date()}")
        ax.legend()

        path_name = f"{self.fig_path}/bayesian_betas_overtime.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _get_table_comparison_between_bayesian_and_non_bayesian_betas(self) -> None:
        bayesian = self.fmp.bayesian_betas.stack().dropna()
        non_bayesian = self.fmp.betas_macro.stack().dropna()
        if bayesian.empty or non_bayesian.empty:
            return

        summary_df = pd.DataFrame({
            "Bayesian Betas": [bayesian.mean(), bayesian.median(), bayesian.std(), bayesian.min(), bayesian.max()],
            "Non-Bayesian Betas": [non_bayesian.mean(), non_bayesian.median(), non_bayesian.std(), non_bayesian.min(), non_bayesian.max()]
        }, index=["Mean", "Median", "Std Dev", "Min", "Max"]).round(2)

        fig = self._render_styled_table(summary_df, "Bayesian vs Non-Bayesian Betas")
        path_name = f"{self.fig_path}/bayesian_vs_non_bayesian_betas_summary.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _get_summary_statistics_rsquared(self) -> None:
        rsquared = self.fmp.adjusted_rsquared.stack().dropna()
        if rsquared.empty: 
            return

        summary_df = pd.DataFrame({
            "Adjusted R-squared": [rsquared.mean(), rsquared.median(), rsquared.std(), rsquared.min(), rsquared.max()]
        }, index=["Mean", "Median", "Std Dev", "Min", "Max"]).round(2)

        fig = self._render_styled_table(summary_df, "Adjusted R-squared Summary")
        path_name = f"{self.fig_path}/rsquared_summary.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _get_cross_sectional_statistics_rsquared(self) -> None:
        rsq = self.fmp.adjusted_rsquared
        if rsq.empty:
            return
        avg, p10, p90 = rsq.mean(axis=1), rsq.quantile(0.10, axis=1), rsq.quantile(0.90, axis=1)
        if avg.dropna().empty: 
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(avg.index, avg, label="Mean adjusted $R^2$", linewidth=2)
        ax.fill_between(p10.index, p10, p90, alpha=0.3, label="10–90 band")
        ax.set_ylim(0, 1)
        ax.set_title("Adjusted $R^2$ Cross-Sectional Dynamics")
        
        path_name = f"{self.fig_path}/rsquared_overtime.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _plot_significant_proportion_bayesian_betas_overtime(self, alpha: float = 0.05) -> None:
        p_values = self.fmp.newey_west_pvalue
        if p_values.empty:
            return
        proportion = (p_values < alpha).sum(axis=1) / p_values.notnull().sum(axis=1).replace(0, np.nan)
        if proportion.dropna().empty:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(proportion.index, proportion, linewidth=2, color='darkgreen')
        ax.set_ylim(0, 1)
        ax.set_title(f"Proportion of Significant Bayesian Betas (α={alpha})")

        path_name = f"{self.fig_path}/proportion_significant_bayesian_betas_overtime.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _plot_equity_curves_fmp(self) -> None:
        returns = pd.concat([self.fmp.positive_betas_fmp_returns, self.fmp.negative_betas_fmp_returns, self.fmp.benchmark_returns], axis=1).dropna(how="all")
        if returns.empty:
            return
        equity = returns.cumsum()

        fig, ax = plt.subplots(figsize=(12, 6))
        for col in equity.columns:
            ax.plot(equity.index, equity[col], label=col, linewidth=2)
        ax.set_title("FMP Equity Curves")
        ax.legend()

        # Sauvegarde Image
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/fmp_equity_curves.png", fig=fig)
        plt.close(fig)

        # Sauvegarde Données
        s3Utils.upload_df_with_index(df=equity, bucket=self.config.aws_bucket_name, 
                                     path=f"{self.data_path}/fmp_equity_curves.parquet")

    def _export_fmp_performance_table(self) -> None:
        """Centralise l'export des métriques de performance"""
        perf_pos = self._compute_performance_metrics(self.fmp.positive_betas_fmp_returns, "long_only")
        perf_neg = self._compute_performance_metrics(self.fmp.negative_betas_fmp_returns, "short_only")
        perf_bench = self._compute_performance_metrics(self.fmp.benchmark_returns, "long_only")

        performance_table = pd.concat([perf_pos, perf_neg, perf_bench], axis=0)
        performance_table.index = ["LO FMP", "SO FMP", "Benchmark"]
        
        # Sauvegarde Données
        s3Utils.upload_df_with_index(df=performance_table, bucket=self.config.aws_bucket_name, 
                                     path=f"{self.data_path}/fmp_performance_table.parquet")

        # Sauvegarde Image Stylisée
        fig = self._render_styled_table(performance_table.round(2), "FMP Performance Comparison")
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/fmp_performance_summary.png", fig=fig)
        plt.close(fig)

    def _compute_performance_metrics(self, returns, portfolio_type="long_only", freq=12) -> pd.DataFrame:
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        if portfolio_type == "short_only":
            returns = -returns
        returns = returns.dropna()
        if returns.empty: 
            return pd.DataFrame()

        ann_return = (1 + returns).prod() ** (freq / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(freq)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        equity = (1 + returns).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()

        return pd.DataFrame({"Return": ann_return, "Vol": ann_vol, "Sharpe": sharpe, "MaxDD": max_dd}, index=["Portfolio"])
    
class AnalyticsForecasting:
    def __init__(
        self,
        config: Config,
        exp_window: ExpandingWindowScheme,
        dm: DataManager,
    ):
        self.config = config
        self.exp_window = exp_window
        self.dm = dm
        self.objs = None
        self.rmse_df = None
        
        # Récupération de la méthode (rolling/expanding)
        self.method = str(self.config.estimation_method).lower()
        
        # Chemins dynamiques pour les rapports (on évite le dossier raw du forecasting)
        self.base_path = f"{self.config.outputs_path}/forecasting/{self.method}"
        self.fig_path = f"{self.base_path}/figures"
        self.data_path = f"{self.base_path}/data"

        self.linear_models = [
            "ols", "ols_pca", "lasso", "lasso_pca",
            "ridge", "ridge_pca", "elastic_net", "elastic_net_pca",
        ]
        self.linear_models_without_pca = ["ols", "lasso", "ridge", "elastic_net"]
        self.non_numeric_hyperparams = [
            "neural_net", "neural_net_pca", "ols", "ols_pca", "svr", "svr_pca",
        ]

    def _render_styled_table(self, df: pd.DataFrame, title: str, figsize: tuple = (12, 6)) -> plt.Figure:
        """Rendu de tableau propre pour le dashboard"""
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        ax.axis("off")
        
        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        # Couleurs style 'Professional'
        header_color = '#2c3e50'
        row_colors = ['#ecf0f1', 'white']
        
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#bdc3c7')
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(header_color)
            elif col == -1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#dcdde1')
            else:
                cell.set_facecolor(row_colors[row % len(row_colors)])

        ax.set_title(title, pad=30, fontsize=13, weight='bold')
        fig.tight_layout()
        return fig

    def get_analytics(self) -> None:
        """Exécute toute la suite d'analyses."""
        self._load_objects()
        self._export_features_table()
        self._plot_best_score_overtime()
        self._plot_best_hyperparams_overtime()
        self._plot_model_parameters_overtime()
        self._export_selected_features_proportion()
        self._export_mean_parameters()
        self._plot_oos_rmse_overtime()
        self._export_oos_rmse_table()
        self._compute_oos_sign_accuracy()
        logger.info(f"Analytics Forecasting ({self.method}) terminées.")

    def _load_objects(self) -> None:
        self.objs = {k: v for k, v in vars(self.exp_window).items()}

    def _export_features_table(self) -> None:
        df = self.objs["x"].copy()
        if df.empty:
            return

        df = df.iloc[:10, :10].round(4)
        fig = self._render_styled_table(df, "Features Table (Sample)")

        path_name = f"{self.fig_path}/features_df_short.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _plot_best_score_overtime(self, y_lim: float = 0.015) -> None:
        df = self.objs["best_score_all_models_overtime"]
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df)
        ax.set_title("Best Validation RMSE Over Time per Model")
        ax.set_ylabel("RMSE")
        ax.set_ylim(0, y_lim)
        ax.legend(df.columns, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # PNG
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/best_val_score_all_models_overtime.png", fig=fig)
        plt.close(fig)

        # DATA (Parquet pour graphe interactif)
        s3Utils.upload_df_with_index(df=df, bucket=self.config.aws_bucket_name, 
                                     path=f"{self.data_path}/best_val_score_overtime.parquet")

    def _plot_best_hyperparams_overtime(self) -> None:
        hyperparams = self.objs["best_hyperparams_all_models_overtime"]
        mdl_names = list(hyperparams.keys())

        fig, axes = plt.subplots(5, 4, figsize=(14, 12))
        axes = axes.flatten()

        for ax, mdl in zip(axes, mdl_names):
            if mdl in self.non_numeric_hyperparams:
                ax.axis("off")
                continue
            df = hyperparams[mdl]
            if df.empty:
                ax.axis("off")
                continue
            ax.plot(df)
            ax.set_title(mdl, fontsize=10)
            ax.grid(True, alpha=0.2)

        for ax in axes[len(mdl_names):]: 
            ax.axis("off")
        fig.tight_layout()

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/best_hyperparams_all_models_overtime.png", fig=fig)
        plt.close(fig)

    def _plot_model_parameters_overtime(self) -> None:
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models}

        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
        axes = axes.flatten()

        for ax, (model, df) in zip(axes, params.items()):
            if df.empty:
                ax.axis("off")
                continue
            ax.plot(df)
            vals = df.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals):
                ax.set_ylim(np.percentile(vals, 1), np.percentile(vals, 99))
            ax.set_title(model)
            ax.grid(True, alpha=0.2)

        fig.tight_layout()
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/best_parameters_all_models_overtime.png", fig=fig)
        plt.close(fig)

    def _export_selected_features_proportion(self, max_rows: int = 25) -> None:
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models_without_pca}
        if not params:
            return

        features = next(iter(params.values())).columns
        df = pd.DataFrame(index=features, columns=list(params.keys()), dtype=float)

        for k, v in params.items():
            valid = v.dropna(how="all")
            df[k] = (abs(valid) > 0.001).sum() / valid.shape[0] * 100 if not valid.empty else np.nan

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False).head(max_rows).round(2)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

        fig = self._render_styled_table(df, "Proportion of Selected Features (%)")
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/proportion_selected_features.png", fig=fig)
        plt.close(fig)

    def _export_mean_parameters(self, max_rows: int = 25) -> None:
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models_without_pca}
        if not params:
            return

        features = next(iter(params.values())).columns
        df = pd.DataFrame(index=features, columns=list(params.keys()), dtype=float)

        for k, v in params.items():
            df[k] = v.mean(axis=0)

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False).head(max_rows).round(2)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

        fig = self._render_styled_table(df, "Mean Parameters")
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/mean_parameters.png", fig=fig)
        plt.close(fig)

    def _plot_oos_rmse_overtime(self, y_lim: float = 0.01, window: int = 12) -> None:
        y_true = self.objs["oos_true"]
        models = list(self.objs["oos_predictions"].keys())
        rmse_df = pd.DataFrame(index=y_true.index, columns=models, dtype=float)

        for model, y_pred in self.objs["oos_predictions"].items():
            rmse_df[model] = (y_pred.iloc[:, 0] - y_true.iloc[:, 0]).abs()

        rolling_rmse = rmse_df.rolling(window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rolling_rmse)
        ax.set_title(f"Rolling {window}M OOS RMSE Over Time")
        ax.set_ylim(0, y_lim)
        ax.legend(rmse_df.columns, fontsize=8)
        ax.grid(True, alpha=0.3)

        # PNG
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/oos_rmse_all_models_overtime.png", fig=fig)
        plt.close(fig)

        # DATA
        s3Utils.upload_df_with_index(df=rolling_rmse, bucket=self.config.aws_bucket_name, 
                                     path=f"{self.data_path}/oos_rmse_overtime.parquet")
        self.rmse_df = rmse_df

    def _export_oos_rmse_table(self) -> None:
        if self.rmse_df is None or self.rmse_df.empty:
            return

        df = self.rmse_df.mean().sort_values().to_frame("mean_rmse")
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df = df.round(4)

        fig = self._render_styled_table(df, "Mean OOS RMSE by Model")
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/oos_rmse_all_models.png", fig=fig)
        plt.close(fig)

        s3Utils.upload_df_with_index(df=df, bucket=self.config.aws_bucket_name, 
                                     path=f"{self.data_path}/oos_rmse_table.parquet")

    def _compute_oos_sign_accuracy(self) -> None:
        y_true = self.objs["oos_true"].iloc[:, 0]
        sign_acc = {}

        for model, y_pred_df in self.objs["oos_predictions"].items():
            y_pred = y_pred_df.iloc[:, 0]
            df = pd.concat([y_true, y_pred], axis=1, join="inner").dropna()
            df.columns = ["y_true", "y_pred"]
            if len(df) == 0:
                continue
            
            correct = (np.sign(df["y_true"]) == np.sign(df["y_pred"]))
            non_zero = (df["y_true"] != 0) & (df["y_pred"] != 0)
            sign_acc[model] = correct[non_zero].mean()

        res = pd.DataFrame.from_dict(sign_acc, orient="index", columns=["sign_accuracy"])
        res = res.sort_values("sign_accuracy", ascending=False).round(4)
        res.insert(0, "rank", np.arange(1, len(res) + 1))

        fig = self._render_styled_table(res, "OOS Sign Accuracy by Model")
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=f"{self.fig_path}/oos_sign_accuracy_all_models.png", fig=fig)
        plt.close(fig)

        s3Utils.upload_df_with_index(df=res, bucket=self.config.aws_bucket_name, 
                                     path=f"{self.data_path}/oos_sign_accuracy.parquet")

class AnalyticsDynamicAllocation:
    def __init__(
        self,
        config: Config,
        dynamic_alloc: DynamicAllocation,
        fmp: FactorMimickingPortfolio,
        dm: DataManager,
    ):
        self.config = config
        self.dynamic_alloc = dynamic_alloc
        self.fmp = fmp
        self.dm = dm

        # Récupération dynamique de la méthode (rolling ou expanding)
        self.method = str(self.config.estimation_method).lower()
        
        # Définition des dossiers de sortie (Module: dynamic_allocation)
        self.base_path = f"{self.config.outputs_path}/dynamic_allocation/{self.method}"
        self.fig_path = f"{self.base_path}/figures"
        self.data_path = f"{self.base_path}/data"

    def _render_styled_table(self, df: pd.DataFrame, title: str, figsize: tuple = (12, 5)) -> plt.Figure:
        """Rendu de tableau stylisé 'Finance Pro'"""
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
        ax.axis("off")
        
        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        # Style visuel (Bleu nuit / Gris / Blanc)
        header_color = '#1f4e78'
        row_colors = ['#f2f2f2', 'white']
        edge_color = '#d9d9d9'

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(edge_color)
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(header_color)
            elif col == -1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e6e6e6')
            else:
                cell.set_facecolor(row_colors[row % len(row_colors)])

        ax.set_title(title, pad=30, fontsize=14, weight='bold')
        fig.tight_layout()
        return fig

    def get_analytics(self) -> None:
        """Exécute toute la suite d'analyses de l'allocation dynamique."""
        returns_df = self._build_aligned_returns_df()
        cum_rets = self._build_cumulative_returns_dict(returns_df)

        # 1. Graphique des rendements cumulés
        self._plot_dynamic_allocation_cum_returns(cum_rets)

        # 2. Tableau de performance
        perf_table = self._build_performance_table(returns_df)
        self._export_performance_table(perf_table)

        # 3. Sauvegarde des Parquets pour le Dashboard interactif
        self._save_dashboard_data(cum_rets, perf_table)

        logger.info(f"Analytics Dynamic Allocation ({self.method}) terminées.")

    def _build_aligned_returns_df(self) -> pd.DataFrame:
        strat_rets = self.dynamic_alloc.net_returns
        returns_df = pd.concat({k: v.iloc[:, 0] for k, v in strat_rets.items()}, axis=1)

        # Benchmark alignement
        bench = self.fmp.benchmark_returns.copy()
        bench.index = bench.index - pd.DateOffset(months=1) - pd.DateOffset(days=1)

        returns_df["Bench LO EW stocks"] = bench.iloc[:, 0]
        returns_df["Bench EW FMP"] = self.dynamic_alloc.benchmark_ew_fmp_net_returns.iloc[:, 0]

        return returns_df.dropna(how="any")

    @staticmethod
    def _build_cumulative_returns_dict(returns_df: pd.DataFrame) -> dict:
        return {col: (1 + returns_df[[col]]).cumprod() - 1 for col in returns_df.columns}

    def _plot_dynamic_allocation_cum_returns(self, cum_rets: dict) -> None:
        if not cum_rets:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        dashed_keys = {"Bench LO EW stocks", "Bench EW FMP"}

        for name, df in cum_rets.items():
            if df.empty: 
                continue
            series = df.iloc[:, 0]
            if name in dashed_keys:
                ax.plot(series.index, series.values, linestyle="--", color="black", label=name, alpha=0.7)
            else:
                ax.plot(series.index, series.values, label=name, linewidth=2)

        ax.set_title("Strategy Cumulative Returns (Dynamic Allocation)")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        path_name = f"{self.fig_path}/dynamic_allocation_cum_returns.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _build_performance_table(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            col: self._compute_performance_metrics(returns_df[col])
            for col in returns_df.columns
        }).T

    def _compute_performance_metrics(self, returns: pd.Series, periods_per_year: int = 12) -> pd.Series:
        returns = returns.dropna()
        if returns.empty:
            return pd.Series({"Ann. Return": np.nan, "Ann. Vol": np.nan, "Sharpe": np.nan, "Max DD": np.nan})

        ann_ret = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
        cum = (1 + returns).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

        return pd.Series({"Ann. Return": ann_ret, "Ann. Vol": ann_vol, "Sharpe": sharpe, "Max DD": max_dd})

    def _export_performance_table(self, perf_table: pd.DataFrame) -> None:
        if perf_table.empty:
            return

        df_fmt = perf_table.copy()
        df_fmt[["Ann. Return", "Ann. Vol", "Max DD"]] *= 100
        df_fmt = df_fmt.round(2)
        df_fmt = df_fmt.sort_values("Sharpe", ascending=False)
        df_fmt.insert(0, "Rank", np.arange(1, len(df_fmt) + 1))

        fig = self._render_styled_table(df_fmt, "Dynamic Allocation Performance Summary")
        path_name = f"{self.fig_path}/performance_table.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _save_dashboard_data(self, cum_rets: dict, perf_table: pd.DataFrame) -> None:
        """Sauvegarde les parquets dans le dossier /data/"""
        # 1. Cumulative returns
        cum_returns_df = pd.concat({k: v.iloc[:, 0] for k, v in cum_rets.items()}, axis=1)
        s3Utils.upload_df_with_index(
            df=cum_returns_df,
            bucket=self.config.aws_bucket_name,
            path=f"{self.data_path}/dynamic_allocation_cum_returns.parquet"
        )

        # 2. Performance Table
        s3Utils.upload_df_with_index(
            df=perf_table,
            bucket=self.config.aws_bucket_name,
            path=f"{self.data_path}/dynamic_allocation_performance_table.parquet"
        )




