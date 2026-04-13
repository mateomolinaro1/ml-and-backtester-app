import pandas as pd
import numpy as np
import matplotlib
from ml_and_backtester_app.dynamic_allocation.dynamic_allocation import DynamicAllocation
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import logging
from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager
from ml_and_backtester_app.fmp.fmp import FactorMimickingPortfolio
from ml_and_backtester_app.machine_learning.schemes.expanding import ExpandingWindowScheme
from ml_and_backtester_app.utils.s3_utils import s3Utils

logger = logging.getLogger(__name__)

class AnalyticsFMP:
    def __init__(
            self,
            config:Config,
            dm:DataManager,
            fmp: FactorMimickingPortfolio,
    ):
        """
        Analytics for Factor Mimicking Portfolio (FMP)
        :param config:
        :param fmp:
        """
        self.config = config
        self.fmp = fmp
        self.dm = dm

    def get_analytics(self) -> None:
        """
        Get analytics for FMP
        :return: None
        """
        self._get_bayesian_betas_distribution()
        self._get_bayesian_betas_overtime()
        self._get_table_comparison_between_bayesian_and_non_bayesian_betas()
        self._get_summary_statistics_rsquared()
        self._get_cross_sectional_statistics_rsquared()
        self._plot_significant_proportion_bayesian_betas_overtime()
        self._plot_equity_curves_fmp()
        self._export_fmp_performance_table()

    def _get_bayesian_betas_distribution(self) -> None:
        # Extract betas
        betas = self.fmp.bayesian_betas.stack().dropna()

        if betas.empty:
            logger.warning("No data available for Bayesian betas distribution.")
            return

        # Dates
        start_date = betas.index[0][0]
        end_date = betas.index[-1][0]

        # Clip extreme tails for visualization
        lo, hi = betas.quantile([0.01, 0.99])
        betas_clip = betas.clip(lo, hi)

        # Create figure properly
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(betas_clip, bins=40, density=True, alpha=0.6, color="steelblue")
        betas_clip.plot(kind="kde", color="darkred", linewidth=2, ax=ax)

        ax.set_title(
            f"Bayesian Betas Distribution (clipped 1–99%)\n"
            f"{start_date.date()} – {end_date.date()}"
        )
        ax.set_xlabel("Bayesian Beta")
        ax.set_ylabel("Density")

        fig.tight_layout()
        # S3 path
        path_name = (
                self.config.outputs_path
                + "/figures/bayesian_betas_distribution.png"
        )
        # Upload to S3
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        # Clean up
        plt.close(fig)

        logger.info("Saved Bayesian betas distribution plot to S3.")

    def _get_bayesian_betas_overtime(self) -> None:
        betas = self.fmp.bayesian_betas

        if betas.empty:
            logger.warning("No data available for Bayesian betas over time.")
            return

        avg = betas.mean(axis=1)
        p10 = betas.quantile(0.10, axis=1)
        p90 = betas.quantile(0.90, axis=1)

        avg_non_na = avg.dropna()
        if avg_non_na.empty:
            logger.warning("No non-missing average Bayesian betas available over time.")
            return

        start_date, end_date = avg_non_na.index[[0, -1]]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(avg.index, avg, label="Cross-sectional mean", linewidth=2)
        ax.fill_between(
            p10.index,
            p10,
            p90,
            alpha=0.3,
            label="10–90 percentile band",
        )

        # Robust y-limits
        ylo, yhi = np.nanpercentile(betas.values, [2, 98])
        ax.set_ylim(ylo, yhi)

        ax.set_title(
            f"Bayesian Betas Cross-Sectional Dynamics\n"
            f"{start_date.date()} – {end_date.date()}"
        )
        ax.set_ylabel("Beta")
        ax.legend()

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/bayesian_betas_overtime.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info("Saved Bayesian betas over time plot to S3.")

    def _get_cross_sectional_statistics_rsquared(self) -> None:
        rsq = self.fmp.adjusted_rsquared

        if rsq.empty:
            logger.warning("No data available for adjusted R² over time.")
            return

        avg = rsq.mean(axis=1)
        p10 = rsq.quantile(0.10, axis=1)
        p90 = rsq.quantile(0.90, axis=1)

        avg_non_na = avg.dropna()
        if avg_non_na.empty:
            logger.warning("No non-missing average adjusted R² available.")
            return

        start_date, end_date = avg_non_na.index[[0, -1]]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(avg.index, avg, label="Mean adjusted $R^2$", linewidth=2)
        ax.fill_between(
            p10.index,
            p10,
            p90,
            alpha=0.3,
            label="10–90 percentile band",
        )

        ax.set_ylim(0, 1)

        ax.set_title(
            f"Adjusted $R^2$ Cross-Sectional Dynamics\n"
            f"{start_date.date()} – {end_date.date()}"
        )
        ax.set_ylabel("Adjusted $R^2$")
        ax.legend()

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/rsquared_overtime.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info("Saved R-squared over time plot to S3.")

    def _plot_significant_proportion_bayesian_betas_overtime(
            self,
            significance_level: float = 0.05,
    ) -> None:
        p_values = self.fmp.newey_west_pvalue

        if p_values.empty:
            logger.warning("No p-values available for Bayesian betas significance plot.")
            return

        significant = p_values < significance_level
        total = p_values.notnull().sum(axis=1)

        # Avoid division by zero
        proportion = significant.sum(axis=1) / total.replace(0, np.nan)

        proportion_non_na = proportion.dropna()
        if proportion_non_na.empty:
            logger.warning("No valid proportion of significant betas available.")
            return

        start_date, end_date = proportion_non_na.index[[0, -1]]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(proportion.index, proportion, linewidth=2)
        ax.set_ylim(0, 1)

        ax.set_title(
            f"Proportion of Significant Bayesian Betas (α={significance_level})\n"
            f"{start_date.date()} – {end_date.date()}"
        )
        ax.set_ylabel("Proportion")

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/proportion_significant_bayesian_betas_overtime.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info("Saved proportion of significant Bayesian betas over time plot to S3.")

    def _get_table_comparison_between_bayesian_and_non_bayesian_betas(self) -> None:
        """
        Create and save to S3 a table comparing Bayesian and non-Bayesian betas
        summary statistics.
        """
        bayesian_betas = self.fmp.bayesian_betas.stack().dropna()
        non_bayesian_betas = self.fmp.betas_macro.stack().dropna()

        if bayesian_betas.empty or non_bayesian_betas.empty:
            logger.warning(
                "No data available to create Bayesian vs non-Bayesian betas summary table."
            )
            return

        stats = ["Mean", "Median", "Std Dev", "Min", "Max"]

        summary_df = pd.DataFrame(
            {
                "Bayesian Betas": [
                    bayesian_betas.mean(),
                    bayesian_betas.median(),
                    bayesian_betas.std(),
                    bayesian_betas.min(),
                    bayesian_betas.max(),
                ],
                "Non-Bayesian Betas": [
                    non_bayesian_betas.mean(),
                    non_bayesian_betas.median(),
                    non_bayesian_betas.std(),
                    non_bayesian_betas.min(),
                    non_bayesian_betas.max(),
                ],
            },
            index=stats,
        ).round(2)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")

        table = ax.table(
            cellText=summary_df.values,
            rowLabels=summary_df.index,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title(
            "Bayesian vs Non-Bayesian Betas Summary Statistics",
            pad=20,
        )

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/bayesian_vs_non_bayesian_betas_summary.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info(
            "Saved Bayesian vs Non-Bayesian Betas summary statistics table to S3."
        )

    def _get_summary_statistics_rsquared(self) -> None:
        """
        Create and save to S3 a table summarizing adjusted R-squared statistics.
        """
        rsquared = self.fmp.adjusted_rsquared.stack().dropna()

        if rsquared.empty:
            logger.warning("No data available to compute R-squared summary statistics.")
            return

        stats = ["Mean", "Median", "Std Dev", "Min", "Max"]

        summary_df = pd.DataFrame(
            {
                "Adjusted R-squared": [
                    rsquared.mean(),
                    rsquared.median(),
                    rsquared.std(),
                    rsquared.min(),
                    rsquared.max(),
                ]
            },
            index=stats,
        ).round(2)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")

        table = ax.table(
            cellText=summary_df.values,
            rowLabels=summary_df.index,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title("Adjusted R-squared Summary Statistics", pad=20)

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/rsquared_summary.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info("Saved R-squared summary statistics table to S3.")

    def _plot_equity_curves_fmp(self) -> None:
        returns = pd.concat(
            [
                self.fmp.positive_betas_fmp_returns,
                self.fmp.negative_betas_fmp_returns,
                self.fmp.benchmark_returns,
            ],
            axis=1,
        ).dropna(how="all")

        if returns.empty:
            logger.warning("No data available for FMP equity curves.")
            return

        start_date = returns.index[0]
        equity = returns.loc[start_date:].cumsum()

        fig, ax = plt.subplots(figsize=(12, 6))

        for col in equity.columns:
            ax.plot(equity.index, equity[col], label=col, linewidth=2)

        ax.set_title("LO / SO Factor-Mimicking Portfolios – Equity Curves")
        ax.set_ylabel("Equity Curve")
        ax.set_xlabel("Date")
        ax.legend()

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/fmp_equity_curves.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        s3Utils.upload_df_with_index(
            df=equity,
            bucket=self.config.aws_bucket_name,
            path=self.config.outputs_path + "/data/fmp_equity_curves.parquet",
        )

        logger.info("Saved FMP equity curves plot to S3.")

    def _compute_performance_metrics(
            self,
            returns: pd.Series | pd.DataFrame,
            portfolio_type: str = "long_only",
            freq: int = 12,
            risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute performance metrics and save the table to S3.
        """

        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()

        if not isinstance(returns, pd.Series):
            raise ValueError("returns must be a pd.Series or single-column DataFrame")

        # Short-only economics
        if portfolio_type == "short_only":
            returns = -returns

        returns = returns.dropna()

        if returns.empty:
            logger.warning("No returns available to compute performance metrics.")
            return pd.DataFrame()

        # === Annualized return ===
        ann_return = (1 + returns).prod() ** (freq / len(returns)) - 1

        # === Annualized volatility ===
        ann_vol = returns.std() * np.sqrt(freq)

        # === Sharpe ratio ===
        sharpe = np.nan
        if ann_vol > 0:
            sharpe = (ann_return - risk_free_rate) / ann_vol

        # === Max drawdown ===
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1
        max_dd = drawdown.min()

        metrics = pd.DataFrame(
            {
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd,
            },
            index=["Portfolio"],
        ).round(2)

        # === Render table ===
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")

        table = ax.table(
            cellText=metrics.values,
            rowLabels=metrics.index,
            colLabels=metrics.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title("Performance Metrics", pad=20)

        fig.tight_layout()

        path_name = (
                self.config.outputs_path
                + "/figures/performance_metrics.png"
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info("Saved performance metrics table to S3.")

        return metrics

    def _export_fmp_performance_table_util(
            self,
            fmp,
            path_name: str,
            freq: int = 12,
            risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute and export performance metrics for FMP portfolios to S3.
        """

        perf_pos = self._compute_performance_metrics(
            fmp.positive_betas_fmp_returns,
            portfolio_type="long_only",
            freq=freq,
            risk_free_rate=risk_free_rate,
        )

        perf_neg = self._compute_performance_metrics(
            fmp.negative_betas_fmp_returns,
            portfolio_type="short_only",
            freq=freq,
            risk_free_rate=risk_free_rate,
        )

        perf_bench = self._compute_performance_metrics(
            fmp.benchmark_returns,
            portfolio_type="long_only",
            freq=freq,
            risk_free_rate=risk_free_rate,
        )

        performance_table = pd.concat(
            [perf_pos, perf_neg, perf_bench],
            axis=0,
        )

        performance_table.index = ["LO FMP", "SO FMP", "Benchmark"]
        performance_table = performance_table.round(2)

        s3Utils.upload_df_with_index(
            df=performance_table,
            bucket=self.config.aws_bucket_name,
            path=self.config.outputs_path + "/data/fmp_performance_table.parquet",
        )

        # === Render table ===
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")

        table = ax.table(
            cellText=performance_table.values,
            rowLabels=performance_table.index,
            colLabels=performance_table.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title("FMP Performance Comparison", pad=20)

        fig.tight_layout()

        # === Upload to S3 ===
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)

        plt.close(fig)

        logger.info("Saved FMP performance table to S3.")

        return performance_table

    def _export_fmp_performance_table(self) -> None:
        path_name = (
                self.config.outputs_path
                + "/figures/fmp_performance_summary.png"
        )

        self._export_fmp_performance_table_util(
            fmp=self.fmp,
            path_name=path_name,
        )

        logger.info("Saved FMP performance summary table to S3.")

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
        self.linear_models = [
            "ols", "ols_pca", "lasso", "lasso_pca",
            "ridge", "ridge_pca", "elastic_net", "elastic_net_pca",
        ]
        self.linear_models_without_pca = ["ols", "lasso", "ridge", "elastic_net"]
        self.non_numeric_hyperparams = [
            "neural_net", "neural_net_pca", "ols", "ols_pca", "svr", "svr_pca",
        ]

    @staticmethod
    def _render_table(
        df: pd.DataFrame,
        title: str,
        figsize: tuple[float, float] = (10, 4),
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
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
        table.scale(1.2, 1.4)

        ax.set_title(title, pad=20)
        fig.tight_layout()
        return fig

    def get_analytics(self) -> None:
        """Run all forecasting analytics."""
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
        logger.info("Completed forecasting analytics.")

    def _load_objects(self) -> None:
        self.objs = {k: v for k, v in vars(self.exp_window).items()}

    def _export_features_table(self) -> None:
        df = self.objs["x"].copy()

        if df.empty:
            logger.warning("No features data available.")
            return

        df = df.iloc[:10, :10].round(4)

        fig = self._render_table(
            df=df,
            title="Features Table (first 10 rows / 10 columns)",
            figsize=(10, 4),
        )

        path_name = self.config.outputs_path + "/figures/features_df_short.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _plot_best_score_overtime(self, y_lim: float = 0.015) -> None:
        df = self.objs["best_score_all_models_overtime"]

        if df.empty:
            logger.warning("No best validation score data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df)
        ax.set_title("Best Validation RMSE overtime per model")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Date")
        ax.set_ylim(0, y_lim)
        ax.legend(df.columns)
        ax.grid(True)

        fig.tight_layout()

        path_name = (
            self.config.outputs_path
            + "/figures/best_val_score_all_models_overtime.png"
        )
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

        s3Utils.upload_df_with_index(
            df=df,
            bucket=self.config.aws_bucket_name,
            path=self.config.outputs_path + "/data/best_val_score_overtime.parquet",
        )

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
            ax.set_title(mdl)
            ax.grid(True)
            ax.legend(df.columns, fontsize=8)

        for ax in axes[len(mdl_names):]:
            ax.axis("off")

        fig.tight_layout()

        path_name = (
            self.config.outputs_path
            + "/figures/best_hyperparams_all_models_overtime.png"
        )
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _plot_model_parameters_overtime(self) -> None:
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models}

        n_cols = 4
        n_rows = 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
        axes = axes.flatten()

        for ax, (model, df) in zip(axes, params.items()):
            if df.empty:
                ax.axis("off")
                continue

            ax.plot(df)

            vals = df.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals):
                q1, q99 = np.percentile(vals, [1, 99])
                ax.set_ylim(q1, q99)

            ax.set_title(model)
            ax.grid(True)

        for ax in axes[len(params):]:
            ax.axis("off")

        fig.tight_layout()

        path_name = (
            self.config.outputs_path
            + "/figures/best_parameters_all_models_overtime.png"
        )
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _export_selected_features_proportion(self, max_rows: int = 25) -> None:
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models_without_pca}

        if not params:
            logger.warning("No linear model parameters available.")
            return

        features = next(iter(params.values())).columns
        df = pd.DataFrame(index=features, columns=list(params.keys()), dtype=float)

        for k, v in params.items():
            valid = v.dropna(how="all")
            if valid.empty:
                df[k] = np.nan
            else:
                df[k] = (abs(valid) > 0.001).sum() / valid.shape[0] * 100

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

        df_to_export = df.head(max_rows).round(2)

        fig = self._render_table(
            df=df_to_export,
            title="Proportion of Selected Features (%)",
            figsize=(12, 6),
        )

        path_name = self.config.outputs_path + "/figures/proportion_selected_features.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _export_mean_parameters(self, max_rows: int = 25) -> None:
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models_without_pca}

        if not params:
            logger.warning("No linear model parameters available.")
            return

        features = next(iter(params.values())).columns
        df = pd.DataFrame(index=features, columns=list(params.keys()), dtype=float)

        for k, v in params.items():
            df[k] = v.mean(axis=0)

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

        df_to_export = df.head(max_rows).round(2)

        fig = self._render_table(
            df=df_to_export,
            title="Mean Parameters",
            figsize=(12, 6),
        )

        path_name = self.config.outputs_path + "/figures/mean_parameters.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    def _plot_oos_rmse_overtime(
        self,
        y_lim: float = 0.01,
        window: int = 12,
        freq: str = "m",
    ) -> None:
        """
        Plot OOS RMSE (here: absolute error) over time for each model.
        """
        y_true = self.objs["oos_true"]
        models = list(self.objs["oos_predictions"].keys())

        rmse_df = pd.DataFrame(index=y_true.index, columns=models, dtype=float)

        for model, y_pred in self.objs["oos_predictions"].items():
            rmse_df[model] = (y_pred.iloc[:, 0] - y_true.iloc[:, 0]).abs()

        rolling_rmse = rmse_df.rolling(window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rolling_rmse)
        ax.set_title(f"Rolling {window}{freq} OOS RMSE overtime per model")
        ax.set_xlabel("Date")
        ax.set_ylabel("OOS RMSE")
        ax.set_ylim(0, y_lim)
        ax.grid(True)
        ax.legend(rmse_df.columns)

        fig.tight_layout()

        path_name = (
            self.config.outputs_path
            + "/figures/oos_rmse_all_models_overtime.png"
        )
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

        s3Utils.upload_df_with_index(
            df=rolling_rmse,
            bucket=self.config.aws_bucket_name,
            path=self.config.outputs_path + "/data/oos_rmse_overtime.parquet",
        )

        self.rmse_df = rmse_df

    def _export_oos_rmse_table(self) -> None:
        if self.rmse_df is None or self.rmse_df.empty:
            logger.warning("RMSE table cannot be exported because rmse_df is empty.")
            return

        mean_rmse = self.rmse_df.mean().sort_values()
        df = mean_rmse.to_frame("mean_rmse")
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df = df.round(2)

        fig = self._render_table(
            df=df,
            title="Mean OOS RMSE by Model",
            figsize=(8, 4),
        )

        path_name = self.config.outputs_path + "/figures/oos_rmse_all_models.png"
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

        s3Utils.upload_df_with_index(
            df=df,
            bucket=self.config.aws_bucket_name,
            path=self.config.outputs_path + "/data/oos_rmse_table.parquet",
        )

    def _compute_oos_sign_accuracy(self) -> None:
        """
        Compute the proportion of times the model predicts the correct sign
        of the target.
        """
        y_true = self.objs["oos_true"].iloc[:, 0]
        sign_acc = {}

        for model, y_pred_df in self.objs["oos_predictions"].items():
            y_pred = y_pred_df.iloc[:, 0]

            df = pd.concat([y_true, y_pred], axis=1, join="inner").dropna()
            df.columns = ["y_true", "y_pred"]

            if len(df) == 0:
                sign_acc[model] = np.nan
                continue

            correct_sign = pd.Series(
                np.sign(df["y_true"]) == np.sign(df["y_pred"]),
                index=df.index,
            )

            non_zero = (df["y_true"] != 0) & (df["y_pred"] != 0)
            sign_acc[model] = correct_sign[non_zero].mean()

        res = pd.DataFrame.from_dict(
            sign_acc,
            orient="index",
            columns=["sign_accuracy"],
        )

        res = res.sort_values("sign_accuracy", ascending=False)
        res.insert(0, "rank", np.arange(1, len(res) + 1))
        res = res.round(2)

        fig = self._render_table(
            df=res,
            title="OOS Sign Accuracy by Model",
            figsize=(8, 4),
        )

        path_name = (
            self.config.outputs_path
            + "/figures/oos_sign_accuracy_all_models.png"
        )
        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

        s3Utils.upload_df_with_index(
            df=res,
            bucket=self.config.aws_bucket_name,
            path=self.config.outputs_path + "/data/oos_sign_accuracy.parquet",
        )

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

    @staticmethod
    def _render_table(
        df: pd.DataFrame,
        title: str,
        figsize: tuple[float, float] = (10, 4),
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
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
        table.scale(1.2, 1.4)

        ax.set_title(title, pad=20)
        fig.tight_layout()
        return fig

    def get_analytics(self) -> None:
        """Run all dynamic allocation analytics."""
        returns_df = self._build_aligned_returns_df()
        cum_rets = self._build_cumulative_returns_dict(returns_df)

        self._plot_dynamic_allocation_cum_returns(
            cum_rets,
            path_name=(
                self.config.outputs_path
                + "/figures/dynamic_allocation_cum_returns.png"
            ),
        )

        perf_table = self._build_performance_table(
            returns_df,
            periods_per_year=12,
        )

        self._export_performance_table(
            perf_table,
            path_name=self.config.outputs_path + "/figures/performance_table.png",
        )

        # Save DataFrames for interactive dashboard charts
        cum_returns_df = pd.concat(
            {k: v.iloc[:, 0] for k, v in cum_rets.items()}, axis=1
        )
        _data_prefix = self.config.outputs_path + "/data"
        s3Utils.upload_df_with_index(
            df=cum_returns_df,
            bucket=self.config.aws_bucket_name,
            path=_data_prefix + "/dynamic_allocation_cum_returns.parquet",
        )
        s3Utils.upload_df_with_index(
            df=perf_table,
            bucket=self.config.aws_bucket_name,
            path=_data_prefix + "/dynamic_allocation_performance_table.parquet",
        )

        logger.info("Completed dynamic allocation analytics.")

    def _build_aligned_returns_df(self) -> pd.DataFrame:
        """
        Build a DataFrame of aligned non-cumulated returns:
        - Dynamic allocation strategies
        - Benchmark LO EW stocks
        - Benchmark EW FMP
        """
        strat_rets = self.dynamic_alloc.net_returns

        returns_df = pd.concat(
            {k: v.iloc[:, 0] for k, v in strat_rets.items()},
            axis=1,
        )

        bench = self.fmp.benchmark_returns.copy()
        bench.index = (
            bench.index
            - pd.DateOffset(months=1)
            - pd.DateOffset(days=1)
        )

        returns_df["Bench LO EW stocks"] = bench.iloc[:, 0]
        returns_df["Bench EW FMP"] = (
            self.dynamic_alloc.benchmark_ew_fmp_net_returns.iloc[:, 0]
        )

        return returns_df.dropna(how="any")

    @staticmethod
    def _build_cumulative_returns_dict(
        returns_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Convert a returns DataFrame into a dict of cumulative return DataFrames.
        """
        cum_rets = {}

        for col in returns_df.columns:
            cum_rets[col] = (1 + returns_df[[col]]).cumprod() - 1

        return cum_rets

    def _plot_dynamic_allocation_cum_returns(
        self,
        cum_rets: dict[str, pd.DataFrame],
        path_name: str,
    ) -> None:
        """
        Plot cumulative returns for dynamic allocation strategies and benchmarks
        and save the figure to S3.
        """
        if not cum_rets:
            logger.warning("No cumulative returns available for plotting.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        dashed_black_keys = {"Bench LO EW stocks", "Bench EW FMP"}

        for name, df in cum_rets.items():
            if df.empty:
                continue

            series = df.iloc[:, 0]

            if name in dashed_black_keys:
                ax.plot(series.index, series.values, linestyle="--", color="black", label=name)
            else:
                ax.plot(series.index, series.values, label=name)

        ax.set_title("Dynamic Allocation Strategy Cumulative Returns")
        ax.set_ylabel("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)

    @staticmethod
    def _compute_performance_metrics(
        returns: pd.Series,
        periods_per_year: int = 12,
    ) -> pd.Series:
        """
        Compute performance, risk and risk-adjusted metrics from a return series.

        Metrics:
        - Annualized return
        - Annualized volatility
        - Sharpe ratio (rf = 0)
        - Max drawdown
        """
        returns = returns.dropna()

        if returns.empty:
            return pd.Series(
                {
                    "Ann. Return": np.nan,
                    "Ann. Vol": np.nan,
                    "Sharpe": np.nan,
                    "Max Drawdown": np.nan,
                }
            )

        ann_ret = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan

        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_dd = drawdown.min()

        return pd.Series(
            {
                "Ann. Return": ann_ret,
                "Ann. Vol": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd,
            }
        )

    def _build_performance_table(
        self,
        returns_df: pd.DataFrame,
        periods_per_year: int = 12,
    ) -> pd.DataFrame:
        """
        Build a performance table for all strategies/benchmarks.
        """
        perf_table = pd.DataFrame(
            {
                col: self._compute_performance_metrics(
                    returns_df[col],
                    periods_per_year=periods_per_year,
                )
                for col in returns_df.columns
            }
        ).T

        return perf_table

    def _export_performance_table(
        self,
        perf_table: pd.DataFrame,
        path_name: str,
    ) -> None:
        """
        Export performance table to S3 as an image.
        """
        if perf_table.empty:
            logger.warning("No performance table available to export.")
            return

        perf_table_fmt = perf_table.copy()
        perf_table_fmt[["Ann. Return", "Ann. Vol", "Max Drawdown"]] *= 100
        perf_table_fmt = perf_table_fmt.round(2)
        perf_table_fmt = perf_table_fmt.sort_values("Sharpe", ascending=False)
        perf_table_fmt.insert(
            0,
            "Rank",
            np.arange(1, len(perf_table_fmt) + 1),
        )

        fig = self._render_table(
            perf_table_fmt,
            title="Dynamic Allocation Performance Table",
            figsize=(12, 4),
        )

        s3Utils.save_plot_to_s3(dm=self.dm, path_name=path_name, fig=fig)
        plt.close(fig)






