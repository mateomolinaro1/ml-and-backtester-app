from __future__ import annotations
import time
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from typing import Dict, Type
import logging
import os

from .base import EstimationScheme
from ml_and_backtester_app.machine_learning.models import Model
from ml_and_backtester_app.machine_learning.features_selection import PCAFactorExtractor

logger = logging.getLogger(__name__)

tmp_dir = os.path.join(os.getcwd(), "tmp") 
os.makedirs(tmp_dir, exist_ok=True)

class RollingWindowScheme(EstimationScheme):

    def run(
        self,
        models: Dict[str, Type[Model]],
        hyperparams_grid: Dict[str, Dict[str, list]]
    ) -> None:
        if self.config.load_or_train_models == "load":
            loaded_obj = self._load_models()
            self._put_in_attributes(loaded_obj=loaded_obj)
            logger.info("Models and results loaded from S3.")
            return
        elif self.config.load_or_train_models != "train":
            raise ValueError("config.load_or_train_models must be either 'load' or 'train'")

        # ---------------------------------------------------------
        # CONFIGURATION HARDCODÉE SELON LA FRÉQUENCE
        # ---------------------------------------------------------
        data_freq = getattr(self.config, "data_frequency", "monthly").lower()
        
        if data_freq == "daily":
            # Paramètres journaliers (Fenêtre 1 an, Val 3 mois, Refit mensuel)
            h_rolling_window = 252 
            h_validation_window = 63
            h_step_size = 22
        else:
            # Paramètres mensuels (Fenêtre 5 ans, Val 1 an, Refit chaque mois)
            h_rolling_window = 60
            h_validation_window = 12
            h_step_size = 1

        logger.info(f"Mode: {data_freq.upper()} | Window: {h_rolling_window} | Val: {h_validation_window} | Step: {h_step_size}")

        # -----------------------------
        # Hyperparams combinations
        # -----------------------------
        hyperparams_all_combinations = self.build_hyperparams_combinations(
            hyperparameters_grid=hyperparams_grid
        )

        # -----------------------------
        # STORE BETAS OVER TIME (linear only)
        # -----------------------------
        linear_models = [m for m in models if any(x in m for x in ["ridge","lasso","elastic_net","ols"])]
        columns_list = ["intercept"] + list(self.data.drop(columns=[self.config.macro_var_name]).columns)
        
        self.best_params_all_models_overtime = {
            m: pd.DataFrame(index=self.date_range, columns=columns_list, dtype=float) if m in linear_models else None 
            for m in models
        }
        
        if self.config.with_pca:
            pca_cols = ["intercept"] + [f"factor_{i + 1}" for i in range(self.config.nb_pca_components)]
            for model_name in self.best_params_all_models_overtime:
                if model_name and model_name.endswith("_pca"):
                    self.best_params_all_models_overtime[model_name] = pd.DataFrame(
                        index=self.date_range, columns=pca_cols, dtype=float
                    )

        # -----------------------------
        # STORAGE
        # -----------------------------
        self.oos_predictions = {m: pd.DataFrame(
            index=self.date_range, data=np.nan, columns=[self.config.macro_var_name]
        ) for m in models}
        
        self.oos_true = pd.DataFrame(index=self.date_range, columns=[self.config.macro_var_name], dtype=float)
        
        self.best_score_all_models_overtime = pd.DataFrame(
            index=self.date_range, columns=list(models.keys()),
        )
        self.best_hyperparams_all_models_overtime = {
            m: pd.DataFrame(index=self.date_range, columns=list(hyperparams_all_combinations[m][0].keys()))
            for m in models
        }

        # -----------------------------
        # WALK-FORWARD LOOP
        # -----------------------------
        self.active_bundles = {}
        # L'index de départ dépend des fenêtres hardcodées
        start_idx = h_rolling_window + h_validation_window + self.forecast_horizon

        for t in range(start_idx, len(self.date_range) - self.forecast_horizon):
            date_t = self.date_range[t]
            t0 = time.time()

            # --- CONDITION DE RÉ-ENTRAÎNEMENT (STEP HARDCODÉ) ---
            if (t - start_idx) % h_step_size == 0:
                logger.info(f"### [REFIT] Training models at {date_t} (Step {t}) ###")
                
                # Utilisation des fenêtres hardcodées
                train_data, val_data, _ = self._get_train_validation_split(t, h_rolling_window, h_validation_window)
                X_train, y_train = self._split_xy(train_data)

                for model_name, ModelClass in models.items():
                    X_train_run = X_train
                    val_data_run = val_data
                    pca_extractor = None
                    
                    if model_name.endswith("_pca"):
                        pca_extractor = PCAFactorExtractor(n_factors=self.config.nb_pca_components)
                        X_train_run = pca_extractor.fit_transform(X_train)
                        val_x = val_data.drop(columns=[self.config.macro_var_name])
                        val_x_pca = pca_extractor.transform(val_x)
                        val_data_run = pd.concat([val_x_pca, val_data[[self.config.macro_var_name]]], axis=1)

                    # --- HYPERPARAMETER SEARCH ---
                    best_score = -np.inf # On cherche à maximiser (-RMSE)
                    best_hyperparams = None

                    for hp in hyperparams_all_combinations[model_name]:
                        model_eval = ModelClass(**hp)
                        model_eval.fit(X_train_run, y_train)

                        ICs = []
                        for d in val_data_run.index.unique():
                            val_d = val_data_run[val_data_run.index == d]
                            X_v, y_v = self._split_xy(val_d)
                            if len(y_v) > 0:
                                y_hat_v = model_eval.predict(X_v)
                                rmse = np.sqrt(mean_squared_error(y_v, y_hat_v))
                                ICs.append(rmse)

                        current_score = -np.mean(ICs) if ICs else -np.inf
                        if current_score > best_score:
                            best_score = current_score
                            best_hyperparams = hp

                    # --- FINAL TRAIN DU MODÈLE ---
                    model_final = ModelClass(**best_hyperparams)
                    model_final.fit(X_train_run, y_train)

                    self.active_bundles[model_name] = {
                        "model": model_final,
                        "pca": pca_extractor,
                        "X_train_cols": X_train_run.columns
                    }

                    # Sauvegarde et Logs
                    self.best_score_all_models_overtime.loc[date_t, model_name] = -best_score # On restock la RMSE positive
                    for k, v in best_hyperparams.items():
                        self.best_hyperparams_all_models_overtime[model_name].loc[date_t, k] = v

                    model_filename = f"rolling_{model_name}_{date_t.strftime('%Y-%m-%d')}.pkl"
                    local_path = os.path.join(tmp_dir, model_filename)
                    joblib.dump(model_final, local_path)
                    self.dm.aws.s3.upload(local_path, key=f"models/rolling/{model_name}/{model_filename}")
                    if os.path.exists(local_path): os.remove(local_path)

                logger.info(f"Refit completed in {round((time.time() - t0) / 60, 3)} min")

            # --- PRÉDICTION OUT-OF-SAMPLE (CHAQUE T) ---
            test_df = self.data[self.data.index == date_t]
            X_test_raw, y_test = self._split_xy(test_df)

            for model_name in models.keys():
                if model_name in self.active_bundles:
                    bundle = self.active_bundles[model_name]
                    model_obj = bundle["model"]
                    pca_obj = bundle["pca"]

                    X_test_run = X_test_raw
                    if pca_obj:
                        X_test_run = pca_obj.transform(X_test_raw)

                    y_hat = model_obj.predict(X_test_run)

                    self.oos_predictions[model_name].loc[date_t, self.config.macro_var_name] = float(np.ravel(y_hat)[0])
                    self.oos_true.loc[date_t, self.config.macro_var_name] = float(np.ravel(y_test.values)[0])

                    if model_name in linear_models:
                        self.best_params_all_models_overtime[model_name].loc[date_t, "intercept"] = (
                            model_obj.model.intercept_ if hasattr(model_obj.model, "intercept_") else np.nan
                        )
                        self.best_params_all_models_overtime[model_name].loc[date_t, bundle["X_train_cols"]] = model_obj.model.coef_
        
        self._save_final_results_to_s3()
        logger.info("Rolling Window Estimation Scheme completed successfully.")

    def _get_train_validation_split(self, t: int, window_size: int, val_size: int):
        train_end_idx = t - val_size - self.forecast_horizon
        train_start_idx = max(0, train_end_idx - window_size)

        train_data = self.data.iloc[train_start_idx : train_end_idx]
        val_data = self.data.iloc[train_end_idx : t - self.forecast_horizon]
        val_end = self.date_range[t - self.forecast_horizon]

        return train_data, val_data, val_end

    def _save_final_results_to_s3(self):
        logger.info("Calcul des performances globales...")
        rmse_results = {}
        accuracy_results = {}

        for model_name, preds_df in self.oos_predictions.items():
            y_p = preds_df[self.config.macro_var_name]
            y_t = self.oos_true[self.config.macro_var_name]
            mask = y_p.notna() & y_t.notna()
            if mask.any():
                rmse_results[model_name] = np.sqrt(mean_squared_error(y_t[mask], y_p[mask]))
                accuracy_results[model_name] = (np.sign(y_t[mask]) == np.sign(y_p[mask])).mean()

        rmse_df = pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])
        accuracy_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])

        base_s3 = self.config.outputs_path + "/ml_model/rolling/"
        self.dm.aws.s3.upload(src=rmse_df, key=base_s3 + "oos_rmse_table.parquet")
        self.dm.aws.s3.upload(src=rmse_df, key=base_s3 + "oos_rmse_overtime.parquet")
        self.dm.aws.s3.upload(src=accuracy_df, key=base_s3 + "oos_sign_accuracy.parquet")
        self.dm.aws.s3.upload(src=self.oos_predictions, key=base_s3 + "oos_predictions.pkl")
        self.dm.aws.s3.upload(src=self.oos_true, key=base_s3 + "oos_true.parquet")
        self.dm.aws.s3.upload(src=self.best_hyperparams_all_models_overtime, key=base_s3 + "best_hyperparams_all_models_overtime.pkl")
        self.dm.aws.s3.upload(src=self.best_params_all_models_overtime, key=base_s3 + "best_params_all_models_overtime.pkl")
        self.dm.aws.s3.upload(src=self.best_score_all_models_overtime, key=base_s3 + "best_score_all_models_overtime.parquet")
        self.dm.aws.s3.upload(src=self.data, key=base_s3 + "data.parquet")

    def _load_models(self):
        paths = [
            self.config.outputs_path + "/ml_model/rolling/best_hyperparams_all_models_overtime.pkl",
            self.config.outputs_path + "/ml_model/rolling/best_params_all_models_overtime.pkl",
            self.config.outputs_path + "/ml_model/rolling/best_score_all_models_overtime.parquet",
            self.config.outputs_path + "/ml_model/rolling/oos_predictions.pkl",
            self.config.outputs_path + "/ml_model/rolling/oos_true.parquet",
            self.config.outputs_path + "/ml_model/rolling/data.parquet",
        ]
        loaded_obj_list = self.dm.aws.s3.load(key=paths)
        dct = {}
        for path, obj in zip(paths, loaded_obj_list):
            filename = path.split("/")[-1].split(".")[0]
            dct[filename] = obj
        return dct

    def _put_in_attributes(self, loaded_obj: dict):
        self.best_hyperparams_all_models_overtime = loaded_obj["best_hyperparams_all_models_overtime"]
        self.best_params_all_models_overtime = loaded_obj["best_params_all_models_overtime"]
        self.best_score_all_models_overtime = loaded_obj["best_score_all_models_overtime"]
        self.oos_predictions = loaded_obj["oos_predictions"]
        self.oos_true = loaded_obj["oos_true"]
        self.data = loaded_obj["data"]