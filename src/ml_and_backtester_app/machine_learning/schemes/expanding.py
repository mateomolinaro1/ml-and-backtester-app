from __future__ import annotations
import time
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from typing import Dict, Type
from .base import EstimationScheme
import logging
import mlflow
import mlflow.sklearn
from ml_and_backtester_app.machine_learning.models import Model
from ml_and_backtester_app.machine_learning.features_selection import PCAFactorExtractor

import os

logger = logging.getLogger(__name__)

tmp_dir = os.path.join(os.getcwd(), "tmp") 
os.makedirs(tmp_dir, exist_ok=True)
logger.info(f"Dossier temporaire configuré : {tmp_dir}")

class ExpandingWindowScheme(EstimationScheme):
    # pass because already inherited

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
        elif self.config.load_or_train_models == "train":
            pass
        else:
            raise ValueError(
                "config.load_or_train_models must be either 'load' or 'train'"
            )

        logger.info("Starting Expanding Window Estimation Scheme...")
        # -----------------------------
        # Hyperparams combinations
        # -----------------------------
        hyperparams_all_combinations = self.build_hyperparams_combinations(
            hyperparameters_grid=hyperparams_grid
        )

        # -----------------------------
        # STORE BETAS OVER TIME (linear only)
        # -----------------------------
        linear_models = []
        for model_name, model in models.items():
            if model_name in ["ridge","lasso","elastic_net","ols",
                              "ridge_pca","lasso_pca","elastic_net_pca","ols_pca"]:
                linear_models.append(model_name)

        columns_list = ["intercept"] + list(
                        self.data.drop(columns=[self.config.macro_var_name]).columns
                    )
        self.best_params_all_models_overtime = {
            m: (
                pd.DataFrame(
                    index=self.date_range,
                    columns=columns_list,
                    dtype=float,
                )
                if m in linear_models
                else None
            )
            for m in models
        }
        # Add pca models if needed
        if self.config.with_pca:
            columns_list = ["intercept"] + [f"factor_{i + 1}" for i in range(self.config.nb_pca_components)]
            for model_name, _ in models.items():
                if model_name in ["ridge_pca","lasso_pca","elastic_net_pca","ols_pca"]:
                    self.best_params_all_models_overtime[model_name] = pd.DataFrame(
                        index=self.date_range,
                        columns=columns_list,
                        dtype=float,
                    )
    

        # -----------------------------
        # STORAGE
        # -----------------------------
        self.oos_predictions = {m: pd.DataFrame(
            index=self.date_range,
            data=np.nan,
            columns=[self.config.macro_var_name]
        ) for m in models}
        self.best_score_all_models_overtime = pd.DataFrame(
            index=self.date_range,
            columns=list(models.keys()),
        )
        self.best_hyperparams_all_models_overtime = {
            m: pd.DataFrame(
                index=self.date_range,
                columns=list(hyperparams_all_combinations[m][0].keys()),
            )
            for m in models
        }
        # Stores the last trained model instance per model_name (used for MLflow logging)
        self._last_trained_models: Dict[str, object] = {}

        # -----------------------------
        # WALK-FORWARD LOOP
        # -----------------------------
        start_idx = (
            self.min_nb_periods_required
            + self.validation_window
            + self.forecast_horizon
        )

        for t in range(start_idx, len(self.date_range) - self.forecast_horizon):
            date_t = self.date_range[t]
            logger.info(
                f"Training models for date {date_t} "
                f"({t}/{len(self.date_range) - self.forecast_horizon - 1})"
            )

            t0 = time.time()

            train_data, val_data, val_end = self._get_train_validation_split(t)
            X_train, y_train = self._split_xy(train_data)

            for model_name, ModelClass in models.items():
                logger.info(f"Model: {model_name}")

                # Apply PCA if needed
                if model_name.endswith("_pca"):
                    pca_extractor = PCAFactorExtractor(
                        n_factors=self.config.nb_pca_components
                    )
                    X_train = pca_extractor.fit_transform(X_train)
                    val_x = val_data.drop(columns=[self.config.macro_var_name])
                    val_x = pca_extractor.fit_transform(val_x)
                    val_data = pd.concat([val_x, val_data[[self.config.macro_var_name]]], axis=1)

                best_score = -np.inf
                best_hyperparams = None

                # -----------------------------
                # HYPERPARAMETER SEARCH
                # -----------------------------
                def evaluate(hyperparams:dict):
                    model = ModelClass(**hyperparams)
                    model.fit(X_train, y_train)

                    ICs = []
                    for d in np.sort(val_data.index.unique()):
                        val_d = val_data[val_data.index == d]
                        X_val, y_val = self._split_xy(val_d)

                        if len(y_val) <= 0:
                            continue

                        y_hat = model.predict(X_val)
                        ic = np.sqrt(mean_squared_error(y_val, y_hat))
                        if not np.isnan(ic):
                            ICs.append(ic)

                    if not ICs:
                        return None

                    return np.mean(ICs), hyperparams

                # =========================
                # PARALLEL GRID SEARCH
                # =========================
                results = Parallel(n_jobs=1)(
                    delayed(evaluate)(hp)
                    for hp in hyperparams_all_combinations[model_name]
                )

                for res in results:
                    if res is None:
                        continue
                    score, hp = res
                    if score > best_score:
                        best_score = score
                        best_hyperparams = hp

                # -----------------------------
                # STORE VALIDATION RESULTS
                # -----------------------------
                self.best_score_all_models_overtime.loc[date_t, model_name] = best_score
                if best_hyperparams is not None:
                    for k, v in best_hyperparams.items():
                        self.best_hyperparams_all_models_overtime[model_name].loc[date_t, k] = v

                # -----------------------------
                # FINAL TRAIN
                # -----------------------------
                full_train = self.data[self.data.index <= val_end]
                X_full, y_full = self._split_xy(full_train)
                if model_name.endswith("_pca"):
                    X_full = pca_extractor.fit_transform(X_full)

                model_final = ModelClass(**best_hyperparams)
                model_final.fit(X_full, y_full)
                self._last_trained_models[model_name] = model_final

                model_filename = f"{model_name}_{date_t.strftime('%Y-%m-%d')}.pkl"
                local_path = os.path.join(tmp_dir, model_filename)
                joblib.dump(model_final, local_path)
                s3_key = f"models/expanding/{model_name}/{model_filename}"
                self.dm.aws.s3.upload(local_path, key=s3_key)
                os.remove(local_path) # Supprime le fichier local après l'envoi réussi

                test_date = self.date_range[t]
                test_df = self.data[self.data.index == test_date]
                X_test, y_test = self._split_xy(test_df)
                if model_name.endswith("_pca"):
                    X_test = pca_extractor.transform(X_test)

                y_hat = model_final.predict(X_test)

                self.oos_predictions[model_name].loc[test_date, self.config.macro_var_name] = float(np.ravel(y_hat)[0])
                self.oos_true.loc[test_date, self.config.macro_var_name] = float(np.ravel(y_test.values)[0])

                logger.info(f"{model_name} done in {round((time.time() - t0) / 60, 3)} min")

                # -----------------------------
                # STORE COEFFICIENTS (linear models only)
                # -----------------------------
                if model_name in linear_models:
                    self.best_params_all_models_overtime[model_name].loc[date_t, "intercept"] = (
                        model_final.model.intercept_
                        if hasattr(model_final.model, "intercept_")
                        else np.nan
                    )

                    self.best_params_all_models_overtime[model_name].loc[
                        date_t,
                        X_train.columns
                    ] = model_final.model.coef_

                logger.info(
                    f"{model_name} done in "
                    f"{round((time.time() - t0) / 60, 3)} min"
                )

        logger.info("Calcul des performances globales...")

        rmse_results = {}
        accuracy_results = {}

        # On parcourt chaque modèle pour calculer son score global
        for model_name, preds_df in self.oos_predictions.items():
            y_p = preds_df[self.config.macro_var_name]
            y_t = self.oos_true[self.config.macro_var_name]
            
            # On enlève les dates vides pour ne pas fausser le calcul
            mask = y_p.notna() & y_t.notna()

            if mask.any():
                # On calcule l'erreur moyenne (RMSE) -> remplit le graph 1
                rmse = np.sqrt(mean_squared_error(y_t[mask], y_p[mask]))
                rmse_results[model_name] = rmse
                
                # On calcule la précision du signe -> remplit le graph 2
                correct_sign = np.sign(y_t[mask]) == np.sign(y_p[mask])
                accuracy_results[model_name] = correct_sign.mean()

        # On transforme ces chiffres en jolis tableaux (DataFrames)
        rmse_df = pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])
        accuracy_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])

        # -----------------------------
        # MLFLOW TRACKING
        # -----------------------------
        mlflow.set_experiment("prévision_expanding")
   
        logger.info("Logging les résultats dans MLflow...")

        for model_name in rmse_results:
            last_hp_df = self.best_hyperparams_all_models_overtime[model_name].dropna(how="all")
            last_hyperparams = last_hp_df.iloc[-1].dropna().to_dict() if not last_hp_df.empty else {}

            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("estimation_method", "expanding")
                mlflow.log_param("forecast_horizon", self.forecast_horizon)
                mlflow.log_param("validation_window", self.validation_window)
                mlflow.log_params({k: v for k, v in last_hyperparams.items()})

                mlflow.log_metric("oos_rmse", rmse_results[model_name])
                mlflow.log_metric("sign_accuracy", float(accuracy_results[model_name]))

                model_obj = self._last_trained_models.get(model_name)
                if model_obj is not None and hasattr(model_obj, "model") and hasattr(model_obj.model, "predict"):
                    try:
                        mlflow.sklearn.log_model(model_obj.model, name="model")
                    except Exception as e:
                        logger.warning(f"Impossible de logguer le modèle sklearn pour {model_name}: {e}")

                logger.info(
                    f"MLflow — {model_name}: OOS RMSE={rmse_results[model_name]:.4f}, "
                    f"Sign accuracy={float(accuracy_results[model_name]):.2%}"
                )

        # -----------------------------
        # MLFLOW MODEL REGISTRY
        # -----------------------------
        best_model_name = min(rmse_results, key=rmse_results.get)
        logger.info(f"Meilleur modèle : {best_model_name} (OOS RMSE={rmse_results[best_model_name]:.4f})")

        best_model_obj = self._last_trained_models.get(best_model_name)
        if best_model_obj is not None and hasattr(best_model_obj, "model"):
            try:
                with mlflow.start_run(run_name=f"registry_{best_model_name}"):
                    mlflow.log_param("best_model", best_model_name)
                    mlflow.log_metric("oos_rmse", rmse_results[best_model_name])
                    mlflow.log_metric("sign_accuracy", float(accuracy_results[best_model_name]))
                    mlflow.sklearn.log_model(
                        best_model_obj.model,
                        name="model",
                        registered_model_name="forecasting_best_model"
                    )

                client = mlflow.MlflowClient()
                latest = client.get_latest_versions("forecasting_best_model")[0]
                client.set_registered_model_alias(
                    name="forecasting_best_model",
                    alias="production",
                    version=latest.version
                )
                logger.info(f"Modèle '{best_model_name}' enregistré en production (version {latest.version})")
            except Exception as e:
                logger.warning(f"Model Registry non disponible, skipping: {e}")

        # --- ENVOI SUR S3 DANS LE BON DOSSIER ---
        # --- 5. ENVOI FINAL SUR S3 (VERSION COMPLÈTE POUR LE DASHBOARD) ---
        base_s3 = self.config.outputs_path + "/ml_model/expanding/"
        logger.info(f"Envoi des résultats finaux vers {base_s3}...")

        # A. Les indispensables (Le HAUT du dashboard)
        # On envoie sous deux noms pour être CERTAIN que le dashboard en trouve un
        self.dm.aws.s3.upload(src=rmse_df, key=base_s3 + "oos_rmse_table.parquet")
        self.dm.aws.s3.upload(src=rmse_df, key=base_s3 + "oos_rmse_overtime.parquet")
        self.dm.aws.s3.upload(src=accuracy_df, key=base_s3 + "oos_sign_accuracy.parquet")

        # B. Les prédictions et la réalité
        self.dm.aws.s3.upload(src=self.oos_predictions, key=base_s3 + "oos_predictions.pkl")
        self.dm.aws.s3.upload(src=self.oos_true, key=base_s3 + "oos_true.parquet")

        # C. Les détails techniques (Le BAS du dashboard - Ne pas les oublier !)
        self.dm.aws.s3.upload(src=self.best_hyperparams_all_models_overtime, key=base_s3 + "best_hyperparams_all_models_overtime.pkl")
        self.dm.aws.s3.upload(src=self.best_params_all_models_overtime, key=base_s3 + "best_params_all_models_overtime.pkl")
        self.dm.aws.s3.upload(src=self.best_score_all_models_overtime, key=base_s3 + "best_score_all_models_overtime.parquet")

        # D. Les données sources
        self.dm.aws.s3.upload(src=self.data, key=base_s3 + "data.parquet")

        logger.info("Pipeline terminé ! Toutes les données du Dashboard ont été écrasées avec tes résultats.")
    

    # -----------------------------
    # SPLITS
    # -----------------------------
    def _get_train_validation_split(self, t: int):
        train_end = self.date_range[t - self.validation_window - self.forecast_horizon]
        val_end = self.date_range[t - self.forecast_horizon]

        train_data = self.data[self.data.index <= train_end]
        val_data = self.data[
            (self.data.index > train_end)
            & (self.data.index <= val_end)
        ]

        return train_data, val_data, val_end

    def _load_models(self):
        paths = [
            self.config.outputs_path + "/ml_model/expanding" + "/best_hyperparams_all_models_overtime.pkl",
            self.config.outputs_path + "/ml_model/expanding" + "/best_params_all_models_overtime.pkl",
            self.config.outputs_path + "/ml_model/expanding" + "/best_score_all_models_overtime.parquet",
            self.config.outputs_path + "/ml_model/expanding" + "/oos_predictions.pkl",
            self.config.outputs_path + "/ml_model/expanding" + "/oos_true.parquet",
            self.config.outputs_path + "/ml_model/expanding" + "/data.parquet",
            # self.config.outputs_path + "/ml_model/expanding" + "/x.parquet",
            # self.config.outputs_path + "/ml_model/expanding" + "/y.parquet",
        ]

        loaded_obj_list = self.dm.aws.s3.load(
            key=paths
        )
        # Transforming list to dict with the filename as key
        dct = {}
        for path, obj in zip(paths, loaded_obj_list):
            filename = path.split("/")[-1].split(".")[0]
            dct[filename] = obj

        return dct

    def _put_in_attributes(self, loaded_obj: dict):
        self.best_hyperparams_all_models_overtime = loaded_obj[
            "best_hyperparams_all_models_overtime"
        ]
        self.best_params_all_models_overtime = loaded_obj[
            "best_params_all_models_overtime"
        ]
        self.best_score_all_models_overtime = loaded_obj[
            "best_score_all_models_overtime"
        ]
        self.oos_predictions = loaded_obj["oos_predictions"]
        self.oos_true = loaded_obj["oos_true"]
        self.data = loaded_obj["data"]