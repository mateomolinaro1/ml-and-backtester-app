from __future__ import annotations
from ml_and_backtester_app.data.data_manager import DataManager
from ml_and_backtester_app.utils.config import Config, logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

config = Config()
data = DataManager(config=config)

class FeaturesEngineering:
    def __init__(
            self,
            config: Config,
            data: DataManager
    ):
        self.config = config
        self.data = data

        self.transformed_fred_data = None
        self.y = None
        self.x = None

    @staticmethod
    def preprocess_var(var: pd.DataFrame, code_transfo: int | float):
        var = var.astype(float)

        if code_transfo == 1.0:
            return var  
        elif code_transfo == 2.0:
            return var.diff()
        elif code_transfo == 3.0:
            return var.diff().diff()
        elif code_transfo == 4.0:
            return np.log(var)
        elif code_transfo == 5.0:
            return np.log(var).diff()
        elif code_transfo == 6.0:
            return np.log(var).diff().diff()
        elif code_transfo == 7.0:
            return var.pct_change()
        else:
            raise ValueError(f"Unknown transformation code: {code_transfo}")

    def _prepare_base_fred(self):
        """Prépare et transforme les données FRED-MD."""
        transformed_data = {}
        for col in self.data.fred_data.columns:
            transformed_data[col] = self.preprocess_var(
                var=self.data.fred_data[col],
                code_transfo=self.data.code_transfo[col]
            )
        df = pd.DataFrame(transformed_data)
        df.index = self.data.fred_data.index
        return df

    def _integrate_sources(self):
        """Fusionne ou sélectionne les données en appliquant les transformations."""
        source = self.config.datasource

        if source == "fred_md":
            self.transformed_fred_data = self._prepare_base_fred()

        elif source == "daily_epu":
            df_daily = self.data.epu_data.copy()
            for col in df_daily.columns:
                if col not in self.data.code_transfo:
                    df_daily[col] = np.log(df_daily[col]).diff()
            self.transformed_fred_data = df_daily

        elif source in ["monthly_epu", "cat_epu"]:
            fred_df = self._prepare_base_fred()
            
            epu_df = self.data.epu_data.copy()
            for col in epu_df.columns:
                if col not in self.data.code_transfo:
                    epu_df[col] = np.log(epu_df[col]).diff()
            
            self.transformed_fred_data = pd.merge(
                epu_df, 
                fred_df, 
                left_index=True, 
                right_index=True, 
                how='inner'
            )
    
        return self.transformed_fred_data

    def _build_lagged_features(self):
        """Crée les lags pour la prévision"""
        lagged_vars = {}
        lags = [1, 5, 22, 63, 126, 252] if self.config.data_frequency == "daily" else [1, 3, 6, 12]
        
        for col in self.transformed_fred_data.columns:
            for lag in lags:
                lagged_vars[f"{col}_lag{lag}"] = self.transformed_fred_data[col].shift(lag)
        
        lagged_df = pd.DataFrame(lagged_vars)
        self.transformed_fred_data = pd.concat([self.transformed_fred_data, lagged_df], axis=1)

    def _split_y_x(self):
        """Sépare la cible (décalée de l'horizon) des features avec alignement."""
        target_name = self.config.macro_var_name
        horizon = self.config.forecast_horizon

        y_full = self.transformed_fred_data[[target_name]].shift(-horizon)
        x_full = self.transformed_fred_data.drop(columns=[target_name])
        combined = pd.concat([y_full, x_full], axis=1).dropna()
        self.y = combined[[target_name]].copy()
        self.x = combined.drop(columns=[target_name]).copy()
        
        logger.info(f"Split effectué. X shape: {self.x.shape}, y shape: {self.y.shape}")
        
        # Libération mémoire
        self.transformed_fred_data = None

    def _handle_missing_values(self, threshold: float = 0.10):
        """Gère les valeurs manquantes en supprimant les colonnes
        avec un taux de NA supérieur au seuil."""
        # On calcule le ratio de NA par colonne
        na_ratio = self.transformed_fred_data.isna().mean()
        
        # On identifie les colonnes à supprimer
        cols_to_drop = na_ratio[na_ratio > threshold].index.tolist()

        if cols_to_drop:
            self.transformed_fred_data = self.transformed_fred_data.drop(columns=cols_to_drop)
        
        self.transformed_fred_data = self.transformed_fred_data.dropna()
        
        return cols_to_drop

    def _crop_date_range(self):
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        if not pd.isna(start_date) and not pd.isna(end_date):
            self.transformed_fred_data = self.transformed_fred_data.loc[
                (self.transformed_fred_data.index >= start_date) &
                (self.transformed_fred_data.index <= end_date)
            ,:]
        if not pd.isna(start_date):
            self.transformed_fred_data = self.transformed_fred_data.loc[
                self.transformed_fred_data.index >= start_date
            ,:]
        if not pd.isna(end_date):
            self.transformed_fred_data = self.transformed_fred_data.loc[
                self.transformed_fred_data.index <= end_date
            ,:]

    def get_features(self):
        """Pipeline d'assemblage"""
        self._integrate_sources()
        self._build_lagged_features()
        self._crop_date_range()
        self._handle_missing_values()
        self._split_y_x()
        return self.x, self.y


class StandardScaling:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, x: pd.DataFrame):
        x_scaled = self.scaler.fit_transform(x)
        return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_scaled = self.scaler.transform(x)
        return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)

