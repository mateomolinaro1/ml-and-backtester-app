import pandas as pd
from abc import ABC, abstractmethod
from src.ml_and_backtester_app.utils.s3_utils import s3Utils

# --- CONFIGURATION ET LOGIQUE DE BASE ---

class DataSource(ABC):
    """Classe abstraite définissant comment récupérer une donnée spécifique."""
    def __init__(self, url_base):
        self.url_base = url_base

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        pass

    def prepare_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise la colonne 'date' pour n'importe quel DF."""
        cols = {c.lower(): c for c in df.columns}
        
        if "date" in cols:
            df["date"] = pd.to_datetime(df[cols["date"]], errors="coerce")
        elif {"year", "month", "day"}.issubset(cols.keys()):
            df["date"] = pd.to_datetime(df[[cols["year"], cols["month"], cols["day"]]])
        elif {"year", "month"}.issubset(cols.keys()):
            # On renomme pour que to_datetime comprenne (nécessite year/month/day ou juste dict)
            date_df = df[[cols["year"], cols["month"]]].copy()
            date_df.columns = ["year", "month"]
            date_df["day"] = 1
            df["date"] = pd.to_datetime(date_df)
        
        return df

# --- IMPLÉMENTATIONS SPÉCIFIQUES ---

class ExcelEPU(DataSource):
    def fetch(self):
        df = pd.read_excel(self.url_base, sheet_name=0)
        return self.prepare_dates(df)

class DailyEPU(DataSource):
    def fetch(self):
        df = pd.read_csv(self.url_base)
        return self.prepare_dates(df)

class FredMD(DataSource):
    def fetch(self) -> pd.DataFrame:
        """
        Gestion de l'absence d'url pour le mois en cours.
        """
        current_month = pd.Timestamp.now().to_period("M")
        url = f"{self.url_base}{current_month}.csv"
        
        try:
            df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
            print(f"Succès : Données Fred-MD de {current_month} récupérées.")
            return self.prepare_dates(df)
        except Exception as e:
            # On log l'info, mais on ne lève pas d'erreur 
            print(f"Info : Le fichier {current_month} n'est pas encore en ligne sur Fred-MD. (Attendu à : {url})")
            return None

# --- LE PILOTE (ORCHESTRATEUR) ---

class S3PolicyDataUpdater:
    # Mapping des stratégies
    STRATEGIES = {
        "fred-md": FredMD("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/"),
        "daily-epu": DailyEPU("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"),
        "monthly-epu": ExcelEPU("https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"),
        "categorical-epu": ExcelEPU("https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx"),
    }

    def __init__(self, data_type: str, s3_path: str):
        if data_type not in self.STRATEGIES:
            raise ValueError(f"Type inconnu. Choix : {list(self.STRATEGIES.keys())}")
        
        self.strategy = self.STRATEGIES[data_type]
        self.s3_path = s3_path
        self.df = None

    def fetch_remote_data(self):
        try:
            return self.strategy.fetch()
        except Exception as e:
            print(f"Erreur lors de la récupération distante : {e}")
            return None

    def load_from_s3(self):
        try:
            df_existing = s3Utils.pull_parquet_file_from_s3(self.s3_path)
            return self.strategy.prepare_dates(df_existing)
        except Exception as e:
            print(f"S3 vide ou erreur : {e}")
            return None

    def sync_and_update(self):
        df_new = self.fetch_remote_data()
        df_existing = self.load_from_s3()

        if df_new is None:
            return

        if df_existing is None:
            self.df = df_new
        else:
            last_date = df_existing["date"].max()
            new_rows = df_new[df_new["date"] > last_date]
            
            if not new_rows.empty:
                self.df = pd.concat([df_existing, new_rows], ignore_index=True)
                self.df = self.df.sort_values("date").drop_duplicates().reset_index(drop=True)
            else:
                self.df = df_existing

    def save_to_s3(self):
        if self.df is not None:
            s3Utils.push_object_to_s3_parquet(self.df, self.s3_path)
        else:
            raise ValueError("Aucune donnée à sauvegarder.")