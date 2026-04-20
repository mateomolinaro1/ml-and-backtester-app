import pandas as pd
from abc import ABC, abstractmethod
from better_aws import AWS
from dotenv import load_dotenv
import httpx
import io
import brotli

load_dotenv()
# --- CONFIGURATION ET LOGIQUE DE BASE ---

#Ici j'ai repris la configuration de l'exemple
aws = AWS(region="eu-north-1", verbose=True)
aws.s3.config(
    bucket="ml-and-backtester-app",
    output_type="pandas",      # tabular loads -> pandas (or "polars")
    file_type="parquet",       # default tabular format for dataframe uploads without extension
    overwrite=True,
)

class DataSource(ABC):
    """Classe abstraite définissant comment récupérer une donnée spécifique."""
    def __init__(self, url_base):
        self.url_base = url_base

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        pass

    def prepare_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise la colonne 'date' et la définit comme index."""
        cols = {c.lower(): c for c in df.columns}
        cols_to_drop = ["sasdate", "year", "month", "day",
                        "Year", "Month", "Day"]  # On prévoit les différentes capitalisations possibles
        # 1. Identification et création de la colonne 'date'
        if "date" in cols:
            df["date"] = pd.to_datetime(df[cols["date"]], errors="coerce")
        elif "sasdate" in cols:
            df["date"] = pd.to_datetime(df["sasdate"], format="%m/%d/%Y", errors="coerce")
        elif {"year", "month", "day"}.issubset(cols.keys()):
            df["date"] = pd.to_datetime(df[[cols["year"], cols["month"], cols["day"]]], errors="coerce")
        elif {"year", "month"}.issubset(cols.keys()):
            date_df = df[[cols["year"], cols["month"]]].copy()
            date_df.columns = ["year", "month"]
            date_df["day"] = 1
            df["date"] = pd.to_datetime(date_df, errors="coerce")
        
        if "News_Based_Policy_Uncert_Index" in df.columns:
            df.rename(columns={"News_Based_Policy_Uncert_Index": "monthly_epu_index"}, inplace=True)
        # 2. Nettoyage des lignes qui n'ont pas pu être converties (ex: texte de copyright en bas du CSV)
        if "date" in df.columns:
            df = df.dropna(subset=["date"])
            
            # 3. Passage en Index
            df = df.set_index("date")
            
            # 4. Tri par date (Essentiel pour le Rolling/Expanding)
            df = df.sort_index()
            to_remove = [c for c in cols_to_drop if c in df.columns]
            df = df.drop(columns=to_remove)
            
            
        return df

# --- IMPLÉMENTATIONS SPÉCIFIQUES ---

class ExcelEPU(DataSource):
    def fetch(self):
        df = pd.read_excel(self.url_base, sheet_name=0)
        return self.prepare_dates(df)

class DailyEPU(DataSource):
    def fetch(self) -> pd.DataFrame:
        
        try:
            # 2. Utilisation de httpx avec les paramètres de ton test réussi
            with httpx.Client(http2=True) as client:
                r = client.get(self.url_base, follow_redirects=True)
                #r.raise_for_status()
                

            # 3. Lecture avec gestion des erreurs de lignes
            df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip')
            
            # Nettoyage standard
            df.columns = [c.strip().lower() for c in df.columns]
            
            return self.prepare_dates(df)
            
        except Exception as e:
            print(f"💥 Échec critique : {e}")
            return None

class FredMD(DataSource):
    def fetch(self) -> pd.DataFrame:
        current_month = pd.Timestamp.now().to_period("M")
        url = f"{self.url_base}{current_month}-md.csv"
        
        try:
            # On utilise l'appel direct qui "marche nickel" pour toi
            # Mais on ajoute skiprows=[1] pour virer les codes de transformation (ligne 2)
            df = pd.read_csv(url, skiprows=[1])
            
            print(f"Succès : Fred-MD {current_month} chargé.")
            #return self.prepare_dates(df)
            return df
            
        except Exception as e:
            # Si le mois actuel (ex: Avril) n'existe pas, on tente le mois précédent
            last_month = (pd.Timestamp.now() - pd.DateOffset(months=1)).to_period("M")
            fallback_url = f"{self.url_base}{last_month}-md.csv"
            print(f"Mois actuel indisponible, tentative sur : {last_month}")
            
            try:
                df = pd.read_csv(fallback_url, skiprows=[1])
                #df = df.drop(columns = ['sasdate'])
                return self.prepare_dates(df)
                #return df
            except:
                print("Échec total de récupération Fred-MD.")
                return None

import io
import pandas as pd

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
        self.aws = aws

    def _load_from_s3(self):
        try:
            df_existing = self.aws.s3.load(self.s3_path)
            return self.strategy.prepare_dates(df_existing)
        except Exception as e:
            print(f"S3 empty or error: {e}")
            return None

    def _sync_and_update(self):
        df_new = self.strategy.fetch()
        df_existing = self._load_from_s3()

        if df_new is None:
            # Si pas de nouvelles données, on garde l'existant s'il existe
            self.df = df_existing
            

        if df_existing is None:
            # Premier run : on prend tout le nouveau DataFrame
            self.df = df_new
        else:
            # 1. On compare les index (dernière date sur S3 vs dates distantes)
            last_date = df_existing.index.max()
            
            # 2. On filtre df_new pour ne garder que ce qui est strictement après
            # Comme c'est un DatetimeIndex, la comparaison fonctionne nativement
            new_rows = df_new[df_new.index > last_date]
            
            if not new_rows.empty:
                # 3. Concaténation (Pandas aligne automatiquement sur l'index)
                self.df = pd.concat([df_existing, new_rows])
                
                # 4. Nettoyage final : tri et suppression d'éventuels doublons sur l'index
                self.df = self.df.sort_index()
                self.df = self.df[~self.df.index.duplicated(keep='first')]
            else:
                # Rien de neuf à ajouter
                self.df = df_existing
        
        return self.df


    def _save_to_s3(self):
        if self.df is not None:
            self.aws.s3.upload(self.df, self.s3_path)
        else:
            raise ValueError("No data to save.")
        
    def update(self):
        self._sync_and_update()
        self._save_to_s3()
        


