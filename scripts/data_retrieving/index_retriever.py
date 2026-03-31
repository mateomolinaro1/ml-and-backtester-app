import pandas as pd
import requests
import io

class S3PolicyDataUpdater:
    def __init__(self, url):
        self.url = url

    def fetch_remote_csv(self):
        """Récupère le CSV source sur le web."""
        try:
            df_new = pd.read_csv(self.url)
            df_new['date'] = pd.to_datetime(df_new[['year', 'month', 'day']])
            return df_new
        except Exception as e:
            print(f"Erreur téléchargement source : {e}")
            return None

    def load_from_s3(self):
        """Charge le fichier existant depuis S3.
        A coder car je n'ai pas la syntaxe pour le faire."""

    def sync_and_update(self):
        """Compare la source web et le stockage S3."""
        df_new = self.fetch_remote_csv()
        df_existing = self.load_from_s3()

        if df_new is None: return

        if df_existing is None:
            self.df = df_new
        else:
            if 'date' not in df_existing.columns:
                df_existing['date'] = pd.to_datetime(df_existing[['year', 'month', 'day']])
            
            last_date_s3 = df_existing['date'].max()
            new_rows = df_new[df_new['date'] > last_date_s3]

            if not new_rows.empty:
                self.df = pd.concat([df_existing, new_rows], ignore_index=True)
                self.df = self.df.sort_values('date').reset_index(drop=True)
            else:
                self.df = df_existing

    def save_to_s3(self):
        """Sauvegarde le DataFrame mis à jour sur S3.
        A coder car je n'ai pas la syntaxe pour le faire."""

