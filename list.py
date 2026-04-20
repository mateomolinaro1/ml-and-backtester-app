import os
from dotenv import load_dotenv
from ml_and_backtester_app.utils.config import Config
from ml_and_backtester_app.data.data_manager import DataManager

# 1. Charger les clés AWS depuis le fichier .env
load_dotenv()

# 2. Initialiser la config et le manager
config = Config()
dm = DataManager(config=config)

# 3. Charger les colonnes du fichier S3
# On utilise le s3_object_name défini dans tes scripts précédents
# Change juste cette ligne dans ton script précédent
s3_key = "data/wrds_funda_gross_query.parquet"

# Relance le script pour voir la liste des colonnes

print(f"Connexion à S3 (Bucket: {config.aws_bucket_name})...")

try:
    # On charge juste les métadonnées ou les premières lignes pour aller vite
    df = dm.aws.s3.load(key=s3_key)
    print("\n--- LISTE DES COLONNES DISPONIBLES ---")
    print(df.columns.tolist())
except Exception as e:
    print(f"Erreur lors de la lecture du fichier : {e}")