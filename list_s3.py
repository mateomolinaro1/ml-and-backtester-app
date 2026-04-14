import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
)

bucket_name = "ml-and-backtester-app"

print(f"--- Arborescence complète des dossiers de {bucket_name} ---")

paginator = s3.get_paginator('list_objects_v2')
folders = set()

# On récupère tous les fichiers pour en déduire les dossiers
for page in paginator.paginate(Bucket=bucket_name):
    if 'Contents' in page:
        for obj in page['Contents']:
            path = obj['Key']
            if '/' in path:
                # On récupère toutes les étapes du chemin
                # Ex: 'models/v1/weights.h5' -> 'models/', 'models/v1/'
                parts = path.split('/')
                current_path = ""
                for i in range(len(parts) - 1):
                    current_path += parts[i] + "/"
                    folders.add(current_path)

# Trier et afficher proprement
for folder in sorted(folders):
    # Compter les slashes pour créer une indentation visuelle
    depth = folder.count('/') - 1
    indent = "  " * depth
    folder_name = folder.split('/')[-2]
    print(f"{indent}📁 {folder_name}/")

if not folders:
    print("Aucun dossier trouvé.")