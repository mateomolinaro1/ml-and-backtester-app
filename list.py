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
prefix_to_search = "outputs/"  # On définit le dossier cible

print(f"--- Liste des fichiers dans le dossier : {prefix_to_search} ---")

paginator = s3.get_paginator('list_objects_v2')

# On utilise 'Prefix' pour dire à S3 de ne regarder que ce dossier
operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix_to_search}

file_found = False

for page in paginator.paginate(**operation_parameters):
    if 'Contents' in page:
        for obj in page['Contents']:
            file_path = obj['Key']
            
            # On ignore l'objet qui représente le dossier lui-même
            if file_path == prefix_to_search:
                continue
            
            file_found = True
            # Taille en KB pour que ce soit lisible
            size = round(obj['Size'] / 1024, 2)
            print(f"📄 {file_path} ({size} KB)")

if not file_found:
    print(f"Aucun fichier trouvé dans le dossier '{prefix_to_search}'.")