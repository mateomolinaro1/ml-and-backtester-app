import boto3
import os
from dotenv import load_dotenv

# Charge tes identifiants depuis le .env
load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
)

bucket_name = "ml-and-backtester-app"
dossier_a_vider = "models/"  # Le '/' final est très important !

print(f"--- Suppression du contenu de '{dossier_a_vider}' dans {bucket_name} ---")

paginator = s3.get_paginator('list_objects_v2')

# On ajoute Prefix=dossier_a_vider pour ne cibler QUE ce dossier
pages = paginator.paginate(Bucket=bucket_name, Prefix=dossier_a_vider)

fichiers_supprimes_total = 0

for page in pages:
    if 'Contents' in page:
        # Préparer la liste des fichiers au format requis par delete_objects
        objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
        
        # S3 permet de supprimer plusieurs fichiers d'un coup (jusqu'à 1000)
        response = s3.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects_to_delete,
                'Quiet': False  # False permet de recevoir la confirmation de ce qui a été supprimé
            }
        )
        
        # Afficher les confirmations
        if 'Deleted' in response:
            for deleted_obj in response['Deleted']:
                print(f"🗑️ Supprimé : {deleted_obj['Key']}")
                fichiers_supprimes_total += 1
                
        # Gérer les éventuelles erreurs de suppression
        if 'Errors' in response:
            for err in response['Errors']:
                print(f"❌ Erreur sur {err['Key']} : {err['Message']}")

if fichiers_supprimes_total == 0:
    print(f"Aucun fichier trouvé dans '{dossier_a_vider}'. Le dossier est déjà vide ou n'existe pas.")
else:
    print(f"✅ Terminé ! {fichiers_supprimes_total} fichier(s) supprimé(s) avec succès.")