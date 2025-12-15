import os
import mlflow
import mlflow.sklearn
from dotenv import find_dotenv, load_dotenv
import logging

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === MLFlow ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
#MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
MODEL_URI = os.getenv("MODEL_URI")


# === LOAD MODEL from MLFlow function ===
def load_mlflow_model(tracking_uri: str = MLFLOW_TRACKING_URI, model_uri: str = MODEL_URI):
    
    logging.info(f"🚀 Chargement du modèle MLflow depuis {tracking_uri} avec le modèle URI {model_uri}...")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)
    logging.info("✅ Modèle récupéré depuis MLflow")
    return model

