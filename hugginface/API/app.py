from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import List
import os
import mlflow
#import mlflow.sklearn
import mlflow.pyfunc
#import boto3
from dotenv import find_dotenv, load_dotenv

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# === MLFlow ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
#MODEL_URI = os.getenv("MODEL_URI")
EXPERIMENT_NAME=os.getenv("EXPERIMENT_NAME")


def last_model_uri():
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    experiment_name = os.getenv('EXPERIMENT_NAME')

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if not experiment_name:
        raise ValueError("La variable d'environnement EXPERIMENT_NAME n'est pas définie.")
    else:
        print(f"Utilisation de l'expérience MLflow : {experiment_name}")

    # Récupération de l'ID de l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable.")

    experiment_id = experiment.experiment_id

    # Récupération du dernier run terminé
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],  # du plus récent au plus ancien
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"Aucun run terminé trouvé pour l'expérience '{experiment_name}'.")

    last_run_id = runs.iloc[0].run_id

    # Construction du chemin du modèle dans MLflow
    model_name = experiment_name  # adapte si besoin
    logged_model = f"runs:/{last_run_id}/{model_name}"

    # Chargement du modèle
    print(f"Chargement du modèle depuis MLflow : {logged_model}")
    return logged_model

def load_model():
    uri = last_model_uri()
    model = mlflow.pyfunc.load_model(uri)
    model.version = model.metadata.run_id
    return model

# # === LOAD MODEL from MLFlow function ===
# def load_mlflow_model(tracking_uri: str = MLFLOW_TRACKING_URI, model_uri: str = MODEL_URI):

#     mlflow.set_tracking_uri(tracking_uri)
#     model = mlflow.sklearn.load_model(model_uri)
#     return model


# === APP ===

app = FastAPI(
    title="Automatic Fraud Detector API",
    description="Fraud detection in real-time transactions",
    version="1.0.0")

# LOAD MODEL from MLFlow
#model = load_mlflow_model("/app/model")
#model = load_mlflow_model("fastAPIserver/API/app")
model = load_model()


class DataRow(BaseModel):
    cc_num: float
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: float
    lat: float
    long: float
    city_pop: float
    job: str
    dob: str
    trans_num: str
    merch_lat: float
    merch_long: float
    unix_time: float
    trans_date_trans_time: str

class Payload(BaseModel):
    data: List[DataRow]

@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "message": "🚀 Automatic Fraud Detector API is running...", 
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": str(type(model).__name__)
    }

@app.post("/predict")
def predict(payload: List[DataRow]):
    """Classification Endpoint"""
    try:
        df = pd.DataFrame([row.model_dump() for row in payload])
        preds = model.predict(df)
        #return {"predictions": preds.to_dict(orient='records')}
        return {"Classification": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
