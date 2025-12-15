from dotenv import find_dotenv, load_dotenv
import os
import boto3
import logging
from extract import extract
from load_model import load_mlflow_model
from transform import (
    build_features_from_transaction,
    save_features_to_s3,
    predict_fraud,
    save_predictions_to_s3)
from load import (
    ensure_predictions_table_exists,
    build_db_rows,
    insert_predictions)

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === S3 ===
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET = os.getenv("BUCKET_NAME")


boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# === ETL function ===

## Apply complete ETL : Extract -> Transform -> Predict -> Load
def run_etl_once():

    logging.info("🚀 Démarrage du pipeline ETL")
    # EXTRACT
    data_api, timestamp = extract()

    # LOAD MODEL
    model = load_mlflow_model()

    # TRANSFORM
    features_df = build_features_from_transaction(data_api)
    save_features_to_s3(features_df, timestamp)

    # PREDICT
    pred_df = predict_fraud(model, features_df)
    save_predictions_to_s3(pred_df, timestamp)

    # LOAD MODEL to Database
    ensure_predictions_table_exists()
    rows = build_db_rows(
        data_api=data_api,
        pred_df=pred_df)
    
    insert_predictions(rows)
    logging.info("✅✅✅ Pipeline réalisé avec succès 💰💰💰")

if __name__ == "__main__":
    run_etl_once()