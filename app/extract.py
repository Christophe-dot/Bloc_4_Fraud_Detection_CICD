import os
import json
from datetime import datetime
import logging
import requests
import boto3
from dotenv import find_dotenv, load_dotenv

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === API Transactions ===
API_URL = os.getenv("API_URL")

# === S3 ===
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET = os.getenv("BUCKET_NAME")
RAW_PREFIX = "bloc4/data/raw"

boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# === EXTRACT function ===

## Connect API to get real_time (simulated) transactions
def get_api() -> dict:

    url = API_URL
    logging.info("🚀 Appel API lancé...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data_api = json.loads(response.text)
    data_api = json.loads(data_api)
    logging.info("✅ Transaction récupérée")
    return data_api

## Save raw data to S3 bucket in json
def save_data_api_to_s3(data_api: dict, timestamp: str) -> str:

    logging.info("🚀 Sauvegarde transaction RAW dans s3...")   
    # Convert raw data to JSON string
    json_data = json.dumps(data_api, ensure_ascii=False, indent=2).encode('utf-8')        
    # Create a unique filename with timestamp and RAW prefix
    raw_file = f"{RAW_PREFIX}/{timestamp}_transaction.json"
           
    # Upload to s3
    s3.put_object(
          Bucket=S3_BUCKET,
          Key=raw_file,
          Body=json_data,
          ContentType='application/json',
          ContentEncoding='utf-8')
    logging.info(f"✅ Transaction RAW enregistrée dans s3://{S3_BUCKET}/{raw_file}")

## Pipeline EXTRACT : get_transaction + save_data_api_to_s3
def extract() -> tuple[dict, str]:

    transaction = get_api()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_data_api_to_s3(transaction, timestamp)
    return transaction, timestamp
