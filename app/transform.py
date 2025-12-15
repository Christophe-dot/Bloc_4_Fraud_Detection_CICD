import os
from datetime import datetime
import pandas as pd
import boto3
from dotenv import find_dotenv, load_dotenv
import logging

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === S3 ===
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET = os.getenv("BUCKET_NAME")
SILVER_PREFIX = "bloc4/data/silver"
GOLD_PREFIX = "bloc4/data/gold"

boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# === TRANSFORM function ===

## API's response to Dataframe (after transformation)
def build_features_from_transaction(data_api: dict) -> pd.DataFrame:

    # Convert API's response to Dataframe
    data_api_transaction = data_api['data']
    data_api_columns = data_api['columns']

    features = pd.DataFrame(data_api_transaction, columns=data_api_columns)
 
   # Preprocess data as expected by the model
    features['unix_time'] = features['current_time']/1000
    features['trans_date_trans_time'] = pd.to_datetime(features['current_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    features.drop(columns=['current_time'], inplace=True)
    features = features.astype({col: "float64" for col in features.select_dtypes(include=["int"]).columns})
    features = features.drop(columns=['is_fraud'])

    return features

## Save data (transformed) as SILVER into S3 
def save_features_to_s3(features_df: pd.DataFrame, timestamp: str) -> str:

    # Convert data (transformed) to CSV string
    csv_silver = features_df.to_csv(index=False)
    
    # Create filename identified with timestamp and SILVER_PREFIX
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    silver_file = f"{SILVER_PREFIX}/{timestamp}_transaction.csv" 
    
    # Upload to S3
    logging.info("🚀 Sauvegarde transaction SILVER dans s3...")
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=silver_file,
            Body=csv_silver,
            ContentType='application/csv')
        logging.info(f"✅ Transaction SILVER enregistrée dans s3://{S3_BUCKET}/{silver_file}")
        return silver_file
    
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'enregistrement SILVER sur S3: {e}")
        raise e
    
## Launch detection model (result within a dataframe)
def predict_fraud(model, features: pd.DataFrame) -> pd.DataFrame:

    preds = model.predict(features)
    result = features.copy()
    result["classification"] = preds

    # Real-time alerting by mail (not configured)
    if (preds == 1):
        print("************** Fraud detected! *****************")
        print("**** Mail sent to mister.blabla@gmail.com ******")

    return result

## Save transaction with classification (GOLD) to csv file
def save_predictions_to_s3(pred_df: pd.DataFrame, timestamp: str) -> str:

    # Convert predicted data to CSV string
    csv_gold = pred_df.to_csv(index=False)

    # Create filename identified with timestamp and GOLD_PREFIX
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gold_file = f'{GOLD_PREFIX}/{timestamp}_transaction.csv'
      
    # Upload to S3
    logging.info("🚀 Sauvegarde transaction GOLD dans s3...")
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=gold_file,
            Body=csv_gold,
            ContentType='application/csv')
        logging.info(f"✅ Transaction GOLD enregistrée dans s3://{S3_BUCKET}/{gold_file}")
        return gold_file
    
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'enregistrement GOLD sur S3: {e}")
        raise e

