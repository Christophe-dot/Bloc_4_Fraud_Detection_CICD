import os
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import logging

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === LOAD function ===

## === NEON PostgreSQL ===
DATABASE_URL = os.getenv("BACKEND_STORE_URI")

## Connexion NEON PostgreSQL
def pg_connect():

    return psycopg2.connect(DATABASE_URL)

## Create table if it doesn't exist
def ensure_predictions_table_exists():
    
    engine = create_engine(DATABASE_URL)

    ddl_fraud_pred = """
    CREATE TABLE IF NOT EXISTS public.transactions (
        cc_num NUMERIC,
        merchant VARCHAR,
        category VARCHAR,
        amt NUMERIC,
        first VARCHAR,
        last VARCHAR,
        gender VARCHAR,
        street VARCHAR,
        city VARCHAR,
        state VARCHAR,
        zip NUMERIC,
        lat NUMERIC,
        long NUMERIC,
        city_pop NUMERIC,
        job VARCHAR,
        dob VARCHAR,        
        trans_num VARCHAR,        
        merch_lat NUMERIC,
        merch_long NUMERIC,        
        is_fraud NUMERIC,
        unix_time NUMERIC,
        trans_date_trans_time VARCHAR,
        classification NUMERIC
    );
    """

    with engine.begin() as conn:
        conn.execute(text(ddl_fraud_pred))

## Build list of tuples ready to be inserted into database
def build_db_rows(data_api: dict, pred_df: pd.DataFrame):

    ## Find index of 'is_fraud' column and get value
    index_is_fraud = data_api['columns'].index('is_fraud')
    is_fraud = data_api['data'][0][index_is_fraud]
 
    rows = []
    for _, row in pred_df.iterrows():
        rows.append(
            (
                float(row["cc_num"]),
                str(row["merchant"]),
                str(row["category"]),
                float(row["amt"]),
                str(row["first"]),
                str(row["last"]),
                str(row["gender"]),
                str(row["street"]), 
                str(row["city"]),  
                str(row["state"]),                                             
                float(row["zip"]),
                float(row["lat"]),
                float(row["long"]),
                float(row["city_pop"]),
                str(row["job"]),
                str(row["dob"]),
                str(row["trans_num"]),
                float(row["merch_lat"]),
                float(row["merch_long"]),
                float(is_fraud),                       
                float(row["unix_time"]),
                str(row["trans_date_trans_time"]),                              
                float(row["classification"])
            )
        )
    return rows

## Insert predictions into database 'transactions'
def insert_predictions(rows):

    logging.info(f"🚀 Insertion de prévisions en base de données...")

    insert_sql = """
    INSERT INTO public.transactions
    (cc_num,merchant,category,amt,first,last,gender,street,city,state,zip,lat,long,
    city_pop,job,dob,trans_num,merch_lat,merch_long,is_fraud,unix_time,
    trans_date_trans_time,classification)
    VALUES %s;
    """
    with pg_connect() as conn, conn.cursor() as cur:
        if rows:
            execute_values(cur, insert_sql, rows)
        conn.commit()
    logging.info(f"✅ Transaction écrite dans la base de données")

 
