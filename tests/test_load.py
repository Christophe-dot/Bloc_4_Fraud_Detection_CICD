# pytest tests/test_load.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from app.load import (
    ensure_predictions_table_exists,
    build_db_rows,
    insert_predictions,
    pg_connect,
    DATABASE_URL)
import logging

# DATABASE_URL = os.getenv("BACKEND_STORE_URI")

def test_pg_connect():
    """
    Test très simple : la connexion doit s'établir sans erreur.
    """
    conn = pg_connect()
    assert conn is not None
    conn.close()
    logging.info("✅ Connexion à la base de données réussie")


def test_ensure_predictions_table_exists_runs():
    """
    Test très simple : la fonction ne doit pas lever d'erreur
    et la table doit exister après
    """

    ensure_predictions_table_exists()
    
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'transactions';
                """
            )
        )
        table = result.fetchone()
        assert table is not None, "❌ Il y a des erreurs dans la table"
        assert table[0] == "transactions", "❌ La table n'existe toujours pas"
        logging.info("✅ Aucune erreur dans la table et elle existe bien après le test")

def test_build_db_rows_basic():
    """
    Teste que build_db_rows renvoie bien une liste de tuples
    avec les bonnes valeurs et la bonne longueur.
    """   
    # Fake json
    fake_data = {
        "columns":[
            "cc_num",
            "merchant",
            "category",
            "amt",
            "first",
            "last",
            "gender",
            "street",
            "city",
            "state",
            "zip",
            "lat",
            "long",
            "city_pop",
            "job",
            "dob",
            "trans_num",
            "merch_lat",
            "merch_long",
            "is_fraud",
            "current_time"
            ],
        "data":[[
            999999,
            "TEST_bidon",
            "kids_pets",
            100.01,
            "Jenna",
            "Brooks",
            "F",
            "South Park",
            "Baton Rouge",
            "LA",
            99999,
            30.4066
            -91.494831,
            795,
            "Designer",
            "1977-02-22",
            "2584478c353ef455506cfe941468cc55",
            30.731498,
            -91.494831,
            0,
            1765483867831]]
            }

    # DataFrame de prédictions minimal
    pred_df = pd.DataFrame(
        [
            {
                "cc_num": 999999,
                "merchant" : "TEST_bidon",
                "category" : "kids_pets",
                "amt" : 100.01,
                "first" : "Jenna",
                "last" : "Brooks",
                "gender" : "F",
                "street" : "South Park",
                "city" : "Baton Rouge",
                "state" : "LA",
                "zip" : 99999,
                "lat" : 30.4066,
                "long" : -91.494831,
                "city_pop" : 7950,
                "job" : "Designer",
                "dob" : "1977-02-22",
                "trans_num" : "2584478c353ef455506cfe941468cc55",
                "merch_lat" : 30.731498,
                "merch_long" : -91.494831,
                "unix_time": 1765483867831,
                "trans_date_trans_time": "2025-12-12 18:00:00",
                "classification": 0.0
            }
        ]
    )

    rows = build_db_rows(data_api=fake_data, pred_df=pred_df)

    # On doit avoir exactement 1 ligne
    assert isinstance(rows, list)
    assert len(rows) == 1, "❌ Le dataframe doit contenir exactement 1 ligne"
    logging.info("✅ Le dataframe contient exactement 1 ligne")

    merchant = "TEST_bidon"
    category = "kids_pets"
    classification = 0

    row = rows[0]
    assert row[1] == merchant, "❌ La valeur 'merchant' est incorrecte"
    assert row[2] == category, "❌ La valeur 'category' est incorrecte"
    assert row[22] == classification, "❌ La valeur 'classification' est incorrecte"
    assert isinstance(row[1], str), "❌ La valeur 'merchant' doit être de type 'str'"
    assert isinstance(row[2], str), "❌ La valeur 'category' doit être de type 'str'"
    assert isinstance(row[22], float), "❌ La valeur 'classification' doit être de type 'float'"
    logging.info("✅ Les valeurs et formats attendus sont OK")

def test_insert_predictions_inserts_rows():
    """
    Test d'intégration simple :
    - on s'assure que la table existe
    - on insère une ligne "test"
    - on vérifie qu'elle est bien présente en base
    """
    ensure_predictions_table_exists()

    test_merchant = "TEST_FRAUD_PYTEST"

    # 1 seule ligne de test

    rows = [
        (
            213156747557083.0,                              # "cc_num"              
            test_merchant,                                  # TEST "merchant"
            "kids_pets",                                    # "category" 
            72.77,                                          # "amt" 
            "Adam",                                         # "first"
            "Santos",                                       # "last"                
            "M",                                            # "gender"
            "725 Jo Trace Apt. 102",                        # "street"
            "Glendale",                                     # "city"
            "CA",                                           # "state"
            91206.0,                                        # "zip"                          
            34.1556,                                        # "lat" 
            -118.2322,                                      # "long"
            172817.0,                                       # "city_pop"
            "Advertising account planner",                  # "job"
            "1982-07-30",                                   # "dob"
            "8a8bc54161fc0db49940c30f8fed1eea",             # "trans_num"
            34.60145,                                       # "merch_lat"
            -118.493708,                                    # "merch_long"
            0.0,                                            # "is_fraud"
            1765483557.381,                                 # "unix_time"
            "2025-12-11 20:05:57",                          # "trans_date_trans_time
            0.0                                             # classification
        )
    ]

    insert_predictions(rows)

    # Vérification en base
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT COUNT(*) 
                FROM public.transactions
                WHERE merchant = :merchant
                """
            ),            
            {"merchant": test_merchant},
        )
        count = result.scalar()
        assert count >= 1, "❌ La ligne test est absente de la base"
        logging.info("✅ La ligne test a bien été insérée dans la base")

