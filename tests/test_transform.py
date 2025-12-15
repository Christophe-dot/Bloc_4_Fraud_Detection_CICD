# pytest tests/test_transform.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.transform import build_features_from_transaction, predict_fraud
from app.load_model import load_mlflow_model
import logging

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

columns = [
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
  ]

index = [438972]

data = [
    [
      4653879239169997,
      "fraud_Morissette-Schaefer",
      "personal_care",
      50.29,
      "Monica",
      "Tucker",
      "F",
      "302 Christina Islands",
      "Smiths Grove",
      "KY",
      42171,
      37.0581,
      -86.1938,
      6841,
      "Therapist, sports",
      "1999-06-06",
      "0cb2d42eabbf90673d5578aeb79b79f8",
      36.708164,
      -86.996368,
      0,
      1765317289296
    ]
  ]


def test_build_features_from_transaction():
    """
    Test simple : vérifier que la fonction produit 1 ligne
    et contient la colonne cc_num
    """

    test_transaction = {"data": data, "index": index, "columns" : columns}

    df = build_features_from_transaction(test_transaction)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1, "❌ La fonction 'build_features' doit produire 1 ligne"
    assert "merchant" in df.columns, "❌ la colonne 'merchant' doit exister"
    assert df.loc[0, "amt"] == 50.29, "❌ la valeur 'amt' est incorrecte"
    logging.info("✅ La fonction 'build_features' est OK")


def test_predict_fraud():
    """
    Test simple :
    - classification doit être 0 ou 1
    """

    model = load_mlflow_model()

    # Fabriquer des features minimales valides
    features = pd.DataFrame(
        [
            {
                "cc_num": 60416207185,
                "merchant": "fraud_Schmeler-Howe",
                "category": "personal_care",
                "amt": 17.94,
                "first": "Mary",
                "last": "Diaz",
                "gender": "F",
                "street": "9886 Anita Drive",
                "city": "Fort Washakie",
                "state": "WY",
                "zip": 82514,
                "lat": 43.0048,
                "long": -108.8964,
                "city_pop": 1645,
                "job": "Information systems manager",
                "dob": "1986-02-17",
                "trans_num": "ee94a194b2f6f00096223809b8310b1c",
                "merch_lat": 42.340281,
                "merch_long": -109.460051,
                "is_fraud": 0.0,
                "unix_time": 1765352050.905,
                "trans_date_trans_time": "2025-12-10 00:24:14" 
            }
        ]
    )

    pred_df = predict_fraud(model, features)
    pred = pred_df.loc[0, "classification"]

    assert pred in [0., 1.], f"❌ La valeur 'is_fraud' doit être 0.0 ou 1.0, obtenu {pred}"
    logging.info("✅ La fonction 'predict_fraud' est OK")


