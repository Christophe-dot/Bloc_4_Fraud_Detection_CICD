# pytest tests/test_load_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.load_model import load_mlflow_model
import logging

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_load_mlflow_model():
    """
    Chargement du modèle MLflow
    On doit récupérer un objet Python 'not None'
    """

    model = load_mlflow_model()
    assert model is not None, "❌ Le modèle MLflow n'a pas été chargé"
    logging.info("✅ Modèle MLflow chargé")
