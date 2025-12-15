# pytest tests/test_extract.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import requests
from dotenv import find_dotenv, load_dotenv
from app.extract import get_api
import logging

# Charger le .env
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Test API Transactions ===
test_API_URL = os.getenv("API_URL")

def test_extract():
    """
    Appel à l'API pour récupérer une transaction
    On vérifie que :
    - le code de statut est 200
    - le contenu est au format JSON
    - le JSON contient 'data'
    - la réponse contient les champs attendus
    """

    # test API's status
    response = requests.get(test_API_URL, timeout=60)
    assert response.status_code == 200, f"❌ API transactions renvoie {response.status_code}, attendu 200"

    # test API's response
    data_api = get_api()
    assert isinstance(data_api, dict), "❌ L'API n'a pas renvoyé un JSON"
    assert "data" in data_api, "❌ Le champ 'data' est absent de la réponse"
    assert len(data_api["data"]) > 0, "❌ Aucune transaction trouvée dans la réponse"

    logging.info("✅ Test API transactions OK : status 200 + JSON valide + fonction interne OK")




