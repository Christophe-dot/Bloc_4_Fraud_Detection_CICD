import time
from run_pipeline import run_etl_once
import logging

# Log infos
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if __name__ == "__main__":

    #while True:
    for i in range(10) : # Limit to 10 iterations for testing
        try:
            run_etl_once()  # Apply complete ETL
        
        except Exception as e:
            logging.error(f"❌ Erreur API : pause de 20 secondes avant prochain appel !")
        # Wait for 20 secondes
        time.sleep(20)
