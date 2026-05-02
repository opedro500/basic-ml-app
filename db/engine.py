import os
import logging

from pymongo import MongoClient
from dotenv import load_dotenv
from app.schema import PredictionResponse

load_dotenv()
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "basic_ml_app") 
ENV = os.getenv("ENV", "prod").lower()

# ==========================================
# CONEXÃO GLOBAL (CONNECTION POOLING)
# ==========================================
mongo_client = None

if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
    except Exception as e:
        logger.error(f"Erro ao conectar no MongoDB: {e}")

# ==========================================
# FUNÇÕES DE BANCO DE DADOS
# ==========================================
def get_mongo_collection(collection_name: str):
    """Retorna a coleção específica do banco de dados."""
    if mongo_client is None:
        raise ValueError("MONGO_URI não configurado. Banco de dados indisponível.")
    
    db = mongo_client[MONGO_DB]
    
    return db[collection_name]

def log_prediction(prediction_data: PredictionResponse) -> dict:
    """
    Insere um log de predição no banco de dados e retorna o
    documento inserido com o ID formatado para a resposta JSON.
    """
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    prediction_dict = prediction_data.model_dump()

    try:
        result = collection.insert_one(prediction_dict)
        
        prediction_dict["id"] = str(result.inserted_id)
        prediction_dict.pop("_id", None)
    except Exception as e:
        logger.error(f"Falha ao salvar log no banco de dados: {e}")
        raise Exception(f"Falha ao registrar predição no banco. Erro: {e}")

    return prediction_dict
