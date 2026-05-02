import logging

from typing import Dict
from datetime import datetime, timezone
from intent_classifier import IntentClassifier
from db.engine import log_prediction
from app.schema import SinglePrediction, PredictionResponse

logger = logging.getLogger(__name__)

def load_all_classifiers(models_to_load_str: str) -> dict:
    """
    Carrega todos os modelos de ML especificados na variável de ambiente
    WANDB_MODELS a partir do registro do Weights & Biases.
    """
    loaded_models = {}
    model_urls = [url.strip() for url in models_to_load_str.split(',') if url.strip()]
    logger.info(f"Carregando {len(model_urls)} modelo(s) do W&B...")

    for url in model_urls:
        try:
            model_name = url.split('/')[-1].split(':')[0]

            logger.info(f"Carregando modelo: '{model_name}' (de {url})")
            loaded_models[model_name] = IntentClassifier(load_model=url)
            logger.info(f"Modelo '{model_name}' carregado com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo de '{url}': {e}")

            raise Exception(f"Falha ao carregar o modelo de '{url}': {e}")
            
    return loaded_models

def predict_and_log_intent(
    text: str, 
    owner: str, 
    models: Dict[str, IntentClassifier]
) -> Dict:
    """
    1. Executa as predições de ML.
    2. Formata o resultado.
    3. Envia o resultado para o log no banco de dados.
    4. Retorna o resultado final formatado.
    """
    predictions = {}

    for model_name, model in models.items():
        top_intent, all_probs = model.predict(text)
        predictions[model_name] = SinglePrediction(top_intent=top_intent, all_probs=all_probs)

    log_document = PredictionResponse(text=text, 
                                      owner=owner, 
                                      predictions=predictions, 
                                      timestamp=int(datetime.now(timezone.utc).timestamp()))
    
    final_result = log_prediction(log_document)

    return final_result
