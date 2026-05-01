from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictRequest(BaseModel):
    """Contrato do que a API espera receber do usuário."""
    text: str = Field(
        ..., 
        description="O texto ou frase que o usuário deseja classificar.", 
        json_schema_extra={"examples": ["I am so confused right now."]}
    )

class SinglePrediction(BaseModel):
    """Representa a predição de um único modelo."""
    top_intent: str = Field(
        ..., 
        description="A intenção que recebeu a maior probabilidade (classe vencedora)."
    )
    all_probs: Dict[str, float] = Field(
        ..., 
        description="Dicionário com as probabilidades individuais de todas as classes."
    )

class PredictionResponse(BaseModel):
    """Contrato do que a API devolve para o usuário no final do processo."""
    id: Optional[str] = Field(
        None, 
        description="ID único gerado automaticamente ao salvar no banco de dados."
    )
    text: str = Field(..., description="O texto original enviado pelo usuário.")
    owner: str = Field(..., description="Identificação do dono do token que fez a requisição.")
    predictions: Dict[str, SinglePrediction] = Field(
        ..., 
        description="Resultados das predições, agrupados pelo nome do modelo que as gerou."
    )
    timestamp: int = Field(..., description="Data e hora exata da predição em formato Unix timestamp.")
