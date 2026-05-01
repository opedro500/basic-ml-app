import os
import traceback
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db.auth import conditional_auth
from app import services
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Lê o modo do ambiente (o padrão é prod para maior segurança)
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Rodando em modo {ENV}")

# Dicionário global para armazenar os modelos de ML carregados em memória.
MODELS = {}

def get_model_urls() -> str:
    """
    Busca a string de URLs de modelos da variável de ambiente WANDB_MODELS.
    Isolar essa lógica em uma função facilita fazer simulações (patching) durante os testes.
    """
    models_env = os.getenv("WANDB_MODELS")
    assert models_env is not None, "Variável de ambiente WANDB_MODELS não definida."
    return models_env

class EndpointFilter(logging.Filter):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def filter(self, record: logging.LogRecord) -> bool:
        # Retorna False se a mensagem contiver a rota, impedindo que ela polua o log do terminal
        return record.getMessage().find(self.path) == -1

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador do ciclo de vida da aplicação (Startup e Shutdown).
    Faz o download e carrega os modelos do W&B na memória ao iniciar.
    """
    global MODELS
    logger.info("Carregando modelos do W&B durante a inicialização do app...")
    try:
        model_urls_str = get_model_urls()
        MODELS = services.load_all_classifiers(model_urls_str)
        logger.info("Modelos do W&B carregados com sucesso.")
    except Exception as e:
        logger.error(f"Falha crítica ao carregar modelos do W&B: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Falha crítica ao carregar modelos do W&B: {str(e)}")
    
    # A partir deste ponto (yield), a aplicação está pronta para receber requisições
    yield
    
    # Código para ser executado no desligamento (shutdown)
    logger.info("Descarregando modelos e limpando recursos da memória...")
    MODELS.clear()

# Inicializa a aplicação FastAPI conectada ao gerenciador de ciclo de vida (lifespan)
app = FastAPI(
    title="Basic ML App",
    description="Uma aplicação básica de Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
)

# Aplica o filtro para ignorar logs da rota /health
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(EndpointFilter(path="/health"))

# Controle de CORS (Cross-Origin Resource Sharing) para prevenir ataques de fontes não autorizadas.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        # "http://localhost:3000",  # React ou outro frontend local
        # "https://meusite.com",    # Domínio oficial em produção
    ],
    allow_credentials=True,
    allow_methods=["*"],              # Permite todos os métodos (GET, POST, PUT, DELETE, etc)
    allow_headers=["*"],              # Permite todos os cabeçalhos (Authorization, Content-Type, etc)
    # Durante o desenvolvimento: você pode usar allow_origins=["*"] para liberar acesso de qualquer lugar.
    # Em produção: evite o "*" e especifique apenas os domínios confiáveis.
)

"""
Rotas (Endpoints da API)
"""
@app.get("/")
async def root():
    return {"message": f"Basic ML App está rodando em modo {ENV}"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    """
    Endpoint de predição.
    Este é um 'Controller' enxuto, que segue o padrão MVC. 
    Ele não faz processamento pesado, apenas recebe o texto e delega a lógica para o services.py.
    """
    try:
        results = services.predict_and_log_intent(
            text=text, 
            owner=owner, 
            models=MODELS
        )
  
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Erro ao processar a predição: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
