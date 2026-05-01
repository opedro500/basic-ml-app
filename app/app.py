import os
import traceback
import logging

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict
from app import services

load_dotenv()
logger = logging.getLogger(__name__)
ENV = os.getenv("ENV", "prod").lower()

class EndpointFilter(logging.Filter):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self.path) == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter(path="/health"))

def get_model_urls() -> str:
    models_env = os.getenv("WANDB_MODELS")
    assert models_env is not None, "Variável WANDB_MODELS não definida."
    return models_env

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando modelos do W&B...")
    try:
        model_urls_str = get_model_urls()
        app.state.models = services.load_all_classifiers(model_urls_str)
        logger.info("Modelos carregados com sucesso.")
    except Exception as e:
        logger.error(f"Falha ao carregar modelos: {str(e)}")
        raise e
    
    yield
    
    logger.info("Limpando memória...")
    app.state.models.clear()

app = FastAPI(title="Basic ML App", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENV == "dev" else ["http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/", tags=["Sistema"])
async def root():
    return {"message": f"Basic ML App rodando em modo {ENV}"}

@app.get("/health", tags=["Sistema"])
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
