import os
import uuid
import logging

from datetime import datetime, timedelta, timezone
from fastapi import Request, HTTPException
from dotenv import load_dotenv
from db.engine import get_mongo_collection

load_dotenv()
logger = logging.getLogger(__name__)

ENV = os.getenv("ENV", "prod").lower()

# ==========================================
# 1. FERRAMENTA DE GERENCIAMENTO (CLI)
# ==========================================
class TokenManager:
    """
    Ferramenta de terminal para gerenciar os tokens da API no banco de dados.
    Uso: python -m db.auth create --owner="cliente1" --expires_in_days=30
    """
    def create(self, owner: str, note: str = "", expires_in_days: int = 180):
        """
        Cria um novo token de acesso e salva no MongoDB.
        Args:
            owner (str): Nome do cliente ou sistema dono do token.
            note (str): Descrição opcional.
            expires_in_days (int): Validade do token em dias.
        """
        token = str(uuid.uuid4())
        tokens_collection = get_mongo_collection("api_tokens")

        now = datetime.now(timezone.utc)
        token_doc = {
            "token": token,
            "owner": owner,
            "note": note,
            "created_at": now,
            "expires_at": now + timedelta(days=expires_in_days),
            "active": True
        }

        tokens_collection.insert_one(token_doc)
        print(f"✅ Token criado para '{owner}' (expira em {expires_in_days} dias): {token}")

    def read_all(self):
        """Lê e imprime todos os tokens armazenados no MongoDB."""
        tokens_collection = get_mongo_collection("api_tokens")
        all_tokens = tokens_collection.find()
        for t in all_tokens:
            print({
                "token": t.get("token"),
                "owner": t.get("owner"),
                "note": t.get("note"),
                "active": t.get("active"),
                "created_at": t.get("created_at")
            })

    def delete_expired(self):
        """Remove tokens que já passaram da data de expiração para limpar o banco."""
        tokens_collection = get_mongo_collection("api_tokens")
    
        result = tokens_collection.delete_many({"expires_at": {"$lt": datetime.now(timezone.utc)}})
        print(f"🧹 Tokens expirados removidos: {result.deleted_count}")


# ==========================================
# 2. MIDDLEWARES DE AUTENTICAÇÃO (FASTAPI)
# ==========================================
def verify_token(request: Request) -> str:
    """Valida o token enviado no header da requisição e retorna o dono."""
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Header de Autorização ausente.")

    token = token.replace("Bearer ", "")
    
    tokens_collection = get_mongo_collection("api_tokens")
    token_entry = tokens_collection.find_one({"token": token, "active": True})

    if not token_entry:
        raise HTTPException(status_code=403, detail="Token inválido ou inativo.")

    if datetime.now(timezone.utc) > token_entry["expires_at"].replace(tzinfo=timezone.utc):
        raise HTTPException(status_code=403, detail="O Token expirou.")

    return token_entry["owner"]

async def conditional_auth(request: Request) -> str:
    """
    Função injetável (Depends) para as rotas do FastAPI.
    Se o ambiente for 'dev', libera o acesso automaticamente.
    Se for 'prod', exige que o 'verify_token' aprove a entrada.
    """
    if ENV == "dev":
        return "dev_user"
    else:
        try:
            return verify_token(request)
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Erro interno de autenticação: {e}")
            raise HTTPException(status_code=401, detail="Falha na autenticação.")

if __name__ == "__main__":
    import fire
    fire.Fire(TokenManager)
