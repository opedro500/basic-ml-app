import logging
import traceback

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from db.auth import conditional_auth
from app import services
from app.schema import PredictRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict", 
    tags=["Predição"]
)

@router.post("/")
async def fazer_predicao(request: Request, body: PredictRequest, owner: str = Depends(conditional_auth)):
    try:
        models = request.app.state.models
        
        results = services.predict_and_log_intent(
            text=body.text, 
            owner=owner, 
            models=models
        )
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Erro ao processar a predição: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.")
