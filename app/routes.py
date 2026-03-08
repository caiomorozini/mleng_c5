from fastapi import APIRouter, HTTPException
from typing import Dict
from api.schemas import (
    HealthResponse,
)
from api.config import model_config

router = APIRouter()

@router.get("/", response_model=Dict)
async def root():
    """Endpoint raiz"""
    return {
        "endpoints": {
            "health": "/health",
            "predict": "/predict (com dados históricos)",
            "model_info": "/model-info",
            "docs": "/docs",
        },
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saúde da API"""
    return {
        "status": "healthy" if model_config.is_loaded() else "unhealthy",
        "model_loaded": model_config.is_loaded(),
        "config": model_config.config if model_config.config else {},
    }


@router.get("/model-info", response_model=Dict)
async def get_model_info():
    """Retorna informações sobre o modelo carregado"""
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    config = model_config.config
    model = model_config.model

    return {
        "features": config["features"],
        "sequence_length": config["seq_length"],
        "metrics": config["metrics"],
        "training_date": config["training_date"],
        "model_architecture": {
            "total_params": model.count_params(),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        },
    }


@router.post("/predict")
async def predict(request):
    raise HTTPException(status_code=501, detail="Endpoint /predict ainda não implementado")

