from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from typing import Dict
from app.model.drift_monitor import build_drift_report, render_drift_dashboard_html
from app.schemas import (
    DriftResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.config import model_config

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
    return {
        "features": config["features"],
        "threshold": config["threshold"],
        "metrics": config["metrics"],
        "training_date": config["training_date"],
        "model_name": config["model_name"],
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    records = [item.values for item in request.records]
    if not records:
        raise HTTPException(status_code=400, detail="Nenhum registro enviado")

    config = model_config.config
    threshold = float(config["threshold"])
    expected_features = config["features"]

    input_df = pd.DataFrame(records)
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = None
    input_df = input_df[expected_features]

    probabilities = model_config.model.predict_proba(input_df)[:, 1]
    predictions = [
        {
            "risco_probabilidade": float(probability),
            "risco_defasagem": int(probability >= threshold),
        }
        for probability in probabilities
    ]

    model_config.register_inputs(input_df.to_dict(orient="records"))

    return {
        "model_name": config["model_name"],
        "threshold": threshold,
        "predictions": predictions,
    }


@router.get("/drift", response_model=DriftResponse)
async def get_drift() -> Dict:
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    inputs_df = model_config.recent_inputs_df()
    if inputs_df.empty:
        return {
            "summary": {
                "samples_monitored": 0,
                "numeric_features_monitored": 0,
                "categorical_features_monitored": 0,
                "numeric_drift_count": 0,
                "categorical_drift_count": 0,
                "has_drift": False,
            },
            "numeric": {},
            "categorical": {},
        }

    report = build_drift_report(model_config.reference_profile, inputs_df)
    return report


@router.get("/drift-dashboard", response_class=HTMLResponse)
async def get_drift_dashboard() -> HTMLResponse:
    report = await get_drift()
    html = render_drift_dashboard_html(report)
    return HTMLResponse(content=html)

