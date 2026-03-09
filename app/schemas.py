from pydantic import BaseModel, Field
from typing import Any, Dict, List


class HealthResponse(BaseModel):
    """Response do health check"""

    status: str
    model_loaded: bool
    config: Dict


class PredictionRecord(BaseModel):
    values: Dict[str, Any] = Field(..., description="Dados de um estudante")


class PredictionRequest(BaseModel):
    records: List[PredictionRecord]


class PredictionItem(BaseModel):
    risco_probabilidade: float
    risco_defasagem: int


class PredictionResponse(BaseModel):
    model_name: str
    threshold: float
    predictions: List[PredictionItem]


class DriftResponse(BaseModel):
    summary: Dict[str, Any]
    numeric: Dict[str, Any]
    categorical: Dict[str, Any]


class DriftSummaryResponse(BaseModel):
    samples_monitored: int
    numeric_features_monitored: int
    categorical_features_monitored: int
    numeric_drift_count: int
    categorical_drift_count: int
    has_drift: bool
