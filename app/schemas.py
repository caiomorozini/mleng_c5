from pydantic import BaseModel, Field
from typing import List, Dict


class HealthResponse(BaseModel):
    """Response do health check"""

    status: str
    model_loaded: bool
    config: Dict
