from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils import load_joblib, load_json, setup_logger


class ModelConfig:
    """Gerenciador de configuração e artefatos do modelo."""

    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]

        self.model_path = Path(
            os.getenv("MODEL_PATH", str(base_dir / "models" / "predictor.joblib"))
        )
        self.config_path = Path(
            os.getenv("CONFIG_PATH", str(base_dir / "models" / "model_config.json"))
        )
        self.reference_profile_path = Path(
            os.getenv(
                "REFERENCE_PROFILE_PATH",
                str(base_dir / "models" / "reference_profile.json"),
            )
        )

        self.logger = setup_logger("api")
        self.model = None
        self.config = None
        self.reference_profile = None
        self.recent_inputs: list[dict] = []
        self.max_recent_inputs = 1500

    def load_artifacts(self):
        self.model = load_joblib(self.model_path)
        self.config = load_json(self.config_path)
        self.reference_profile = load_json(self.reference_profile_path)
        self.logger.info("Modelo carregado de %s", self.model_path)

    def register_inputs(self, records: list[dict]) -> None:
        self.recent_inputs.extend(records)
        if len(self.recent_inputs) > self.max_recent_inputs:
            self.recent_inputs = self.recent_inputs[-self.max_recent_inputs :]

    def recent_inputs_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.recent_inputs)

    def is_loaded(self) -> bool:
        return all([self.model is not None, self.config is not None, self.reference_profile is not None])


model_config = ModelConfig()
