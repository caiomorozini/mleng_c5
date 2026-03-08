import os
import pickle
import json
from tensorflow import keras


class ModelConfig:
    """Gerenciador de configuração e artefatos do modelo"""

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.MODEL_PATH = os.getenv(
            "MODEL_PATH", os.path.join(BASE_DIR, "models", "predictor.keras")
        )
        self.SCALER_PATH = os.getenv(
            "SCALER_PATH", os.path.join(BASE_DIR, "models", "scaler.pkl")
        )
        self.CONFIG_PATH = os.getenv(
            "CONFIG_PATH", os.path.join(BASE_DIR, "models", "model_config.json")
        )

        self.model = None
        self.scaler = None
        self.config = None

    def load_artifacts(self):
        """Carrega modelo, scaler e configurações"""
        try:
            self.model = keras.models.load_model(self.MODEL_PATH)
            print(f"✓ Modelo carregado: {self.MODEL_PATH}")

            with open(self.SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler carregado: {self.SCALER_PATH}")

            with open(self.CONFIG_PATH, "r") as f:
                self.config = json.load(f)
            print(f"Configurações carregadas: {self.CONFIG_PATH}")


        except Exception as e:
            raise

    def is_loaded(self) -> bool:
        """Verifica se todos os artefatos foram carregados"""
        return all(
            [self.model is not None, self.scaler is not None, self.config is not None]
        )


model_config = ModelConfig()
