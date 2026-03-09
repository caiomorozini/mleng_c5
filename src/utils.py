from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
MODEL_DIR = BASE_DIR / "models"
TARGET_COLUMN = "Defas"
RISK_TARGET_COLUMN = "risco_defasagem"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def build_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada")
    output = df.copy()
    output[RISK_TARGET_COLUMN] = (output[TARGET_COLUMN] < 0).astype(int)
    return output


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_joblib(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path | str) -> Any:
    return joblib.load(path)


def save_json(data: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
