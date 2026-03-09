from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import RISK_TARGET_COLUMN

ID_OR_LEAKAGE_COLUMNS = {
    "RA",
    "Nome",
    "Defas",
    "Fase ideal",
    "Destaque IEG",
    "Destaque IDA",
    "Destaque IPV",
}


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    model_cols = [column for column in df.columns if column not in ID_OR_LEAKAGE_COLUMNS]
    return df[model_cols].copy()


def split_features_target(
    df: pd.DataFrame, target_column: str = RISK_TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada")
    y = df[target_column].astype(int)
    x = df.drop(columns=[target_column])
    return x, y


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def _infer_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = list(df.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [
        column for column in df.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(
    x: pd.DataFrame, forced_categorical: Iterable[str] | None = None
) -> ColumnTransformer:
    numeric_columns, categorical_columns = _infer_column_groups(x)

    if forced_categorical:
        forced_categorical_set = set(forced_categorical)
        categorical_columns = sorted(set(categorical_columns) | forced_categorical_set)
        numeric_columns = [
            column for column in numeric_columns if column not in forced_categorical_set
        ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
