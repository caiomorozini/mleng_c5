from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_binary_classifier, find_best_threshold
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor, select_model_columns, split_features_target
from src.utils import (
    MODEL_DIR,
    RISK_TARGET_COLUMN,
    build_binary_target,
    load_raw_data,
    save_joblib,
    save_json,
    setup_logger,
    utc_now_iso,
)

LOGGER = setup_logger("training")


def _build_reference_profile(df: pd.DataFrame) -> dict:
    numeric = {}
    categorical = {}

    numeric_columns = list(df.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        numeric[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0) if series.notna().sum() > 1 else 0.0),
            "p05": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
        }

    for column in categorical_columns:
        distribution = (
            df[column]
            .astype(str)
            .fillna("NA")
            .value_counts(normalize=True)
            .head(20)
            .to_dict()
        )
        categorical[column] = {key: float(value) for key, value in distribution.items()}

    return {"numeric": numeric, "categorical": categorical}


def train_and_save_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    LOGGER.info("Carregando dados")
    df = load_raw_data()
    df = build_binary_target(df)
    df = engineer_features(df)
    df = select_model_columns(df)

    x, y = split_features_target(df, target_column=RISK_TARGET_COLUMN)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }

    best = None
    all_results = {}

    for model_name, model in candidates.items():
        LOGGER.info("Treinando candidato: %s", model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_proba = pipeline.predict_proba(x_test)[:, 1]

        best_threshold, best_metrics = find_best_threshold(
            y_true=y_test.to_numpy(),
            y_proba=np.asarray(y_proba),
            min_recall=0.7,
        )

        baseline_metrics = evaluate_binary_classifier(
            y_true=y_test.to_numpy(),
            y_proba=np.asarray(y_proba),
            threshold=0.5,
        )

        result = {
            "model_name": model_name,
            "threshold": float(best_threshold),
            "metrics_best_threshold": best_metrics,
            "metrics_threshold_0_5": baseline_metrics,
        }
        all_results[model_name] = result

        if best is None or result["metrics_best_threshold"]["f1"] > best["metrics_best_threshold"]["f1"]:
            best = {**result, "pipeline": pipeline}

    if best is None:
        raise RuntimeError("Nenhum modelo foi treinado")

    model_dir.mkdir(parents=True, exist_ok=True)

    predictor_path = model_dir / "predictor.joblib"
    config_path = model_dir / "model_config.json"
    metrics_path = model_dir / "metrics.json"
    reference_profile_path = model_dir / "reference_profile.json"

    save_joblib(best["pipeline"], predictor_path)

    config = {
        "model_name": best["model_name"],
        "target_name": RISK_TARGET_COLUMN,
        "target_definition": "risco_defasagem = 1 quando Defas < 0",
        "threshold": best["threshold"],
        "features": list(x.columns),
        "training_date": utc_now_iso(),
        "metrics": best["metrics_best_threshold"],
    }
    save_json(config, config_path)
    save_json(all_results, metrics_path)
    save_json(_build_reference_profile(x_train), reference_profile_path)

    LOGGER.info("Artefatos salvos em %s", model_dir)
    return {
        "predictor": str(predictor_path),
        "config": str(config_path),
        "metrics": str(metrics_path),
        "reference_profile": str(reference_profile_path),
        "selected_model": best["model_name"],
    }


if __name__ == "__main__":
    output = train_and_save_artifacts()
    print(output)
