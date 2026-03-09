import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src import train as train_module


def _synthetic_training_df(size: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    defas = np.where(rng.random(size) > 0.4, -1, 0)
    return pd.DataFrame(
        {
            "RA": [f"RA-{i}" for i in range(size)],
            "Nome": [f"Aluno-{i}" for i in range(size)],
            "Fase": rng.integers(1, 8, size=size),
            "Turma": rng.choice(["A", "B", "C"], size=size),
            "Ano ingresso": rng.integers(2016, 2024, size=size),
            "Idade 22": rng.integers(10, 20, size=size),
            "Gênero": rng.choice(["Menino", "Menina"], size=size),
            "Instituição de ensino": rng.choice(["Pública", "Privada"], size=size),
            "Pedra 20": rng.choice(["Ametista", "Quartzo"], size=size),
            "Pedra 21": rng.choice(["Ametista", "Quartzo"], size=size),
            "Pedra 22": rng.choice(["Ametista", "Quartzo"], size=size),
            "INDE 22": rng.normal(6, 1, size=size),
            "Cg": rng.integers(100, 900, size=size),
            "Cf": rng.integers(1, 20, size=size),
            "Ct": rng.integers(1, 10, size=size),
            "Nº Av": rng.integers(1, 5, size=size),
            "IAA": rng.normal(7, 1, size=size),
            "IEG": rng.normal(5, 1, size=size),
            "IPS": rng.normal(6, 1, size=size),
            "Rec Psicologia": rng.choice(["Sem limitações", "Requer avaliação"], size=size),
            "IDA": rng.normal(6, 1, size=size),
            "Matem": rng.normal(6, 2, size=size),
            "Portug": rng.normal(6, 2, size=size),
            "Inglês": rng.normal(6, 2, size=size),
            "Indicado": rng.choice(["Sim", "Não"], size=size),
            "Atingiu PV": rng.choice(["Sim", "Não"], size=size),
            "IPV": rng.normal(6, 1, size=size),
            "IAN": rng.normal(6, 1, size=size),
            "Fase ideal": rng.choice(["Fase 5", "Fase 6"], size=size),
            "Defas": defas,
            "Destaque IEG": ["-" for _ in range(size)],
            "Destaque IDA": ["-" for _ in range(size)],
            "Destaque IPV": ["-" for _ in range(size)],
        }
    )


def test_train_and_save_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "load_raw_data", lambda: _synthetic_training_df())
    monkeypatch.setattr(
        train_module,
        "RandomForestClassifier",
        lambda **kwargs: RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1),
    )

    result = train_module.train_and_save_artifacts(model_dir=tmp_path)

    assert (tmp_path / "predictor.joblib").exists()
    assert (tmp_path / "model_config.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "reference_profile.json").exists()
    assert result["selected_model"] in {"logistic_regression", "random_forest"}

    with Path(tmp_path / "model_config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert "features" in config
    assert config["target_name"] == "risco_defasagem"
