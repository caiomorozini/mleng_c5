import pandas as pd

from src.preprocessing import (
    build_preprocessor,
    select_model_columns,
    split_features_target,
)


def test_select_model_columns_removes_id_and_leakage_columns():
    df = pd.DataFrame(
        {
            "RA": ["RA-1"],
            "Nome": ["Aluno"],
            "Defas": [-1],
            "Fase ideal": ["Fase 7"],
            "feature_a": [1],
            "feature_b": ["x"],
            "risco_defasagem": [1],
        }
    )

    out = select_model_columns(df)

    assert "RA" not in out.columns
    assert "Nome" not in out.columns
    assert "Defas" not in out.columns
    assert "Fase ideal" not in out.columns
    assert "feature_a" in out.columns
    assert "risco_defasagem" in out.columns


def test_split_and_preprocessor_fit_transform():
    df = pd.DataFrame(
        {
            "feature_num": [1.0, 2.0, None, 4.0],
            "feature_cat": ["a", "b", "a", None],
            "risco_defasagem": [0, 1, 0, 1],
        }
    )

    x, y = split_features_target(df)
    preprocessor = build_preprocessor(x)
    matrix = preprocessor.fit_transform(x)

    assert len(y) == 4
    assert matrix.shape[0] == 4
    assert matrix.shape[1] >= 2
