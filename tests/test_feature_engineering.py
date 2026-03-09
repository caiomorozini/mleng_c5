import pandas as pd

from src.feature_engineering import engineer_features


def test_engineer_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "Ano ingresso": [2020, 2022],
            "Matem": [7.0, 5.0],
            "Portug": [8.0, 6.0],
            "Inglês": [9.0, 7.0],
            "INDE 22": [6.0, 5.0],
            "IEG": [5.0, 4.0],
            "IDA": [7.0, 6.0],
            "IPV": [8.0, 7.0],
            "Atingiu PV": ["Sim", "Não"],
            "Indicado": ["Não", "Sim"],
        }
    )

    out = engineer_features(df)

    assert "anos_no_programa" in out.columns
    assert "media_notas" in out.columns
    assert "media_indices_pede" in out.columns
    assert "atingiu_pv_bin" in out.columns
    assert "indicado_bin" in out.columns
    assert out.loc[0, "atingiu_pv_bin"] == 1
    assert out.loc[1, "indicado_bin"] == 1
