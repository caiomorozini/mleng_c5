from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    if "Ano ingresso" in output.columns:
        output["anos_no_programa"] = (2024 - output["Ano ingresso"]).clip(lower=0)

    score_columns = [
        column for column in ["Matem", "Portug", "Inglês"] if column in output.columns
    ]
    if score_columns:
        output["media_notas"] = output[score_columns].mean(axis=1)

    inde_columns = [
        column for column in ["INDE 22", "IEG", "IDA", "IPV"] if column in output.columns
    ]
    if inde_columns:
        output["media_indices_pede"] = output[inde_columns].mean(axis=1)

    if "Atingiu PV" in output.columns:
        output["atingiu_pv_bin"] = (
            output["Atingiu PV"].astype(str).str.strip().str.lower().eq("sim").astype(int)
        )

    if "Indicado" in output.columns:
        output["indicado_bin"] = (
            output["Indicado"].astype(str).str.strip().str.lower().eq("sim").astype(int)
        )

    return output
