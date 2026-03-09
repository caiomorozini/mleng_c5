from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _population_stability_index(reference_dist: dict, current_series: pd.Series) -> float:
    if current_series.empty:
        return 0.0

    current_dist = current_series.astype(str).value_counts(normalize=True).to_dict()
    all_categories = set(reference_dist.keys()) | set(current_dist.keys())
    epsilon = 1e-6
    psi = 0.0
    for category in all_categories:
        ref = float(reference_dist.get(category, epsilon))
        cur = float(current_dist.get(category, epsilon))
        ref = max(ref, epsilon)
        cur = max(cur, epsilon)
        psi += (cur - ref) * np.log(cur / ref)
    return float(psi)


def build_drift_report(reference_profile: dict, current_df: pd.DataFrame) -> dict:
    numeric_report = {}
    categorical_report = {}

    ref_numeric = reference_profile.get("numeric", {})
    ref_categorical = reference_profile.get("categorical", {})

    for column, stats in ref_numeric.items():
        if column not in current_df.columns:
            continue
        current_series = pd.to_numeric(current_df[column], errors="coerce")
        current_mean = _safe_float(current_series.mean())
        current_std = _safe_float(current_series.std(ddof=0))
        ref_mean = _safe_float(stats.get("mean"))
        ref_std = _safe_float(stats.get("std"), default=1e-6)
        z_score_shift = abs(current_mean - ref_mean) / max(ref_std, 1e-6)
        numeric_report[column] = {
            "reference_mean": ref_mean,
            "current_mean": current_mean,
            "reference_std": ref_std,
            "current_std": current_std,
            "z_score_shift": float(z_score_shift),
            "drift_detected": bool(z_score_shift > 0.5),
        }

    for column, reference_dist in ref_categorical.items():
        if column not in current_df.columns:
            continue
        psi = _population_stability_index(reference_dist, current_df[column].fillna("NA"))
        categorical_report[column] = {
            "psi": psi,
            "drift_detected": bool(psi > 0.2),
        }

    numeric_drift_count = sum(1 for value in numeric_report.values() if value["drift_detected"])
    categorical_drift_count = sum(
        1 for value in categorical_report.values() if value["drift_detected"]
    )

    summary = {
        "samples_monitored": int(len(current_df)),
        "numeric_features_monitored": int(len(numeric_report)),
        "categorical_features_monitored": int(len(categorical_report)),
        "numeric_drift_count": int(numeric_drift_count),
        "categorical_drift_count": int(categorical_drift_count),
        "has_drift": bool((numeric_drift_count + categorical_drift_count) > 0),
    }

    return {
        "summary": summary,
        "numeric": numeric_report,
        "categorical": categorical_report,
    }


def render_drift_dashboard_html(report: dict) -> str:
    summary = report.get("summary", {})
    numeric = report.get("numeric", {})
    categorical = report.get("categorical", {})

    numeric_rows = "".join(
        [
            "<tr>"
            f"<td>{escape(column)}</td>"
            f"<td>{values['reference_mean']:.4f}</td>"
            f"<td>{values['current_mean']:.4f}</td>"
            f"<td>{values['z_score_shift']:.4f}</td>"
            f"<td>{'SIM' if values['drift_detected'] else 'NÃO'}</td>"
            "</tr>"
            for column, values in numeric.items()
        ]
    )

    categorical_rows = "".join(
        [
            "<tr>"
            f"<td>{escape(column)}</td>"
            f"<td>{values['psi']:.4f}</td>"
            f"<td>{'SIM' if values['drift_detected'] else 'NÃO'}</td>"
            "</tr>"
            for column, values in categorical.items()
        ]
    )

    return f"""
    <html>
      <head>
        <title>Dashboard de Drift</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background-color: #f2f2f2; }}
        </style>
      </head>
      <body>
        <h1>Dashboard de Drift</h1>
        <p>Amostras monitoradas: <b>{summary.get('samples_monitored', 0)}</b></p>
        <p>Drift numérico: <b>{summary.get('numeric_drift_count', 0)}</b> | Drift categórico: <b>{summary.get('categorical_drift_count', 0)}</b></p>

        <h2>Features Numéricas</h2>
        <table>
          <tr>
            <th>Feature</th>
            <th>Média de Referência</th>
            <th>Média Atual</th>
            <th>Shift (Z-score)</th>
            <th>Drift</th>
          </tr>
          {numeric_rows}
        </table>

        <h2>Features Categóricas</h2>
        <table>
          <tr>
            <th>Feature</th>
            <th>PSI</th>
            <th>Drift</th>
          </tr>
          {categorical_rows}
        </table>
      </body>
    </html>
    """