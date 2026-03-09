from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.config import model_config


class DummyModel:
    def predict_proba(self, x):
        probs = np.full(shape=(len(x),), fill_value=0.8)
        return np.column_stack([1 - probs, probs])


def _bootstrap_loaded_model_state():
    model_config.model = DummyModel()
    model_config.config = {
        "model_name": "dummy",
        "threshold": 0.5,
        "features": ["f1", "f2"],
        "metrics": {"f1": 0.8},
        "training_date": "2026-03-08T00:00:00+00:00",
    }
    model_config.reference_profile = {
        "numeric": {"f1": {"mean": 1.0, "std": 1.0}},
        "categorical": {"f2": {"a": 0.5, "b": 0.5}},
    }
    model_config.recent_inputs = []


def test_predict_endpoint_returns_predictions(monkeypatch):
    monkeypatch.setattr(model_config, "load_artifacts", lambda: None)
    _bootstrap_loaded_model_state()

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"records": [{"values": {"f1": 1.2, "f2": "a"}}, {"values": {"f1": 0.2, "f2": "b"}}]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "dummy"
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["risco_defasagem"] == 1


def test_drift_endpoint_returns_summary(monkeypatch):
    monkeypatch.setattr(model_config, "load_artifacts", lambda: None)
    _bootstrap_loaded_model_state()
    model_config.register_inputs([{"f1": 5.0, "f2": "z"}, {"f1": 4.0, "f2": "z"}])

    with TestClient(app) as client:
        response = client.get("/drift")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["samples_monitored"] >= 2
    assert "numeric" in payload
    assert "categorical" in payload


def test_health_model_info_and_dashboard(monkeypatch):
    monkeypatch.setattr(model_config, "load_artifacts", lambda: None)
    _bootstrap_loaded_model_state()
    model_config.register_inputs([{"f1": 1.0, "f2": "a"}])

    with TestClient(app) as client:
        health = client.get("/health")
        model_info = client.get("/model-info")
        dashboard = client.get("/drift-dashboard")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert model_info.status_code == 200
    assert model_info.json()["model_name"] == "dummy"
    assert dashboard.status_code == 200
    assert "Dashboard de Drift" in dashboard.text


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(model_config, "load_artifacts", lambda: None)
    model_config.model = None
    model_config.config = None
    model_config.reference_profile = None

    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [{"values": {"f1": 1}}]})

    assert response.status_code == 503
