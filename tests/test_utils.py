from pathlib import Path

import pandas as pd

from src import utils


def test_build_binary_target_and_timestamp():
    df = pd.DataFrame({"Defas": [-1, 0, 2]})
    out = utils.build_binary_target(df)

    assert list(out[utils.RISK_TARGET_COLUMN]) == [1, 0, 0]
    assert "T" in utils.utc_now_iso()


def test_json_and_joblib_roundtrip(tmp_path):
    json_path = tmp_path / "data.json"
    model_path = tmp_path / "artifact.joblib"

    payload = {"a": 1, "b": "x"}
    utils.save_json(payload, json_path)
    loaded_payload = utils.load_json(json_path)

    obj = {"coef": [1, 2, 3]}
    utils.save_joblib(obj, model_path)
    loaded_obj = utils.load_joblib(model_path)

    assert loaded_payload == payload
    assert loaded_obj == obj


def test_load_raw_data_uses_read_excel(monkeypatch):
    called = {"ok": False}

    def fake_read_excel(path):
        called["ok"] = True
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = utils.load_raw_data(Path("fake.xlsx"))

    assert called["ok"] is True
    assert df.shape == (1, 1)
