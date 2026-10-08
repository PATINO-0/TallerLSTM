from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from backend import config
from backend.examples import example_payload
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_onnx_predictions_match_original_keras_references(index):
    report = json.loads((ROOT / "model_conversion_report.json").read_text(encoding="utf-8"))
    case = report["golden_cases"][index]
    payload = example_payload(datetime.fromisoformat(case["end"]))
    response = TestClient(app).post("/api/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_demand_mw"] == pytest.approx(case["expected_demand_mw"], abs=0.002)


@pytest.mark.parametrize("file", ["production_model.keras", "preprocessing.pkl", "metadata.json", "production_model.onnx"])
def test_changed_artifacts_require_a_new_verified_conversion(file, tmp_path, monkeypatch):
    parameters = json.loads((ROOT / "inference_preprocessing.json").read_text(encoding="utf-8"))
    for name in [*parameters["sources_sha256"], "production_model.onnx", "inference_preprocessing.json"]:
        shutil.copyfile(ROOT / name, tmp_path / name)
    altered = tmp_path / file
    altered.write_bytes(altered.read_bytes() + b"changed")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    with pytest.raises(ValueError, match="cambió después de la conversión"):
        config.read_inference_parameters.__wrapped__()


def test_exported_parameters_do_not_accept_zero_scales(tmp_path, monkeypatch):
    parameters = json.loads((ROOT / "inference_preprocessing.json").read_text(encoding="utf-8"))
    parameters["feature_scale"][0] = 0
    (tmp_path / "inference_preprocessing.json").write_text(json.dumps(parameters), encoding="utf-8")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    with pytest.raises(ValueError, match="positiva"):
        config.read_inference_parameters.__wrapped__()
