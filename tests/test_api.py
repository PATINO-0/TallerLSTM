from __future__ import annotations

import copy
import csv
import io
import math
import json
from pathlib import Path
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import app as api_module
from backend.config import FEATURES
from backend.examples import example_payload
from backend.inference import ModelUnavailable, get_predictor

client = TestClient(api_module.app)


@pytest.fixture
def payload():
    return example_payload(datetime.fromisoformat("2026-06-10T23:00:00-05:00"))


def test_api_matches_verified_reference_from_original_keras_model(payload):
    report = json.loads((Path(__file__).resolve().parents[1] / "model_conversion_report.json").read_text(encoding="utf-8"))
    expected = report["golden_cases"][0]["expected_demand_mw"]
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["predicted_demand_mw"] == pytest.approx(expected, abs=0.002)
    assert result["unit"] == "MW"
    assert result["target"] == "demanda_objetivo"
    assert result["prediction_timestamp"] == "2026-06-11T00:00:00-05:00"
    assert result["next_hour"] == 0
    assert result["source"] == "demo"
    assert any("sintéticos" in item for item in result["warnings"])
    assert math.isfinite(result["predicted_demand_mw"])
    assert get_predictor().engine == "onnxruntime"
    assert report["maximum_absolute_error_mw"] <= report["allowed_absolute_error_mw"]


def test_predictions_are_repeatable_and_json_key_order_does_not_change_features(payload):
    first = client.post("/api/predict", json=payload).json()
    payload["observations"] = [dict(reversed(list(row.items()))) for row in payload["observations"]]
    second = client.post("/api/predict", json=payload).json()
    assert first["predicted_demand_mw"] == second["predicted_demand_mw"]


@pytest.mark.parametrize("count", [0, 1, 23, 25])
def test_requires_exactly_24_observations(payload, count):
    payload["observations"] = (payload["observations"] * 2)[:count]
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"]


@pytest.mark.parametrize("feature,value", [
    ("demanda_mw", -1), ("temperatura_c", 101), ("humedad_pct", 101),
    ("humedad_pct", -1), ("viento_kmh", -1), ("radiacion_wm2", -1),
    ("precipitacion_mm", -1), ("hora", 24), ("hora", 1.5),
    ("dia_semana", 7), ("mes", 0), ("festivo", 2),
    ("demanda_mw", "NaN"), ("precio_kwh", "Infinity"),
])
def test_rejects_invalid_features_with_serializable_errors(payload, feature, value):
    payload["observations"][0][feature] = value
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["message"]


def test_rejects_missing_and_unknown_features(payload):
    del payload["observations"][0]["demanda_mw"]
    payload["observations"][1]["unknown"] = 10
    assert client.post("/api/predict", json=payload).status_code == 422


def test_rejects_non_one_hot_period(payload):
    for feature in FEATURES[12:]:
        payload["observations"][0][feature] = 0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "período" in response.json()["errors"][0]["message"]


def test_rejects_calendar_inconsistency(payload):
    payload["observations"][0]["fin_semana"] = 1
    assert client.post("/api/predict", json=payload).status_code == 422


def test_rejects_out_of_order_and_gaps(payload):
    payload["observations"][0], payload["observations"][1] = payload["observations"][1], payload["observations"][0]
    assert client.post("/api/predict", json=payload).status_code == 422


def test_rejects_naive_and_partially_missing_timestamps(payload):
    changed = copy.deepcopy(payload)
    changed["observations"][0]["timestamp"] = "2026-06-10T00:00:00"
    assert client.post("/api/predict", json=changed).status_code == 422
    del payload["observations"][0]["timestamp"]
    assert client.post("/api/predict", json=payload).status_code == 422


def test_raw_features_without_timestamps_are_supported(payload):
    for row in payload["observations"]:
        del row["timestamp"]
    payload["source"] = "csv"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction_timestamp"] is None
    assert response.json()["next_hour"] == 0


def test_zero_last_demand_has_no_division_by_zero(payload):
    payload["observations"][-1]["demanda_mw"] = 0
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["change_pct"] is None


def test_out_of_distribution_warning_is_based_on_saved_scaler(payload):
    payload["observations"][0]["viento_kmh"] = 99
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert any("viento_kmh" in warning for warning in response.json()["warnings"])


def test_unavailable_model_returns_503_instead_of_fake_prediction(payload, monkeypatch):
    def unavailable():
        raise ModelUnavailable("Modelo no disponible para esta prueba.")
    monkeypatch.setattr(api_module, "get_predictor", unavailable)
    assert client.post("/api/predict", json=payload).status_code == 503
    assert client.get("/api/health").status_code == 503


def test_frontend_metadata_health_and_blank_template():
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.mjs").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/api/health").json()["model_ready"] is True
    assert client.get("/api/metadata").json()["features"] == FEATURES
    response = client.get("/api/template.csv")
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"))))
    assert len(rows) == 24
    assert all(row["demanda_mw"] == "" for row in rows)
    assert all(row["timestamp"] for row in rows)
    assert list(rows[0]) == ["timestamp", *FEATURES[:7], "festivo"]
    full = client.get("/api/template.csv?layout=full")
    full_rows = list(csv.DictReader(io.StringIO(full.text.lstrip("\ufeff"))))
    assert list(full_rows[0]) == ["timestamp", *FEATURES]
    assert len(full_rows) == 24
    reference = client.get("/api/metadata").json()["price_training_reference"]
    assert reference["mean"] == pytest.approx(get_predictor().feature_mean[6])
    assert reference["std"] == pytest.approx(get_predictor().feature_scale[6])


def test_future_weather_alone_does_not_match_the_trained_model():
    payload = {"hora": 18, "temperatura_c": 17.8, "humedad_pct": 74, "viento_kmh": 6.2, "radiacion_wm2": 40, "precipitacion_mm": 0.5, "precio_kwh": 810, "festivo": 0}
    assert client.post("/api/predict", json=payload).status_code == 422


@pytest.mark.parametrize("end", ["2026-01-01T10:00:00-05:00", "2024-03-01T10:00:00-05:00", "2026-10-05T06:00:00-05:00"])
def test_calendar_at_year_leap_day_and_weekend_boundaries(end):
    payload = example_payload(datetime.fromisoformat(end))
    assert client.post("/api/predict", json=payload).status_code == 200
