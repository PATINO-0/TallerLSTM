from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from functools import lru_cache
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT))).resolve()
FRONTEND_DIR = ROOT / "frontend"
TIMEZONE = ZoneInfo("America/Bogota")
FEATURES = [
    "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
    "radiacion_wm2", "precipitacion_mm", "precio_kwh", "hora",
    "dia_semana", "fin_semana", "festivo", "mes",
    "periodo_dia_madrugada", "periodo_dia_manana",
    "periodo_dia_tarde", "periodo_dia_noche",
]
WINDOW_SIZE = 24
PERIOD_FEATURES = FEATURES[12:]


def read_metadata() -> dict:
    with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("features") != FEATURES or metadata.get("window_hours") != WINDOW_SIZE:
        raise ValueError("Los metadatos no corresponden a una ventana de 24 × 16.")
    return metadata


@lru_cache(maxsize=1)
def read_inference_parameters() -> dict:
    parameters = json.loads((MODEL_DIR / "inference_preprocessing.json").read_text(encoding="utf-8"))
    if parameters.get("format_version") != 1 or parameters.get("feature_columns") != FEATURES or parameters.get("window_size") != WINDOW_SIZE:
        raise ValueError("Los parámetros de inferencia no corresponden al modelo de 24 × 16.")
    means, scales = parameters["feature_mean"], parameters["feature_scale"]
    values = [*means, *scales, parameters["target_mean"], parameters["target_scale"]]
    if len(means) != 16 or len(scales) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError("Los escaladores exportados contienen parámetros inválidos.")
    if any(value <= 0 for value in [*scales, parameters["target_scale"]]):
        raise ValueError("La escala de cada variable debe ser positiva.")
    sources = parameters["sources_sha256"]
    if set(sources) != {"production_model.keras", "preprocessing.pkl", "metadata.json"}:
        raise ValueError("El manifiesto de los artefactos originales está incompleto.")
    for name, expected in {**sources, "production_model.onnx": parameters["onnx_sha256"]}.items():
        actual = hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} cambió después de la conversión. Ejecuta scripts/export_onnx.py de nuevo.")
    return parameters


def read_price_reference() -> dict:
    parameters = read_inference_parameters()
    index = FEATURES.index("precio_kwh")
    return {"mean": parameters["feature_mean"][index], "std": parameters["feature_scale"][index]}
