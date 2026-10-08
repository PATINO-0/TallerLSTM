from __future__ import annotations

import json
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
def read_price_reference() -> dict:
    import joblib

    preprocessing = joblib.load(MODEL_DIR / "preprocessing.pkl")
    index = preprocessing["feature_columns"].index("precio_kwh")
    scaler = preprocessing["feature_scaler"]
    return {"mean": float(scaler.mean_[index]), "std": float(scaler.scale_[index])}
