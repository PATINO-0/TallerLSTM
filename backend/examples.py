from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timedelta

from backend.config import FEATURES, PERIOD_FEATURES, TIMEZONE, WINDOW_SIZE


def calendar_features(timestamp: datetime) -> dict:
    # Convención de la interfaz; los artefactos no documentan los cortes originales.
    # El usuario puede cambiar el período explícitamente o importar su codificación.
    hour = timestamp.hour
    period = 0 if hour < 6 else 1 if hour < 12 else 2 if hour < 18 else 3
    return {
        "hora": hour, "dia_semana": timestamp.weekday(),
        "fin_semana": int(timestamp.weekday() >= 5), "festivo": 0, "mes": timestamp.month,
        **{feature: int(index == period) for index, feature in enumerate(PERIOD_FEATURES)},
    }


def example_payload(end: datetime | None = None) -> dict:
    end = end or datetime.now(TIMEZONE).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    observations = []
    for index in range(WINDOW_SIZE):
        timestamp = end - timedelta(hours=WINDOW_SIZE - 1 - index)
        hour = timestamp.hour
        daylight = max(0, math.sin(math.pi * (hour - 6) / 12))
        observations.append({
            "timestamp": timestamp.isoformat(),
            "demanda_mw": round(640 + 70 * math.sin((hour - 7) * math.pi / 12) + 20 * math.cos(hour * math.pi / 6), 2),
            "temperatura_c": round(18 + 5 * daylight, 2),
            "humedad_pct": round(83 - 14 * daylight, 2),
            "viento_kmh": round(10 + 3 * daylight, 2),
            "radiacion_wm2": round(350 * daylight, 2),
            "precipitacion_mm": 0.0,
            "precio_kwh": round(24 + 2 * math.sin(hour * math.pi / 12), 2),
            **calendar_features(timestamp),
        })
    return {"source": "demo", "observations": observations}


def template_csv(layout: str = "simple") -> str:
    output = io.StringIO(newline="")
    fields = ["timestamp", *FEATURES[:7], "festivo"] if layout == "simple" else ["timestamp", *FEATURES]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in example_payload()["observations"]:
        writer.writerow({key: row[key] if key == "timestamp" or key in FEATURES[7:] else "" for key in fields})
    return output.getvalue()
