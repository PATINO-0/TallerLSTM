from __future__ import annotations

import logging
import os
import threading
import time
from datetime import timedelta
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from backend.config import FEATURES, MODEL_DIR, TIMEZONE, WINDOW_SIZE, read_metadata
from backend.schemas import PredictionRequest, PredictionResponse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
logger = logging.getLogger(__name__)
_load_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self) -> None:
        # Solo se deserializan los artefactos locales del proyecto, nunca archivos del usuario.
        import keras

        self.metadata = read_metadata()
        preprocessing = joblib.load(MODEL_DIR / "preprocessing.pkl")
        if preprocessing["feature_columns"] != FEATURES or preprocessing["window_size"] != WINDOW_SIZE:
            raise ValueError("El preprocesamiento no coincide con los metadatos.")
        self.feature_scaler = preprocessing["feature_scaler"]
        self.target_scaler = preprocessing["target_scaler"]
        if list(self.feature_scaler.feature_names_in_) != FEATURES:
            raise ValueError("El orden de las variables del escalador no coincide.")
        if self.feature_scaler.n_features_in_ != 16 or self.target_scaler.n_features_in_ != 1:
            raise ValueError("Los escaladores tienen dimensiones incompatibles.")
        self.model = keras.saving.load_model(MODEL_DIR / "production_model.keras", compile=False, safe_mode=True)
        if tuple(self.model.input_shape) != (None, 24, 16) or tuple(self.model.output_shape) != (None, 1):
            raise ValueError("El modelo debe aceptar (batch, 24, 16) y devolver (batch, 1).")
        self._predict_lock = threading.Lock()
        # Calienta la inferencia una vez por proceso, usando entradas estandarizadas.
        self.model(np.zeros((1, WINDOW_SIZE, len(FEATURES)), dtype=np.float32), training=False)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        started = time.perf_counter()
        frame = pd.DataFrame(
            [{feature: getattr(row, feature) for feature in FEATURES} for row in request.observations],
            columns=FEATURES,
        )
        scaled = self.feature_scaler.transform(frame)
        if not np.isfinite(scaled).all():
            raise ValueError("Los valores exceden el rango numérico admitido por el modelo.")
        tensor = np.asarray(scaled, dtype=np.float32).reshape(1, WINDOW_SIZE, len(FEATURES))
        if not np.isfinite(tensor).all():
            raise ValueError("Los valores exceden el rango numérico admitido por el modelo.")
        with self._predict_lock:
            raw = np.asarray(self.model(tensor, training=False), dtype=np.float64).reshape(1, 1)
        # La salida de Keras sigue en escala estándar; restaurar MW es indispensable.
        demand = float(self.target_scaler.inverse_transform(raw)[0, 0])
        if not np.isfinite(demand):
            raise ModelUnavailable("El modelo produjo una salida no finita.")
        warnings = []
        if request.source == "demo":
            warnings.append("Esta predicción usa datos sintéticos de demostración. Sustitúyelos por observaciones reales.")
        unusual = [feature for index, feature in enumerate(FEATURES[:7]) if np.any(np.abs(scaled[:, index]) > 3)]
        if unusual:
            warnings.append("Valores alejados de la distribución de entrenamiento (>3 desviaciones estándar): " + ", ".join(unusual) + ".")
        if demand < 0:
            warnings.append("El modelo devolvió demanda negativa. Revisa las entradas; la salida se conserva sin recortarla.")
        last = request.observations[-1]
        change = demand - last.demanda_mw
        predicted_at = (last.timestamp + timedelta(hours=1)).astimezone(TIMEZONE) if last.timestamp else None
        return PredictionResponse(
            predicted_demand_mw=round(demand, 3), target="demanda_objetivo", unit="MW",
            prediction_timestamp=predicted_at, next_hour=(last.hora + 1) % 24,
            last_demand_mw=last.demanda_mw, change_mw=round(change, 3),
            change_pct=round(change / last.demanda_mw * 100, 2) if last.demanda_mw else None,
            model=self.metadata["model"], window_hours=WINDOW_SIZE, source=request.source,
            warnings=warnings, inference_ms=round((time.perf_counter() - started) * 1000, 2),
        )


@lru_cache(maxsize=1)
def _cached_predictor() -> Predictor:
    return Predictor()


def get_predictor() -> Predictor:
    # Lazy loading evita cargar TensorFlow cuando solo se sirven HTML/metadatos.
    with _load_lock:
        try:
            return _cached_predictor()
        except Exception as exc:
            logger.exception("No se pudo cargar el modelo entrenado")
            raise ModelUnavailable("No se pudo cargar el modelo. Revisa los artefactos y las dependencias del servidor.") from exc
