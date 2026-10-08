from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from functools import lru_cache

import numpy as np

from backend.config import FEATURES, MODEL_DIR, TIMEZONE, WINDOW_SIZE, read_inference_parameters, read_metadata
from backend.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)
_load_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self) -> None:
        import onnxruntime as ort

        self.metadata = read_metadata()
        self.parameters = read_inference_parameters()
        self.feature_mean = np.asarray(self.parameters["feature_mean"], dtype=np.float64)
        self.feature_scale = np.asarray(self.parameters["feature_scale"], dtype=np.float64)
        self.target_mean = self.parameters["target_mean"]
        self.target_scale = self.parameters["target_scale"]
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(str(MODEL_DIR / "production_model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.model.get_inputs(), self.model.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[1:] != [24, 16] or len(outputs) != 1 or outputs[0].shape[1:] != [1]:
            raise ValueError("El modelo ONNX debe aceptar (batch, 24, 16) y devolver (batch, 1).")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        source_hash = self.model.get_modelmeta().custom_metadata_map.get("source_model_sha256")
        if source_hash != self.parameters["sources_sha256"]["production_model.keras"]:
            raise ValueError("El modelo ONNX no corresponde al artefacto Keras original.")
        self.engine = "onnxruntime"
        self._predict_lock = threading.Lock()
        # Calienta la inferencia una vez por proceso, usando entradas estandarizadas.
        self.model.run([self.output_name], {self.input_name: np.zeros((1, WINDOW_SIZE, len(FEATURES)), dtype=np.float32)})

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        started = time.perf_counter()
        values = np.array(
            [[getattr(row, feature) for feature in FEATURES] for row in request.observations], dtype=np.float64,
        )
        # Mismos parámetros y operaciones que StandardScaler.transform.
        scaled = (values - self.feature_mean) / self.feature_scale
        if not np.isfinite(scaled).all():
            raise ValueError("Los valores exceden el rango numérico admitido por el modelo.")
        with np.errstate(over="ignore", invalid="ignore"):
            tensor = np.asarray(scaled, dtype=np.float32).reshape(1, WINDOW_SIZE, len(FEATURES))
        if not np.isfinite(tensor).all():
            raise ValueError("Los valores exceden el rango numérico admitido por el modelo.")
        with self._predict_lock:
            raw = np.asarray(self.model.run([self.output_name], {self.input_name: tensor})[0], dtype=np.float64).reshape(1, 1)
        # Misma fórmula que target_scaler.inverse_transform; la salida se restaura a MW.
        demand = float(raw[0, 0] * self.target_scale + self.target_mean)
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
    # Se mantiene una sesión de inferencia por proceso; no se carga TensorFlow.
    with _load_lock:
        try:
            return _cached_predictor()
        except Exception as exc:
            logger.exception("No se pudo cargar el modelo entrenado")
            raise ModelUnavailable("No se pudo cargar el modelo. Revisa los artefactos y las dependencias del servidor.") from exc
