"""Convierte los artefactos originales a ONNX y JSON, con verificación numérica.

Ejecutar localmente con requirements-export.txt. Los archivos originales se leen
sin modificarse. No se reentrena, cuantiza ni ajusta el modelo.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import joblib
import keras
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
MAX_ERROR_MW = 0.002


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def keras_to_onnx(model) -> onnx.ModelProto:
    if [type(layer).__name__ for layer in model.layers] != ["LSTM", "Dense", "Dense"]:
        raise ValueError("La conversión admite la arquitectura LSTM → Dense → Dense de este proyecto.")
    if tuple(model.input_shape) != (None, 24, 16) or tuple(model.output_shape) != (None, 1):
        raise ValueError("Las dimensiones del modelo original han cambiado.")
    lstm, dense, output = model.layers
    config = lstm.get_config()
    required = {
        "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": True,
        "return_sequences": False, "return_state": False, "go_backwards": False,
        "stateful": False, "dropout": 0.0, "recurrent_dropout": 0.0,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("La configuración LSTM no coincide con la conversión validada.")
    if dense.get_config()["activation"] != "relu" or output.get_config()["activation"] != "linear":
        raise ValueError("Las activaciones Dense han cambiado.")
    if any(layer.compute_dtype != "float32" for layer in model.layers):
        raise ValueError("La conversión conserva pesos float32; la política de precisión ha cambiado.")

    kernel, recurrent, bias = lstm.get_weights()
    units = lstm.units

    def reorder(values, axis):
        # Keras: input, forget, cell, output. ONNX: input, output, forget, cell.
        gates = np.split(values, 4, axis=axis)
        return np.concatenate([gates[index] for index in [0, 3, 1, 2]], axis=axis)

    weights = reorder(kernel, 1).T[None, :, :]
    recurrent_weights = reorder(recurrent, 1).T[None, :, :]
    input_bias = reorder(bias, 0)
    onnx_bias = np.concatenate([input_bias, np.zeros_like(input_bias)])[None, :]
    dense_kernel, dense_bias = dense.get_weights()
    output_kernel, output_bias = output.get_weights()
    arrays = {
        "lstm_w": weights, "lstm_r": recurrent_weights, "lstm_b": onnx_bias,
        "squeeze_axes": np.array([0], dtype=np.int64),
        "dense_w": dense_kernel, "dense_b": dense_bias,
        "output_w": output_kernel, "output_b": output_bias,
    }
    nodes = [
        helper.make_node("Transpose", ["observations"], ["sequence"], perm=[1, 0, 2]),
        helper.make_node(
            "LSTM", ["sequence", "lstm_w", "lstm_r", "lstm_b"],
            ["sequence_output", "last_hidden", "last_cell"],
            hidden_size=units, direction="forward", activations=["Sigmoid", "Tanh", "Tanh"],
        ),
        helper.make_node("Squeeze", ["last_hidden", "squeeze_axes"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "dense_w"], ["dense_product"]),
        helper.make_node("Add", ["dense_product", "dense_b"], ["dense_sum"]),
        helper.make_node("Relu", ["dense_sum"], ["dense_activation"]),
        helper.make_node("MatMul", ["dense_activation", "output_w"], ["output_product"]),
        helper.make_node("Add", ["output_product", "output_b"], ["scaled_demand"]),
    ]
    graph = helper.make_graph(
        nodes, "energy_lstm",
        [helper.make_tensor_value_info("observations", TensorProto.FLOAT, ["batch", 24, 16])],
        [helper.make_tensor_value_info("scaled_demand", TensorProto.FLOAT, ["batch", 1])],
        initializer=[numpy_helper.from_array(np.ascontiguousarray(array), name) for name, array in arrays.items()],
    )
    converted = helper.make_model(graph, producer_name="energy-export", opset_imports=[helper.make_opsetid("", 17)])
    converted.ir_version = 10
    onnx.checker.check_model(converted, full_check=True)
    return converted


def main():
    sys.path.insert(0, str(ROOT))
    from backend.examples import example_payload

    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    preprocessing = joblib.load(ROOT / "preprocessing.pkl")
    columns = preprocessing["feature_columns"]
    if columns != metadata["features"] or len(columns) != 16 or preprocessing["window_size"] != 24:
        raise ValueError("El modelo y el preprocesamiento deben usar 24 × 16 con el mismo orden de variables.")
    feature_scaler, target_scaler = preprocessing["feature_scaler"], preprocessing["target_scaler"]
    if list(feature_scaler.feature_names_in_) != columns:
        raise ValueError("El orden de las variables del escalador no coincide.")
    if not feature_scaler.with_mean or not feature_scaler.with_std or not target_scaler.with_mean or not target_scaler.with_std:
        raise ValueError("Los escaladores deben ser StandardScaler con media y desviación estándar.")
    model = keras.saving.load_model(ROOT / "production_model.keras", compile=False, safe_mode=True)
    sources = {name: sha256((ROOT / name).read_bytes()) for name in ["production_model.keras", "preprocessing.pkl", "metadata.json"]}
    converted = keras_to_onnx(model)
    helper.set_model_props(converted, {"source_model_sha256": sources["production_model.keras"], "model_name": metadata["model"]})
    model_bytes = converted.SerializeToString()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(20261007)
    tensors = rng.normal(0, 1.5, (128, 24, 16)).astype(np.float32)
    tensors = np.concatenate([tensors, np.zeros((1, 24, 16), np.float32), np.full((1, 24, 16), 6, np.float32), np.full((1, 24, 16), -6, np.float32)])
    reference_scaled = np.asarray(model(tensors, training=False), dtype=np.float64)
    actual_scaled = session.run(["scaled_demand"], {"observations": tensors})[0].astype(np.float64)
    reference_mw = target_scaler.inverse_transform(reference_scaled)
    actual_mw = target_scaler.inverse_transform(actual_scaled)
    maximum_error = float(np.max(np.abs(reference_mw - actual_mw)))
    if maximum_error > MAX_ERROR_MW:
        raise ValueError(f"La diferencia ONNX/Keras es {maximum_error} MW; excede {MAX_ERROR_MW} MW.")

    golden_cases = []
    for end in ["2026-06-10T23:00:00-05:00", "2026-10-07T23:00:00-05:00", "2026-01-01T10:00:00-05:00"]:
        payload = example_payload(datetime.fromisoformat(end))
        inputs = np.array([[row[name] for name in columns] for row in payload["observations"]], dtype=np.float64)
        scaled = (inputs - feature_scaler.mean_) / feature_scaler.scale_
        # Comprueba que el escalado ligero conserva exactamente la fórmula original.
        np.testing.assert_allclose(scaled, feature_scaler.transform(pd.DataFrame(inputs, columns=columns)), atol=1e-12, rtol=0)
        tensor = scaled.astype(np.float32)[None, :, :]
        expected = float(target_scaler.inverse_transform(np.asarray(model(tensor, training=False), dtype=np.float64))[0, 0])
        actual = float(target_scaler.inverse_transform(session.run(None, {"observations": tensor})[0].astype(np.float64))[0, 0])
        difference = abs(expected - actual)
        if difference > MAX_ERROR_MW:
            raise ValueError(f"La diferencia de la petición de ejemplo excede {MAX_ERROR_MW} MW.")
        maximum_error = max(maximum_error, difference)
        golden_cases.append({"end": end, "expected_demand_mw": expected, "onnx_demand_mw": actual})

    edge_errors = []
    baseline = example_payload(datetime.fromisoformat("2026-06-10T23:00:00-05:00"))
    inputs = np.array([[row[name] for name in columns] for row in baseline["observations"]], dtype=np.float64)
    edge_inputs = []
    zero_last = inputs.copy()
    zero_last[-1, 0] = 0
    edge_inputs.append(zero_last)
    expensive = inputs.copy()
    expensive[:, columns.index("precio_kwh")] = 810
    edge_inputs.append(expensive)
    high_wind = inputs.copy()
    high_wind[:, columns.index("viento_kmh")] = 99
    edge_inputs.append(high_wind)
    for inputs in edge_inputs:
        tensor = ((inputs - feature_scaler.mean_) / feature_scaler.scale_).astype(np.float32)[None, :, :]
        expected = float(target_scaler.inverse_transform(np.asarray(model(tensor, training=False), dtype=np.float64))[0, 0])
        actual = float(target_scaler.inverse_transform(session.run(None, {"observations": tensor})[0].astype(np.float64))[0, 0])
        edge_errors.append(abs(expected - actual))
    maximum_error = max(maximum_error, *edge_errors)
    if maximum_error > MAX_ERROR_MW:
        raise ValueError("La diferencia numérica excede la tolerancia en una entrada extrema.")

    light = {
        "format_version": 1, "model_name": metadata["model"], "feature_columns": columns, "window_size": 24,
        "feature_mean": feature_scaler.mean_.tolist(), "feature_scale": feature_scaler.scale_.tolist(),
        "target_mean": float(target_scaler.mean_[0]), "target_scale": float(target_scaler.scale_[0]),
        "sources_sha256": sources, "onnx_sha256": sha256(model_bytes),
    }
    report = {
        "format_version": 1, "sources_sha256": sources, "onnx_sha256": light["onnx_sha256"],
        "keras_version": keras.__version__, "onnxruntime_version": ort.__version__,
        "validation_windows": len(tensors) + len(golden_cases) + len(edge_errors), "random_seed": 20261007,
        "maximum_absolute_error_mw": maximum_error, "allowed_absolute_error_mw": MAX_ERROR_MW,
        "golden_cases": golden_cases,
        "edge_cases": ["zero_last_demand", "price_810", "wind_99"],
    }
    # Publica los artefactos únicamente después de superar las verificaciones.
    (ROOT / "production_model.onnx").write_bytes(model_bytes)
    (ROOT / "inference_preprocessing.json").write_text(json.dumps(light, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "model_conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Modelo ONNX: {len(model_bytes):,} bytes. Ventanas verificadas: {report['validation_windows']}.")
    print(f"Diferencia máxima frente a Keras: {maximum_error:.9f} MW (límite: {MAX_ERROR_MW} MW).")


if __name__ == "__main__":
    main()
