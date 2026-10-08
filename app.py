from __future__ import annotations

import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.config import FEATURES, FRONTEND_DIR, TIMEZONE, WINDOW_SIZE, read_metadata, read_price_reference
from backend.examples import example_payload, template_csv
from backend.inference import ModelUnavailable, get_predictor
from backend.schemas import PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="Energy · Predicción de demanda energética",
    description="Inferencia del modelo LSTM entrenado: 24 horas × 16 variables → demanda de la siguiente hora en MW.",
    version="1.0.0",
)
origins = [item.strip() for item in os.environ.get("CORS_ORIGINS", "").split(",") if item.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    # No se devuelve el objeto ctx de Pydantic (puede contener excepciones no serializables).
    details = [{"location": list(item["loc"]), "message": item["msg"], "type": item["type"]} for item in error.errors()]
    return JSONResponse(status_code=422, content={"message": "Revisa los datos ingresados.", "errors": details})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health", tags=["Modelo"])
def health() -> dict:
    try:
        predictor = get_predictor()
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model_ready": True, "model": predictor.metadata["model"], "engine": predictor.engine, "input_shape": [1, 24, 16]}


@app.get("/api/metadata", tags=["Modelo"])
def metadata() -> dict:
    try:
        saved = read_metadata()
        price_reference = read_price_reference()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="No se pueden leer los metadatos del modelo.") from exc
    return {
        **saved, "target": "demanda_objetivo", "unit": "MW", "timezone": str(TIMEZONE),
        "input_shape": [1, WINDOW_SIZE, len(FEATURES)], "features": FEATURES,
        "price_training_reference": price_reference,
        "calendar_convention": {
            "dia_semana": "lunes=0, domingo=6", "fin_semana": "sábado y domingo=1",
            "period_defaults": "madrugada 00–05, mañana 06–11, tarde 12–17, noche 18–23",
            "note": "Los cortes de período y la moneda del precio no están documentados en los artefactos. Comprueba la convención del entrenamiento; el período es editable.",
            "festivo": "Se ingresa manualmente; no se infieren festivos.",
        },
    }


@app.get("/api/example", tags=["Datos"])
def example() -> dict:
    return example_payload()


@app.get("/api/template.csv", tags=["Datos"])
def template(layout: Literal["simple", "full"] = "simple") -> Response:
    return Response(content="\ufeff" + template_csv(layout), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="plantilla_24_horas.csv"'})


@app.post("/api/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        return get_predictor().predict(payload)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Vercel promueve este directorio al CDN. Localmente lo sirve FastAPI.
# La ruta raíz está antes del mount para servir index.html explícitamente.
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
