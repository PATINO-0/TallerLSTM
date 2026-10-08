from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.config import PERIOD_FEATURES, TIMEZONE, WINDOW_SIZE

NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Binary = Literal[0, 1]


class HourlyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    timestamp: datetime | None = Field(default=None, description="Hora en ISO 8601 con zona horaria; opcional.")
    demanda_mw: NonNegative = Field(description="Demanda observada durante esta hora, en MW; es una entrada histórica.")
    temperatura_c: Annotated[float, Field(ge=-100, le=100)]
    humedad_pct: Annotated[float, Field(ge=0, le=100)]
    viento_kmh: NonNegative
    radiacion_wm2: NonNegative
    precipitacion_mm: NonNegative
    precio_kwh: float = Field(description="Precio por kWh en la misma unidad monetaria del entrenamiento.")
    hora: Annotated[int, Field(ge=0, le=23)]
    dia_semana: Annotated[int, Field(ge=0, le=6)]
    fin_semana: Binary
    festivo: Binary
    mes: Annotated[int, Field(ge=1, le=12)]
    periodo_dia_madrugada: Binary
    periodo_dia_manana: Binary
    periodo_dia_tarde: Binary
    periodo_dia_noche: Binary

    @model_validator(mode="after")
    def validate_calendar(self) -> HourlyObservation:
        if sum(getattr(self, key) for key in PERIOD_FEATURES) != 1:
            raise ValueError("Selecciona exactamente un período del día por hora.")
        if self.fin_semana != int(self.dia_semana >= 5):
            raise ValueError("fin_semana debe coincidir con dia_semana (lunes=0, domingo=6).")
        if self.timestamp is not None:
            if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
                raise ValueError("timestamp debe incluir zona horaria, por ejemplo -05:00.")
            local = self.timestamp.astimezone(TIMEZONE)
            if local.minute or local.second or local.microsecond:
                raise ValueError("Cada timestamp debe corresponder a una hora exacta.")
            if (self.hora, self.dia_semana, self.mes) != (local.hour, local.weekday(), local.month):
                raise ValueError("hora, dia_semana y mes deben coincidir con timestamp en America/Bogota.")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: Annotated[list[HourlyObservation], Field(min_length=WINDOW_SIZE, max_length=WINDOW_SIZE)]
    source: Literal["manual", "csv", "demo"] = "manual"

    @model_validator(mode="after")
    def validate_sequence(self) -> PredictionRequest:
        timestamps = [row.timestamp for row in self.observations]
        if any(item is not None for item in timestamps) and not all(item is not None for item in timestamps):
            raise ValueError("Incluye timestamp en las 24 filas o en ninguna.")
        for previous, current in zip(self.observations, self.observations[1:]):
            if current.hora != (previous.hora + 1) % 24:
                raise ValueError("Las 24 observaciones deben ser horas consecutivas, de la más antigua a la más reciente.")
            expected_day = (previous.dia_semana + int(previous.hora == 23)) % 7
            if current.dia_semana != expected_day:
                raise ValueError("dia_semana debe avanzar al cambiar de día.")
            if previous.hora != 23 and previous.mes != current.mes:
                raise ValueError("mes solo puede cambiar al pasar de las 23:00 a las 00:00.")
            if previous.hora == 23 and current.mes not in {previous.mes, previous.mes % 12 + 1}:
                raise ValueError("La secuencia de meses no es válida.")
            if previous.timestamp is not None and current.timestamp is not None:
                if current.timestamp - previous.timestamp != timedelta(hours=1):
                    raise ValueError("Los timestamps deben estar separados exactamente por una hora.")
        return self


class PredictionResponse(BaseModel):
    predicted_demand_mw: float
    target: str
    unit: str
    prediction_timestamp: datetime | None
    next_hour: int
    last_demand_mw: float
    change_mw: float
    change_pct: float | None
    model: str
    window_hours: int
    source: str
    warnings: list[str]
    inference_ms: float
