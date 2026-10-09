"""FastAPI service for Citi Bike trip-duration inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.config import MODEL_DIR
from src.data.weather import WEATHER_FEATURES, forecast_weather_at
from src.models.predict import load_bundle, predict_rows

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "duration_model.joblib")))
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Green Wheels — Citi Bike Duration API",
    version="0.1.0",
    description="Predict trip duration using only information available at departure.",
)


class TripStart(BaseModel):
    started_at: str = Field(examples=["2024-06-15T09:30:00"])
    rideable_type: str = Field(min_length=1, max_length=50, examples=["classic_bike"])
    member_casual: str = Field(pattern="^(member|casual)$")
    start_station_id: str = Field(min_length=1, max_length=100, examples=["HB101"])
    start_lat: float = Field(ge=40.4, le=41.1)
    start_lng: float = Field(ge=-74.4, le=-73.5)
    end_station_id: str = Field(min_length=1, max_length=100, examples=["JC115"])
    end_lat: float = Field(ge=40.4, le=41.1)
    end_lng: float = Field(ge=-74.4, le=-73.5)
    temperature_2m: float | None = Field(default=None, ge=-40, le=60)
    relative_humidity_2m: float | None = Field(default=None, ge=0, le=100)
    precipitation: float | None = Field(default=None, ge=0, le=200)
    wind_speed_10m: float | None = Field(default=None, ge=0, le=250)

    @field_validator("started_at")
    @classmethod
    def valid_datetime(cls, value: str) -> str:
        try:
            pd.Timestamp(value)
        except ValueError as exc:
            raise ValueError("started_at must be an ISO-8601 datetime") from exc
        return value


class Prediction(BaseModel):
    predicted_duration_minutes: float
    model_name: str
    holdout_mae_minutes: float | None
    weather_source: str
    message: str


@lru_cache(maxsize=1)
def model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return load_bundle(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    try:
        bundle = model_bundle()
    except (FileNotFoundError, ValueError):
        return {"status": "not_ready", "model_path": str(MODEL_PATH)}
    return {
        "status": "ok",
        "model_name": bundle["model_name"],
        "trained_at_utc": bundle.get("trained_at_utc"),
    }


@app.post("/predict", response_model=Prediction)
def predict(trip: TripStart) -> Prediction:
    try:
        bundle = model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained. Run python -m src.models.train first.",
        ) from exc
    row_data = trip.model_dump()
    missing_weather = [name for name in WEATHER_FEATURES if row_data.get(name) is None]
    weather_source = "request"
    if missing_weather:
        try:
            weather = forecast_weather_at(trip.started_at, trip.start_lat, trip.start_lng)
            row_data.update(weather)
            weather_source = "open-meteo forecast"
        except (requests.RequestException, ValueError, KeyError):
            weather_source = "training median fallback"
    row = pd.DataFrame([row_data])
    minutes = float(predict_rows(bundle, row)[0])
    mae = bundle.get("metrics", {}).get("mae_minutes")
    return Prediction(
        predicted_duration_minutes=round(minutes, 1),
        model_name=bundle["model_name"],
        holdout_mae_minutes=round(float(mae), 2) if mae is not None else None,
        weather_source=weather_source,
        message="Schatting op basis van geplande bestemming; route, verkeer en evenementen blijven onzeker.",
    )


@app.get("/model-card")
def model_card() -> dict:
    try:
        bundle = model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model is not trained.") from exc
    return {key: value for key, value in bundle.items() if key != "model"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
