"""FastAPI service for Citi Bike trip-duration inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.config import MODEL_DIR
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
    row = pd.DataFrame([trip.model_dump()])
    minutes = float(predict_rows(bundle, row)[0])
    mae = bundle.get("metrics", {}).get("mae_minutes")
    return Prediction(
        predicted_duration_minutes=round(minutes, 1),
        model_name=bundle["model_name"],
        holdout_mae_minutes=round(float(mae), 2) if mae is not None else None,
        message="Schatting, geen garantie: weer, evenementen en beschikbaarheid ontbreken.",
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

