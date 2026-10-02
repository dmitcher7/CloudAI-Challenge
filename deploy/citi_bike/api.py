"""Citi Bike-model: ritduur (minuten) op basis van wat bij vertrek bekend is.

Wordt door ../app.py gekoppeld onder /citi_bike. Het model is de bundle die
citi_bike/Brent/src/models/train.py wegschrijft (dict met 'model', 'model_name',
'features', 'metrics', ...); een ander model gebruiken = model.pkl vervangen
(of CITI_BIKE_MODEL_PATH zetten) en herstarten.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("CITI_BIKE_MODEL_PATH", HERE / "model.pkl"))

bundle = joblib.load(MODEL_PATH)
missing = {"model", "model_name", "features", "metrics"} - set(bundle)
if missing:
    raise ValueError(f"{MODEL_PATH.name} is geen geldige model-bundle, ontbreekt: {sorted(missing)}")
model = bundle["model"]
FEATURES = list(bundle["features"])
MAE = bundle["metrics"].get("mae_minutes")


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Zelfde kenmerken als citi_bike/Brent/src/features/build.py (cyclische tijd, geen leakage)."""
    result = frame.copy()
    started = pd.to_datetime(result["started_at"], errors="coerce")
    hour = started.dt.hour + started.dt.minute / 60
    weekday = started.dt.dayofweek
    month = started.dt.month - 1

    result["start_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["start_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    result["is_weekend"] = (weekday >= 5).astype("int8")

    for column in ["rideable_type", "member_casual", "start_station_id"]:
        result[column] = result[column].astype("string").fillna("unknown")
    for column in ["start_lat", "start_lng"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[FEATURES]


class TripStart(BaseModel):
    started_at: str = Field(examples=["2024-06-15T09:30:00"])
    rideable_type: str = Field(min_length=1, max_length=50, examples=["classic_bike"])
    member_casual: str = Field(pattern="^(member|casual)$", examples=["member"])
    start_station_id: str = Field(min_length=1, max_length=100, examples=["HB101"])
    start_lat: float = Field(ge=40.4, le=41.1, examples=[40.7359])
    start_lng: float = Field(ge=-74.4, le=-73.5, examples=[-74.0303])

    @field_validator("started_at")
    @classmethod
    def valid_datetime(cls, value: str) -> str:
        try:
            pd.Timestamp(value)
        except ValueError as exc:
            raise ValueError("started_at moet een ISO-8601 datum/tijd zijn") from exc
        return value


router = APIRouter(tags=["citi_bike"])


@router.get("/")
def info():
    return {
        "model": bundle["model_name"],
        "model_file": MODEL_PATH.name,
        "target": bundle.get("target", "duration_minutes"),
        "features": FEATURES,
        "trained_at_utc": bundle.get("trained_at_utc"),
        "metrics": bundle["metrics"],
        "limitations": bundle.get("limitations"),
    }


@router.post("/predict")
def predict(trip: TripStart):
    row = pd.DataFrame([trip.model_dump()])
    minutes = float(np.maximum(0, model.predict(make_features(row)))[0])
    return {
        "predicted_duration_minutes": round(minutes, 1),
        "model_name": bundle["model_name"],
        "holdout_mae_minutes": round(float(MAE), 2) if MAE is not None else None,
        "message": "Schatting, geen garantie: weer, evenementen en beschikbaarheid ontbreken.",
    }
