"""Citi Bike-model: ritduur (minuten) op basis van wat bij vertrek bekend is, inclusief de geplande bestemming.

Wordt door ../app.py gekoppeld onder /citi_bike. Het model is de bundle die
citi_bike/Brent/src/models/train.py wegschrijft (dict met 'model', 'model_name', 'features', 'metrics',
'route_history', ...). Een ander model gebruiken = model.pkl vervangen (of CITI_BIKE_MODEL_PATH zetten) en herstarten.
Het model moet opgeslagen zijn met dezelfde scikit-learn- en numpy-versie als deploy/requirements.txt.

De kenmerken worden berekend zoals in citi_bike/Brent/src/features/build.py en history.py (die code gaat niet mee
naar de server, daarom staat ze hier opnieuw).
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from pandas.tseries.holiday import USFederalHolidayCalendar
from pydantic import BaseModel, Field, field_validator

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("CITI_BIKE_MODEL_PATH", HERE / "model.pkl"))
STATIONS_PATH = HERE / "stations.json"
MAX_DURATION_MINUTES = 180.0          # het geldige bereik van de training (1-180 minuten)

bundle = joblib.load(MODEL_PATH)
missing = {"model", "model_name", "features", "metrics", "route_history"} - set(bundle)
if missing:
    raise ValueError(f"{MODEL_PATH.name} is geen geldige model-bundle, ontbreekt: {sorted(missing)}")
model = bundle["model"]
FEATURES = list(bundle["features"])
HISTORY = bundle["route_history"]
MAE = bundle["metrics"].get("mae_minutes")
STATIONS = json.loads(STATIONS_PATH.read_text(encoding="utf-8")) if STATIONS_PATH.exists() else []
WEATHER_FEATURES = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
CATEGORICAL = ["rideable_type", "member_casual", "start_station_id", "end_station_id"]


def _key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("unknown").astype(str)


def apply_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Typische duur op de route en van/naar de stations, uit de trainingsperiode (met smoothing)."""
    result = frame.copy()
    global_mean, smoothing = float(HISTORY["global_mean"]), float(HISTORY.get("smoothing", 20.0))

    def estimate(keys: pd.Series, mapping: dict) -> tuple[np.ndarray, np.ndarray]:
        records = keys.map(mapping)
        means = records.map(lambda value: value[0] if isinstance(value, tuple) else global_mean).to_numpy(dtype=float)
        counts = records.map(lambda value: value[1] if isinstance(value, tuple) else 0).to_numpy(dtype=float)
        return (means * counts + smoothing * global_mean) / (counts + smoothing), counts

    route, route_count = estimate(_key(result["start_station_id"]) + "\x1f" + _key(result["end_station_id"]), HISTORY["route"])
    result["route_typical_minutes"] = route
    result["route_history_log_count"] = np.log1p(route_count)
    result["start_typical_minutes"] = estimate(_key(result["start_station_id"]), HISTORY["start"])[0]
    result["end_typical_minutes"] = estimate(_key(result["end_station_id"]), HISTORY["end"])[0]
    return result


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Zelfde kenmerken als citi_bike/Brent/src/features/build.py (geen informatie van na het vertrek)."""
    result = apply_history(frame)
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
    # Op hele dagen zoeken: met één vertrekmoment (bv. 4 juli 8u30) zou een bereik tot op de minuut de feestdag missen.
    holidays = USFederalHolidayCalendar().holidays(start=started.min().normalize(), end=started.max().normalize())
    result["is_holiday"] = started.dt.normalize().isin(holidays).astype("int8")

    lat1, lng1 = np.radians(result["start_lat"].astype(float)), np.radians(result["start_lng"].astype(float))
    lat2, lng2 = np.radians(result["end_lat"].astype(float)), np.radians(result["end_lng"].astype(float))
    haversine = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    result["direct_distance_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(haversine.clip(0, 1)))
    result["delta_lat"] = result["end_lat"].astype(float) - result["start_lat"].astype(float)
    result["delta_lng"] = result["end_lng"].astype(float) - result["start_lng"].astype(float)
    north_km, east_km = result["delta_lat"] * 111.0, result["delta_lng"] * 84.0
    grid_angle = np.radians(29.0)                      # het stratenrooster van Manhattan staat 29° gedraaid
    grid_x = east_km * np.cos(grid_angle) + north_km * np.sin(grid_angle)
    grid_y = -east_km * np.sin(grid_angle) + north_km * np.cos(grid_angle)
    result["grid_distance_km"] = grid_x.abs() + grid_y.abs()
    bearing = np.arctan2(east_km, north_km)
    result["bearing_sin"], result["bearing_cos"] = np.sin(bearing), np.cos(bearing)

    for column in CATEGORICAL:
        result[column] = result[column].astype("string").fillna("unknown").astype(object)
    for column in WEATHER_FEATURES:                    # ontbrekend weer: het model vult de mediaan uit de training in
        values = result[column] if column in result else pd.Series(np.nan, index=result.index)
        result[column] = pd.to_numeric(values, errors="coerce").astype(float)
    return result[FEATURES]


class TripStart(BaseModel):
    started_at: str = Field(examples=["2026-06-15T08:30:00"])
    rideable_type: str = Field(pattern="^(classic_bike|electric_bike)$", examples=["classic_bike"])
    member_casual: str = Field(pattern="^(member|casual)$", examples=["member"])
    start_station_id: str = Field(min_length=1, max_length=100, examples=["6140.05"])
    start_lat: float = Field(ge=40.4, le=41.1, examples=[40.74174])
    start_lng: float = Field(ge=-74.4, le=-73.5, examples=[-73.99416])
    end_station_id: str = Field(min_length=1, max_length=100, examples=["5905.12"])
    end_lat: float = Field(ge=40.4, le=41.1, examples=[40.73497])
    end_lng: float = Field(ge=-74.4, le=-73.5, examples=[-73.99073])
    temperature_2m: float | None = Field(default=None, ge=-40, le=60, examples=[20.0])
    relative_humidity_2m: float | None = Field(default=None, ge=0, le=100, examples=[60.0])
    precipitation: float | None = Field(default=None, ge=0, le=200, examples=[0.0])
    wind_speed_10m: float | None = Field(default=None, ge=0, le=250, examples=[10.0])

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
        "training_period": bundle.get("training_period"),
        "holdout_period": bundle.get("holdout_period"),
        "metrics": bundle["metrics"],
        "runtime_versions": bundle.get("runtime_versions"),
        "limitations": bundle.get("limitations"),
    }


@router.get("/stations")
def stations():
    """Huidige publieke stations (id, naam, coördinaten) voor de webpagina."""
    return STATIONS


@router.post("/predict")
def predict(trip: TripStart):
    row = pd.DataFrame([trip.model_dump()])
    features = make_features(row)
    minutes = float(np.clip(model.predict(features), 0, MAX_DURATION_MINUTES)[0])
    weather_given = all(getattr(trip, name) is not None for name in WEATHER_FEATURES)
    return {
        "predicted_duration_minutes": round(minutes, 1),
        "direct_distance_km": round(float(features["direct_distance_km"].iloc[0]), 2),
        "route_trips_in_history": int(np.expm1(features["route_history_log_count"].iloc[0]).round()),
        "weather_source": "meegegeven" if weather_given else "ontbreekt: mediaan uit de training",
        "model_name": bundle["model_name"],
        "holdout_mae_minutes": round(float(MAE), 2) if MAE is not None else None,
        "message": "Schatting, geen garantie: de echte route, tussenstops, verkeer en evenementen zijn onbekend.",
    }
