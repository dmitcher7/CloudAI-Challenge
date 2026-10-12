"""Citi Bike-vraag: verwacht aantal vertrekken per zone per uur.

Wordt door ../app.py gekoppeld onder /citi_bike_demand. Het model is de bundle die
citi_bike/Kobe/train.py (of notebook 06) wegschrijft: een dict met 'model', 'model_name', 'features',
'metrics', 'zones', ... Een ander model gebruiken = model.pkl vervangen (of CITI_BIKE_DEMAND_MODEL_PATH
zetten) en herstarten.
"""
import os
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pandas.tseries.holiday import USFederalHolidayCalendar
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("CITI_BIKE_DEMAND_MODEL_PATH", HERE / "model.pkl"))

bundle = joblib.load(MODEL_PATH)
missing = {"model", "model_name", "features", "metrics", "zones"} - set(bundle)
if missing:
    raise ValueError(f"{MODEL_PATH.name} is geen geldige model-bundle, ontbreekt: {sorted(missing)}")
model = bundle["model"]
FEATURES = list(bundle["features"])
ZONES = {int(zone["zone"]): zone for zone in bundle["zones"]}


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Zelfde kenmerken als make_features() in citi_bike/Kobe/citibike.py (test: citi_bike/Kobe/test_features.py)."""
    time = pd.to_datetime(frame["time"])
    holidays = USFederalHolidayCalendar().holidays(start=time.min().normalize(), end=time.max().normalize())
    out = pd.DataFrame(index=frame.index)
    out["zone"] = frame["zone"].astype(int)
    out["hour"] = time.dt.hour
    out["weekday"] = time.dt.dayofweek
    out["month"] = time.dt.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["is_holiday"] = time.dt.normalize().isin(holidays).astype(int)
    out["temperature_c"] = frame["temperature_c"].astype(float)
    out["precipitation_mm"] = frame["precipitation_mm"].astype(float)
    out["wind_kmh"] = frame["wind_kmh"].astype(float)
    out["year"] = time.dt.year
    if "active_stations" in frame:
        out["active_stations"] = frame["active_stations"].astype(int)
    if "snow_depth_cm" in frame:
        out["snow_depth_cm"] = frame["snow_depth_cm"].astype(float)
    return out[FEATURES]


def predict_frame(frame: pd.DataFrame) -> np.ndarray:
    # Voorspellingen gaan over nu of de toekomst: we gebruiken het huidige aantal actieve stations per zone.
    if "active_stations" in FEATURES:
        frame = frame.assign(active_stations=frame["zone"].map(lambda z: ZONES[int(z)]["active_stations"]))
    return np.maximum(0, model.predict(make_features(frame)))


class Weather(BaseModel):
    temperature_c: float = Field(ge=-30, le=45, examples=[18.0])
    precipitation_mm: float = Field(ge=0, le=100, examples=[0.0])
    wind_kmh: float = Field(ge=0, le=150, examples=[12.0])
    snow_depth_cm: float = Field(default=0, ge=0, le=300, examples=[0.0],
                                 description="Sneeuw op de grond (cm); 0 als er geen sneeuw ligt")


class HourRequest(Weather):
    zone: int = Field(ge=0, examples=[0])
    time: datetime = Field(examples=["2024-06-11T08:00:00"])


class DayRequest(BaseModel):
    zone: int = Field(ge=0, examples=[0])
    date: Date = Field(examples=["2024-06-11"])
    weather: list[Weather] = Field(min_length=24, max_length=24, description="Weer per uur, 00:00 t/m 23:00")


def check_zone(zone: int):
    if zone not in ZONES:
        raise HTTPException(status_code=422, detail=f"Onbekende zone {zone}; geldig: 0 t/m {max(ZONES)}")


router = APIRouter(tags=["citi_bike_demand"])


@router.get("/")
def info():
    return {
        "model": bundle["model_name"],
        "model_file": MODEL_PATH.name,
        "target": bundle.get("target", "vertrekken per zone per uur"),
        "features": FEATURES,
        "trained_at_utc": bundle.get("trained_at_utc"),
        "metrics": bundle["metrics"],
        "limitations": bundle.get("limitations"),
        "zones": list(ZONES.values()),
    }


@router.post("/predict")
def predict(request: HourRequest):
    check_zone(request.zone)
    row = pd.DataFrame([request.model_dump()])
    trips = float(predict_frame(row)[0])
    return {
        "zone": request.zone,
        "zone_name": ZONES[request.zone]["name"],
        "time": request.time.isoformat(),
        "predicted_trips": round(trips, 1),
        "holdout_mae_trips": bundle["metrics"].get("mae"),
        "model_name": bundle["model_name"],
    }


@router.post("/predict_day")
def predict_day(request: DayRequest):
    check_zone(request.zone)
    frame = pd.DataFrame([w.model_dump() for w in request.weather])
    frame["zone"] = request.zone
    frame["time"] = pd.date_range(pd.Timestamp(request.date), periods=24, freq="h")
    trips = predict_frame(frame)
    return {
        "zone": request.zone,
        "zone_name": ZONES[request.zone]["name"],
        "date": request.date.isoformat(),
        "hourly_trips": [round(float(t), 1) for t in trips],
        "total_trips": round(float(trips.sum())),
        "holdout_mae_trips": bundle["metrics"].get("mae"),
        "model_name": bundle["model_name"],
    }
