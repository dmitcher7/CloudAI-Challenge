"""Create model features known at trip start."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.config import MODEL_FEATURES
from src.data.weather import WEATHER_FEATURES
from src.features.history import HISTORY_FEATURES


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclic time features without using any post-departure columns."""
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
    holidays = (
        USFederalHolidayCalendar().holidays(start=started.min(), end=started.max())
        if started.notna().any()
        else pd.DatetimeIndex([])
    )
    result["is_holiday"] = started.dt.normalize().isin(holidays).astype("int8")

    lat1 = np.radians(pd.to_numeric(result["start_lat"], errors="coerce"))
    lng1 = np.radians(pd.to_numeric(result["start_lng"], errors="coerce"))
    lat2 = np.radians(pd.to_numeric(result["end_lat"], errors="coerce"))
    lng2 = np.radians(pd.to_numeric(result["end_lng"], errors="coerce"))
    delta_lat_radians = lat2 - lat1
    delta_lng_radians = lng2 - lng1
    haversine = np.sin(delta_lat_radians / 2) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin(delta_lng_radians / 2) ** 2
    )
    result["direct_distance_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(haversine.clip(0, 1)))
    result["delta_lat"] = pd.to_numeric(result["end_lat"], errors="coerce") - pd.to_numeric(
        result["start_lat"], errors="coerce"
    )
    result["delta_lng"] = pd.to_numeric(result["end_lng"], errors="coerce") - pd.to_numeric(
        result["start_lng"], errors="coerce"
    )
    north_km = result["delta_lat"] * 111.0
    east_km = result["delta_lng"] * 84.0
    grid_angle = np.radians(29.0)
    grid_x = east_km * np.cos(grid_angle) + north_km * np.sin(grid_angle)
    grid_y = -east_km * np.sin(grid_angle) + north_km * np.cos(grid_angle)
    result["grid_distance_km"] = grid_x.abs() + grid_y.abs()
    bearing = np.arctan2(east_km, north_km)
    result["bearing_sin"] = np.sin(bearing)
    result["bearing_cos"] = np.cos(bearing)

    for column in ["rideable_type", "member_casual", "start_station_id", "end_station_id"]:
        result[column] = result[column].astype("string").fillna("unknown")
    for column in [
        "start_lat", "start_lng", "end_lat", "end_lng", *WEATHER_FEATURES, *HISTORY_FEATURES
    ]:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[MODEL_FEATURES]
