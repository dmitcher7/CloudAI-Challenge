"""Create model features known at trip start."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES


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

    for column in ["rideable_type", "member_casual", "start_station_id"]:
        result[column] = result[column].astype("string").fillna("unknown")
    for column in ["start_lat", "start_lng"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[MODEL_FEATURES]

