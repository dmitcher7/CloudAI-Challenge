from pathlib import Path

import pandas as pd

from src.data.weather import WEATHER_FEATURES, attach_hourly_weather, parse_hourly_weather


def test_parse_and_attach_hourly_weather(tmp_path: Path):
    payload = {
        "hourly": {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [5.0, 6.0],
            "relative_humidity_2m": [70.0, 68.0],
            "precipitation": [0.0, 1.2],
            "wind_speed_10m": [12.0, 14.0],
        }
    }
    weather = parse_hourly_weather(payload)
    path = tmp_path / "weather.parquet"
    weather.to_parquet(path, index=False)
    trips = pd.DataFrame({"started_at": ["2024-01-01T10:37:00"]})
    enriched = attach_hourly_weather(trips, path)
    assert enriched.loc[0, "temperature_2m"] == 5.0
    assert enriched.loc[0, "precipitation"] == 0.0


def test_missing_weather_cache_produces_imputable_values(tmp_path: Path):
    trips = pd.DataFrame({"started_at": ["2024-01-01T10:37:00"]})
    enriched = attach_hourly_weather(trips, tmp_path / "missing.parquet")
    assert enriched[WEATHER_FEATURES].isna().all().all()
