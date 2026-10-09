"""Download and join hourly NYC weather for training and live inference.

Historical data comes from Open-Meteo's archive endpoint. The live helper uses
the forecast endpoint and gracefully lets callers fall back when a requested
time is unavailable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import DEFAULT_END_MONTH, DEFAULT_START_MONTH, WEATHER_PATH

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060
NYC_TIMEZONE = "America/New_York"
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]


def _month_start(value: str) -> str:
    return f"{value}-01"


def _month_end(value: str) -> str:
    period = pd.Period(value, freq="M")
    return period.end_time.date().isoformat()


def parse_hourly_weather(payload: dict) -> pd.DataFrame:
    """Validate an Open-Meteo response and return one row per local hour."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Weather response has no hourly time series")
    expected = len(hourly["time"])
    missing = [name for name in WEATHER_FEATURES if name not in hourly]
    if missing:
        raise ValueError(f"Weather response is missing variables: {missing}")
    if any(len(hourly[name]) != expected for name in WEATHER_FEATURES):
        raise ValueError("Weather variables have inconsistent lengths")

    frame = pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(hourly["time"], errors="raise"),
            **{name: pd.to_numeric(hourly[name], errors="coerce") for name in WEATHER_FEATURES},
        }
    )
    return frame.drop_duplicates("weather_hour").sort_values("weather_hour")


def _weather_params(start_date: str, end_date: str) -> dict:
    return {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_FEATURES),
        "timezone": NYC_TIMEZONE,
    }


def download_hourly_weather(
    start_month: str = DEFAULT_START_MONTH,
    end_month: str = DEFAULT_END_MONTH,
    output_path: Path = WEATHER_PATH,
) -> pd.DataFrame:
    """Download an hourly weather table atomically for the selected months."""
    start_date = _month_start(start_month)
    end_date = _month_end(end_month)
    params = _weather_params(start_date, end_date)
    with requests.get(ARCHIVE_URL, params=params, timeout=(15, 180)) as response:
        response.raise_for_status()
        frame = parse_hourly_weather(response.json())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "source": ARCHIVE_URL,
                "latitude": NYC_LATITUDE,
                "longitude": NYC_LONGITUDE,
                "timezone": NYC_TIMEZONE,
                "start_date": start_date,
                "end_date": end_date,
                "rows": len(frame),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(frame):,} hourly weather rows to {output_path}")
    return frame


def attach_hourly_weather(
    frame: pd.DataFrame, weather_path: Path = WEATHER_PATH
) -> pd.DataFrame:
    """Join local-hour weather; missing cache produces imputable NaN values."""
    result = frame.copy()
    if all(name in result for name in WEATHER_FEATURES):
        for name in WEATHER_FEATURES:
            result[name] = pd.to_numeric(result[name], errors="coerce")
        return result
    if not weather_path.exists():
        for name in WEATHER_FEATURES:
            result[name] = np.nan
        return result

    weather = pd.read_parquet(weather_path, columns=["weather_hour", *WEATHER_FEATURES])
    weather["weather_hour"] = pd.to_datetime(weather["weather_hour"], errors="coerce")
    result["_weather_hour"] = pd.to_datetime(result["started_at"], errors="coerce").dt.floor("h")
    result = result.merge(
        weather,
        how="left",
        left_on="_weather_hour",
        right_on="weather_hour",
        validate="many_to_one",
    )
    return result.drop(columns=["_weather_hour", "weather_hour"])


def forecast_weather_at(started_at: str, latitude: float, longitude: float) -> dict[str, float]:
    """Fetch the closest available forecast hour for one prediction request."""
    target = pd.Timestamp(started_at)
    if target.tzinfo is not None:
        target = target.tz_convert(NYC_TIMEZONE).tz_localize(None)
    date = target.date().isoformat()
    params = _weather_params(date, date)
    params.update({"latitude": latitude, "longitude": longitude})
    with requests.get(FORECAST_URL, params=params, timeout=(5, 12)) as response:
        response.raise_for_status()
        weather = parse_hourly_weather(response.json())
    if weather.empty:
        raise ValueError("No forecast hours returned")
    distance = (weather["weather_hour"] - target.floor("h")).abs()
    row = weather.loc[distance.idxmin()]
    if distance.min() > pd.Timedelta(hours=1):
        raise ValueError("Requested time is outside the available forecast")
    return {name: float(row[name]) for name in WEATHER_FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START_MONTH, help="first month (YYYY-MM)")
    parser.add_argument("--end", default=DEFAULT_END_MONTH, help="last month (YYYY-MM)")
    parser.add_argument("--output", type=Path, default=WEATHER_PATH)
    args = parser.parse_args()
    download_hourly_weather(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
