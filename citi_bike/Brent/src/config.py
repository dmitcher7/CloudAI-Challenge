"""Central paths and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
WEATHER_PATH = INTERIM_DIR / "nyc_weather_hourly.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
MIN_DURATION_MINUTES = 1.0
MAX_DURATION_MINUTES = 180.0

# Use one complete calendar year by default so seasonality is represented.
DEFAULT_START_MONTH = "2024-01"
DEFAULT_END_MONTH = "2024-12"

RAW_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

MODEL_FEATURES = [
    "rideable_type",
    "member_casual",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "direct_distance_km",
    "grid_distance_km",
    "delta_lat",
    "delta_lng",
    "bearing_sin",
    "bearing_cos",
    "route_typical_minutes",
    "route_history_log_count",
    "start_typical_minutes",
    "end_typical_minutes",
    "start_hour_sin",
    "start_hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]
TARGET = "duration_minutes"
