"""Central paths and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
MIN_DURATION_MINUTES = 1.0
MAX_DURATION_MINUTES = 180.0

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
    "start_lat",
    "start_lng",
    "start_hour_sin",
    "start_hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]
TARGET = "duration_minutes"

