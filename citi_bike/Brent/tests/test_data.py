from pathlib import Path

import pandas as pd

from src.config import DEFAULT_END_MONTH, DEFAULT_START_MONTH
from src.data.download import month_range, parse_month
from src.data.prepare import normalize_chunk


def test_month_range_crosses_year_boundary():
    assert list(month_range(parse_month("2023-11"), parse_month("2024-02"))) == [
        "202311",
        "202312",
        "202401",
        "202402",
    ]


def test_default_period_covers_complete_calendar_year():
    months = list(month_range(parse_month(DEFAULT_START_MONTH), parse_month(DEFAULT_END_MONTH)))
    assert months == [f"2024{month:02d}" for month in range(1, 13)]


def test_normalize_chunk_maps_legacy_schema_and_removes_invalid_rows():
    frame = pd.DataFrame(
        {
            "starttime": ["2024-01-01 10:00", "2024-01-01 10:00"],
            "stoptime": ["2024-01-01 10:12", "2024-01-01 15:00"],
            "start station id": ["A", "B"],
            "start station latitude": [40.72, 40.72],
            "start station longitude": [-73.99, -73.99],
            "usertype": ["Subscriber", "Customer"],
        }
    )
    clean, audit = normalize_chunk(frame, "202401")
    assert len(clean) == 1
    assert clean.iloc[0]["member_casual"] == "member"
    assert clean.iloc[0]["duration_minutes"] == 12
    assert audit["removed_invalid_duration"] == 1
