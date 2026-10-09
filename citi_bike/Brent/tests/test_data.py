from pathlib import Path

import pandas as pd

from src.data.download import month_range, parse_month
from src.data.prepare import normalize_chunk


def test_month_range_crosses_year_boundary():
    assert list(month_range(parse_month("2023-11"), parse_month("2024-02"))) == [
        "202311",
        "202312",
        "202401",
        "202402",
    ]


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

