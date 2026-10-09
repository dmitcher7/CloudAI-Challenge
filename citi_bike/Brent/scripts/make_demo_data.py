"""Create deterministic, realistic-enough data for tests and CI only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(path: Path, rows: int = 2_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    started = pd.date_range("2024-01-01", "2024-04-01", periods=rows)
    member = rng.choice(["member", "casual"], rows, p=[0.7, 0.3])
    bike = rng.choice(["classic_bike", "electric_bike"], rows, p=[0.65, 0.35])
    station = rng.choice(["HB101", "JC001", "NY300", "NY301", "BK100"], rows)
    hour = started.hour.to_numpy()
    weekend = (started.dayofweek.to_numpy() >= 5).astype(int)
    duration = (
        7
        + 5 * (member == "casual")
        + 2 * (bike == "classic_bike")
        + 3 * weekend
        + 2 * np.sin(2 * np.pi * hour / 24)
        + rng.gamma(2, 2, rows)
    )
    frame = pd.DataFrame(
        {
            "ride_id": [f"demo-{i}" for i in range(rows)],
            "rideable_type": bike,
            "started_at": started,
            "ended_at": started + pd.to_timedelta(duration, unit="minute"),
            "duration_minutes": duration,
            "start_station_name": "Demo station",
            "start_station_id": station,
            "end_station_name": "Demo destination",
            "end_station_id": "END",
            "start_lat": 40.72 + rng.normal(0, 0.02, rows),
            "start_lng": -74.00 + rng.normal(0, 0.02, rows),
            "end_lat": 40.73,
            "end_lng": -73.99,
            "member_casual": member,
            "source_month": started.strftime("%Y%m"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"Wrote {len(frame):,} synthetic CI rows to {path} (not research evidence).")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2_000)
    args = parser.parse_args()
    make_demo_data(args.output, args.rows)


if __name__ == "__main__":
    main()

