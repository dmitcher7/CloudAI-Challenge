"""Stream raw Citi Bike ZIPs into one validated Parquet dataset."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    MAX_DURATION_MINUTES,
    MIN_DURATION_MINUTES,
    PROCESSED_DIR,
    RAW_DIR,
)

COLUMN_ALIASES = {
    "starttime": "started_at",
    "stoptime": "ended_at",
    "start_station_name": "start_station_name",
    "start_station_id": "start_station_id",
    "start_station_latitude": "start_lat",
    "start_station_longitude": "start_lng",
    "end_station_name": "end_station_name",
    "end_station_id": "end_station_id",
    "end_station_latitude": "end_lat",
    "end_station_longitude": "end_lng",
    "usertype": "member_casual",
}

OUTPUT_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "duration_minutes",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
    "source_month",
]


def canonical_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return COLUMN_ALIASES.get(name, name)


def normalize_chunk(frame: pd.DataFrame, source_month: str) -> tuple[pd.DataFrame, dict]:
    """Normalize schema and filter technically invalid rows; return audit counts."""
    frame = frame.rename(columns={column: canonical_name(column) for column in frame.columns})
    before = len(frame)

    required = {"started_at", "ended_at", "start_lat", "start_lng", "member_casual"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Unsupported source schema; missing: {sorted(missing)}")

    for name in ["started_at", "ended_at"]:
        frame[name] = pd.to_datetime(frame[name], errors="coerce")
    for name in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    computed_duration = (frame["ended_at"] - frame["started_at"]).dt.total_seconds() / 60
    frame["duration_minutes"] = computed_duration
    if "ride_id" not in frame:
        # Stable within a source file; legacy rows do not have official ride IDs.
        frame["ride_id"] = source_month + "-legacy-" + frame.index.astype(str)
    if "rideable_type" not in frame:
        frame["rideable_type"] = "unknown"

    frame["member_casual"] = (
        frame["member_casual"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"subscriber": "member", "customer": "casual"})
    )
    frame["rideable_type"] = frame["rideable_type"].astype("string").str.strip().str.lower()

    for name in ["start_station_id", "end_station_id", "start_station_name", "end_station_name"]:
        if name not in frame:
            frame[name] = pd.NA
        frame[name] = frame[name].astype("string").str.strip()

    valid_time = frame["duration_minutes"].between(
        MIN_DURATION_MINUTES, MAX_DURATION_MINUTES, inclusive="both"
    )
    valid_geo = frame["start_lat"].between(40.4, 41.1) & frame["start_lng"].between(-74.4, -73.5)
    valid_user = frame["member_casual"].isin(["member", "casual"])
    valid_start = frame["started_at"].notna() & frame["start_station_id"].notna()
    kept = valid_time & valid_geo & valid_user & valid_start

    frame = frame.loc[kept].copy()
    frame["source_month"] = source_month
    frame = frame.drop_duplicates(subset=["ride_id"], keep="first")

    audit = {
        "source_month": source_month,
        "input_rows": int(before),
        "output_rows": int(len(frame)),
        "removed_invalid_duration": int((~valid_time).sum()),
        "removed_invalid_geo": int((~valid_geo).sum()),
        "removed_invalid_user_type": int((~valid_user).sum()),
        "removed_missing_start": int((~valid_start).sum()),
    }
    return frame[OUTPUT_COLUMNS], audit


def csv_members(zip_path: Path):
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.lower().endswith(".csv") and not member.startswith("__MACOSX"):
                yield archive, member


def prepare_archives(
    raw_dir: Path = RAW_DIR,
    output_path: Path = PROCESSED_DIR / "trips.parquet",
    chunksize: int = 250_000,
) -> dict:
    archives = sorted(raw_dir.glob("*-citibike-tripdata.zip"))
    if not archives:
        raise FileNotFoundError(
            f"No archives in {raw_dir}. Run: python -m src.data.download --start 2024-01 --end 2024-03"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.part")
    writer: pq.ParquetWriter | None = None
    audits: list[dict] = []

    try:
        for archive_path in archives:
            source_month = archive_path.name[:6]
            for archive, member in csv_members(archive_path):
                print(f"Processing {archive_path.name}:{member}", flush=True)
                with archive.open(member) as csv_handle:
                    for chunk in pd.read_csv(csv_handle, chunksize=chunksize, low_memory=False):
                        cleaned, audit = normalize_chunk(chunk, source_month)
                        audits.append(audit)
                        if cleaned.empty:
                            continue
                        table = pa.Table.from_pandas(cleaned, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                        writer.write_table(table)
        if writer is None:
            raise ValueError("No valid rows remained after cleaning")
    finally:
        if writer is not None:
            writer.close()

    temporary.replace(output_path)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives": len(archives),
        "input_rows": sum(item["input_rows"] for item in audits),
        "output_rows": sum(item["output_rows"] for item in audits),
        "filter_counts_are_nonexclusive": True,
        "rules": {
            "duration_minutes": [MIN_DURATION_MINUTES, MAX_DURATION_MINUTES],
            "latitude": [40.4, 41.1],
            "longitude": [-74.4, -73.5],
            "valid_user_types": ["member", "casual"],
        },
        "chunks": audits,
    }
    audit_path = output_path.with_name("cleaning_audit.json")
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary['output_rows']:,} rows to {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "trips.parquet")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    prepare_archives(args.raw_dir, args.output, args.chunksize)


if __name__ == "__main__":
    main()

