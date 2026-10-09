"""Download official Citi Bike monthly archives reproducibly.

Examples
--------
python -m src.data.download
python -m src.data.download --start 2024-01 --end 2024-12 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests

from src.config import DEFAULT_END_MONTH, DEFAULT_START_MONTH, RAW_DIR

BASE_URL = "https://s3.amazonaws.com/tripdata/{yyyymm}-citibike-tripdata.zip"


def parse_month(value: str) -> tuple[int, int]:
    """Parse YYYY-MM and fail early with a useful CLI error."""
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use YYYY-MM, e.g. 2024-01") from exc
    return parsed.year, parsed.month


def month_range(start: tuple[int, int], end: tuple[int, int]):
    """Yield inclusive calendar months as YYYYMM."""
    year, month = start
    while (year, month) <= end:
        yield f"{year:04d}{month:02d}"
        month += 1
        if month == 13:
            month = 1
            year += 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip(path: Path) -> None:
    """Raise when an archive is corrupt or has no tabular file."""
    try:
        with ZipFile(path) as archive:
            broken_member = archive.testzip()
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if broken_member:
                raise ValueError(f"corrupt ZIP member: {broken_member}")
            if not members:
                raise ValueError("archive contains no CSV file")
    except BadZipFile as exc:
        raise ValueError(f"not a valid ZIP: {path}") from exc


def download_month(yyyymm: str, output_dir: Path, force: bool = False) -> dict:
    """Download one month atomically and return its provenance record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(yyyymm=yyyymm)
    destination = output_dir / f"{yyyymm}-citibike-tripdata.zip"

    if destination.exists() and not force:
        validate_zip(destination)
        status = "reused"
    else:
        temporary = destination.with_suffix(".zip.part")
        try:
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            validate_zip(temporary)
            temporary.replace(destination)
            status = "downloaded"
        finally:
            if temporary.exists():
                temporary.unlink()

    return {
        "month": yyyymm,
        "url": url,
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_range(start: str, end: str, output_dir: Path = RAW_DIR, force: bool = False):
    start_month = parse_month(start)
    end_month = parse_month(end)
    if start_month > end_month:
        raise ValueError("start month must not be later than end month")

    records = []
    for yyyymm in month_range(start_month, end_month):
        print(f"[{yyyymm}] downloading/checking …", flush=True)
        records.append(download_month(yyyymm, output_dir, force=force))

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {manifest} with {len(records)} verified archive(s).")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", default=DEFAULT_START_MONTH, help="first month (YYYY-MM)"
    )
    parser.add_argument(
        "--end", default=DEFAULT_END_MONTH, help="last month, inclusive (YYYY-MM)"
    )
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="redownload verified files")
    args = parser.parse_args()
    download_range(args.start, args.end, args.output_dir, args.force)


if __name__ == "__main__":
    main()
