"""Download all official Citi Bike NYC trip-history archives reproducibly.

Examples
--------
python -m src.data.download
python -m src.data.download --start 2013-06 --end latest --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import requests

from src.config import DEFAULT_END_MONTH, DEFAULT_START_MONTH, RAW_DIR

BUCKET_URL = "https://s3.amazonaws.com/tripdata"
ARCHIVE_PATTERN = re.compile(r"^(?P<period>\d{4}|\d{6})-citibike-tripdata\.zip$")


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
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".csv", ".zip"))
                and not name.startswith("__MACOSX")
            ]
            if broken_member:
                raise ValueError(f"corrupt ZIP member: {broken_member}")
            if not members:
                raise ValueError("archive contains no CSV or nested ZIP file")
    except BadZipFile as exc:
        raise ValueError(f"not a valid ZIP: {path}") from exc


def parse_inventory_xml(payload: bytes | str) -> tuple[list[dict], str | None]:
    """Parse one public S3 inventory page without depending on an AWS SDK."""
    root = ElementTree.fromstring(payload)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    archives = []
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", namespaces=namespace) or ""
        match = ARCHIVE_PATTERN.fullmatch(key)
        if not match:
            continue
        period = match.group("period")
        if len(period) == 4:
            first_month = f"{period}01"
            last_month = f"{period}12"
            if period == "2013":
                first_month = "201306"
        else:
            first_month = last_month = period
        archives.append(
            {
                "key": key,
                "period": period,
                "first_month": first_month,
                "last_month": last_month,
                "bytes": int(item.findtext("s3:Size", default="0", namespaces=namespace)),
                "last_modified": item.findtext(
                    "s3:LastModified", default="", namespaces=namespace
                ),
            }
        )
    truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace)
    token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
    return archives, token if truncated.lower() == "true" else None


def list_official_archives() -> list[dict]:
    """Return the current official NYC archive inventory (Jersey City excluded)."""
    archives: list[dict] = []
    token: str | None = None
    while True:
        params = {"list-type": "2"}
        if token:
            params["continuation-token"] = token
        with requests.get(BUCKET_URL, params=params, timeout=(15, 180)) as response:
            response.raise_for_status()
            page, token = parse_inventory_xml(response.content)
        archives.extend(page)
        if not token:
            break
    if not archives:
        raise ValueError("The official Citi Bike inventory contained no NYC archives")
    return sorted(archives, key=lambda item: item["first_month"])


def _compact_month(value: str) -> str:
    year, month = parse_month(value)
    return f"{year:04d}{month:02d}"


def select_archives(archives: list[dict], start: str, end: str) -> list[dict]:
    """Select archives overlapping a month range; ``latest`` uses the newest object."""
    start_month = _compact_month(start)
    end_month = (
        max(item["last_month"] for item in archives)
        if end == "latest"
        else _compact_month(end)
    )
    if start_month > end_month:
        raise ValueError("start month must not be later than end month")
    selected = [
        item
        for item in archives
        if item["last_month"] >= start_month and item["first_month"] <= end_month
    ]
    if not selected:
        raise ValueError(f"No official archives overlap {start} through {end}")
    return selected


def download_archive(archive: dict, output_dir: Path, force: bool = False) -> dict:
    """Download one official archive atomically and return its provenance record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET_URL}/{archive['key']}"
    destination = output_dir / archive["key"]

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
        **archive,
        "url": url,
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_month(yyyymm: str, output_dir: Path, force: bool = False) -> dict:
    """Backward-compatible helper for a known monthly archive."""
    archive = {
        "key": f"{yyyymm}-citibike-tripdata.zip",
        "period": yyyymm,
        "first_month": yyyymm,
        "last_month": yyyymm,
    }
    return download_archive(archive, output_dir, force)


def download_range(
    start: str = DEFAULT_START_MONTH,
    end: str = DEFAULT_END_MONTH,
    output_dir: Path = RAW_DIR,
    force: bool = False,
):
    inventory = list_official_archives()
    selected = select_archives(inventory, start, end)
    records = []
    for number, archive in enumerate(selected, start=1):
        print(
            f"[{number}/{len(selected)}] downloading/checking {archive['key']} ...",
            flush=True,
        )
        records.append(download_archive(archive, output_dir, force=force))

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {manifest} with {len(records)} verified archive(s).")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START_MONTH, help="first month (YYYY-MM)")
    parser.add_argument(
        "--end", default=DEFAULT_END_MONTH, help="last month (YYYY-MM), or latest"
    )
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="redownload verified files")
    args = parser.parse_args()
    download_range(args.start, args.end, args.output_dir, args.force)


if __name__ == "__main__":
    main()
