from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.config import DEFAULT_END_MONTH, DEFAULT_START_MONTH
from src.data.download import month_range, parse_inventory_xml, parse_month, select_archives
from src.data.prepare import csv_members, normalize_chunk, source_period


def test_month_range_crosses_year_boundary():
    assert list(month_range(parse_month("2023-11"), parse_month("2024-02"))) == [
        "202311",
        "202312",
        "202401",
        "202402",
    ]


def test_default_period_covers_complete_public_history():
    assert DEFAULT_START_MONTH == "2013-06"
    assert DEFAULT_END_MONTH == "latest"


def test_inventory_supports_annual_and_monthly_archives():
    payload = """<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <Contents><Key>2013-citibike-tripdata.zip</Key><Size>10</Size></Contents>
      <Contents><Key>202501-citibike-tripdata.zip</Key><Size>20</Size></Contents>
      <Contents><Key>JC-202501-citibike-tripdata.zip</Key><Size>30</Size></Contents>
    </ListBucketResult>"""
    archives, token = parse_inventory_xml(payload)
    assert token is None
    assert [item["key"] for item in archives] == [
        "2013-citibike-tripdata.zip",
        "202501-citibike-tripdata.zip",
    ]
    assert archives[0]["first_month"] == "201306"
    assert select_archives(archives, "2013-06", "latest") == archives


def test_source_period_handles_year_bundle_member_names():
    archive = Path("2019-citibike-tripdata.zip")
    assert source_period(archive, "2019/201907-citibike-tripdata.csv") == "201907"


def test_csv_members_avoids_duplicate_repackaged_month(tmp_path: Path):
    archive = tmp_path / "2013-citibike-tripdata.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("2013/201306-citibike-tripdata.csv", "a\n1\n")
        output.writestr("2013/6_June/201306-citibike-tripdata_1.csv", "a\n1\n")
        output.writestr("2013/7_July/201307-citibike-tripdata_1.csv", "a\n2\n")
        output.writestr("2013/7_July/201307-citibike-tripdata_2.csv", "a\n3\n")
    selected = [member for _, member in csv_members(archive)]
    assert selected == [
        "2013/201306-citibike-tripdata.csv",
        "2013/7_July/201307-citibike-tripdata_1.csv",
        "2013/7_July/201307-citibike-tripdata_2.csv",
    ]


def test_csv_members_prefers_complete_file_over_same_depth_parts(tmp_path: Path):
    archive = tmp_path / "2018-citibike-tripdata.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("2018/201804-citibike-tripdata_1.csv", "a\n1\n")
        output.writestr("2018/201804-citibike-tripdata_2.csv", "a\n2\n")
        output.writestr("2018/201804-citibike-tripdata.csv", "a\n1\n2\n")

    selected = [member for _, member in csv_members(archive)]

    assert selected == ["2018/201804-citibike-tripdata.csv"]


def test_csv_members_reads_nested_month_zip(tmp_path: Path):
    nested = BytesIO()
    with ZipFile(nested, "w") as month:
        month.writestr("202001-citibike-tripdata.csv", "a\n1\n")
    archive = tmp_path / "2020-citibike-tripdata.zip"
    with ZipFile(archive, "w") as year:
        year.writestr("202001-citibike-tripdata.zip", nested.getvalue())
    generator = csv_members(archive)
    inner_archive, member = next(generator)
    with inner_archive.open(member) as handle:
        assert handle.read() == b"a\n1\n"
    generator.close()


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


def test_normalize_chunk_maps_spaced_legacy_schema():
    frame = pd.DataFrame(
        {
            "Start Time": ["2016-12-01 10:00:00"],
            "Stop Time": ["2016-12-01 10:12:00"],
            "Start Station ID": ["A"],
            "Start Station Latitude": [40.72],
            "Start Station Longitude": [-73.99],
            "User Type": ["Subscriber"],
        }
    )

    clean, _ = normalize_chunk(frame, "201612")

    assert len(clean) == 1
    assert clean.iloc[0]["member_casual"] == "member"
    assert clean.iloc[0]["duration_minutes"] == 12
