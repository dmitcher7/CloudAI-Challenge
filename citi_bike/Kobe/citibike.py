"""Gedeelde code voor de Citi Bike-notebooks van Kobe: vraag (vertrekken) per zone per uur.

De notebooks (01 t/m 06) én train.py (dat het model voor de API schrijft) gebruiken deze functies,
zodat de data overal op exact dezelfde manier verwerkt wordt.

Er zijn twee studies:
    2024 (data/2024, models/2024)   modelselectie op één jaar: welke techniek werkt het best? (02, 03, 05b-05d)
    alle data (data, models)        juni 2013 - nu, met de beste techniek (01, 02b, 04, 05a, 05e, 06)

Overzicht:
    download_all()            alle NYC-bestanden van de officiële S3-bucket (+ manifest met SHA-256)
    aggregate_all()           elk bestand in stukken inlezen -> vertrekken per station per uur
    download_weather_range()  uurlijks weer in NYC via de Open-Meteo archive API
    fit_zones()               stations groeperen tot zones (KMeans, gewogen met het aantal ritten)
    prepare_full_table()      rooster zone x uur over alle jaren, met actieve stations en weer
    make_features()           modelkenmerken; dezelfde code staat in deploy/citi_bike_demand/api.py
    backtest_folds()          tijdsgebonden validatie (train op het verleden, valideer op het jaar erna)
    evaluate()                MAE, RMSE, R² en Poisson-deviance
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = HERE / "models"
METRICS_DIR = MODEL_DIR / "metrics"
PRED_DIR = MODEL_DIR / "predictions"
# De studie op 2024 (modelselectie) heeft haar eigen mappen.
DATA_2024 = DATA_DIR / "2024"
MODEL_2024 = MODEL_DIR / "2024"
METRICS_2024 = MODEL_2024 / "metrics"
PRED_2024 = MODEL_2024 / "predictions"

BUCKET_URL = "https://s3.amazonaws.com/tripdata/"
YEAR = 2024                                       # jaar van de modelselectie-studie
NYC_CENTER = (40.73, -73.97)                      # voor de weer-API (één punt voor heel NYC)
NYC_BBOX = {"lat": (40.4, 41.1), "lng": (-74.4, -73.5)}
SEED = 42

# Stil houden van een joblib-waarschuwing op Windows ("could not find the number of physical cores"):
# met een opgegeven maximum (alle kernen op één na) zoekt joblib het aantal fysieke kernen niet meer op.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

# Kolommen die we uit de ruwe CSV's lezen (alles als tekst: types zetten we zelf, zodat
# "gemengde types" in station-id's zoals 7407.13 / "HB101" geen problemen geven).
RAW_COLUMNS = ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_name",
               "start_station_id", "end_station_name", "end_station_id", "start_lat", "start_lng",
               "end_lat", "end_lng", "member_casual"]

# Modelkenmerken: alles is gekend op het moment dat je een voorspelling vraagt (geen leakage).
FEATURES_2024 = ["zone", "hour", "weekday", "month", "is_weekend", "is_holiday",
                 "temperature_c", "precipitation_mm", "wind_kmh"]
# Over 13 jaar groeide het netwerk van ± 330 naar ± 2.200 stations: het model moet weten hoe groot een zone
# op dat moment is (actieve stations) en in welk jaar we zitten (trend: e-bikes, groei, corona).
# Sneeuwhoogte kwam erbij na de foutenanalyse in 09: de grootste fouten zaten op droge dagen ná een sneeuwstorm.
FEATURES = FEATURES_2024 + ["year", "active_stations", "snow_depth_cm"]
CATEGORICAL = ["zone"]
TARGET = "trips"


# ----------------------------------------------------------------------------------------------
# 1. Downloaden
# ----------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_bucket() -> pd.DataFrame:
    """Alle bestanden in de S3-bucket (naam, grootte, laatst gewijzigd). De listing komt per 1.000 terug."""
    import xml.etree.ElementTree as ET

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows, token = [], None
    while True:
        params = {"list-type": "2", **({"continuation-token": token} if token else {})}
        root = ET.fromstring(requests.get(BUCKET_URL, params=params, timeout=60).content)
        for item in root.findall("s3:Contents", ns):
            rows.append({"file": item.find("s3:Key", ns).text, "bytes": int(item.find("s3:Size", ns).text),
                         "last_modified": item.find("s3:LastModified", ns).text})
        token = root.find("s3:NextContinuationToken", ns)
        if token is None:
            return pd.DataFrame(rows)
        token = token.text


def nyc_files(bucket: pd.DataFrame) -> pd.DataFrame:
    """De NYC-bestanden: jaar-zip's (2013-2023, `2013-citibike-tripdata.zip`) en maand-zip's (vanaf 2024).

    De `JC-...`-bestanden (Jersey City, een apart systeem) laten we weg, net zoals de JC-stations in de cleaning.
    """
    nyc = bucket[bucket["file"].str.fullmatch(r"\d{4}(\d{2})?-citibike-tripdata(\.csv)?\.zip")].copy()
    nyc["period"] = nyc["file"].str.extract(r"^(\d{4}(?:\d{2})?)")[0]
    return nyc.sort_values("period").reset_index(drop=True)


def download_file(name: str, size: int, raw_dir: Path = RAW_DIR) -> dict:
    """Download één bestand (als het nog niet volledig aanwezig is) en geef een manifest-regel terug."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = BUCKET_URL + name
    target = raw_dir / name
    if target.exists() and target.stat().st_size == size:
        status = "al aanwezig"
    else:
        partial = target.with_suffix(".part")
        if not (partial.exists() and partial.stat().st_size == size):
            with requests.get(url, stream=True, timeout=(15, 300)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                        handle.write(block)
        if partial.stat().st_size != size:
            raise IOError(f"{name}: download onvolledig ({partial.stat().st_size} van {size} bytes)")
        for attempt in range(10):          # Windows: de virusscanner houdt een vers bestand soms even vast
            try:
                partial.replace(target)
                break
            except PermissionError:
                time.sleep(3)
        else:
            partial.replace(target)
        status = "gedownload"
    return {"file": name, "url": url, "bytes": size, "csv_files": [n for n, _ in list_csvs(target)],
            "sha256": _sha256(target), "status": status,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def download_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download alle NYC-bestanden uit de bucket en schrijf data/raw/manifest.json (bron, grootte, checksum)."""
    files = nyc_files(list_bucket())
    records = []
    for row in files.itertuples():
        started = datetime.now()
        record = download_file(row.file, row.bytes, raw_dir)
        print(f"{row.file:<36} {record['status']:<11} {row.bytes / 1e9:5.2f} GB, "
              f"{len(record['csv_files']):>3} CSV's ({(datetime.now() - started).seconds} s)", flush=True)
        records.append(record)
        (raw_dir / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------------------
# 2. Inlezen en aggregeren (per maand, in stukken)
# ----------------------------------------------------------------------------------------------
def _walk_zip(archive: ZipFile, prefix: str = ""):
    """(volledige naam, archief, naam in archief) voor elke CSV, ook in geneste zip's.

    De jaarbestanden 2013-2023 bevatten mappen en/of zip's per maand; die openen we in het geheugen.
    """
    for name in sorted(archive.namelist()):
        base = Path(name).name
        if name.endswith("/") or "__MACOSX" in name or base.startswith("._"):
            continue
        if name.lower().endswith(".zip"):
            with ZipFile(io.BytesIO(archive.read(name))) as inner:
                yield from _walk_zip(inner, f"{prefix}{name}/")
        elif name.lower().endswith(".csv"):
            yield f"{prefix}{name}", archive, name


def list_csvs(zip_path: Path) -> list[tuple[str, None]]:
    with ZipFile(zip_path) as archive:
        return [(full, None) for full, _, _ in _walk_zip(archive)]


# Kolomnamen door de jaren heen (2013-2016: "starttime", 2016-2017: "Start Time", vanaf 2021: "started_at"
# en geen ritnummer/fietstype in de oude data) -> één schema.
COLUMN_ALIASES = {"starttime": "started_at", "start_time": "started_at", "stoptime": "ended_at",
                  "stop_time": "ended_at", "start_station_latitude": "start_lat",
                  "start_station_longitude": "start_lng", "end_station_latitude": "end_lat",
                  "end_station_longitude": "end_lng", "usertype": "member_casual", "user_type": "member_casual",
                  "bikeid": "bike_id"}


def standardize_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    """Hernoem naar het schema van vandaag; ontbrekende kolommen worden leeg (of 'unknown')."""
    chunk = chunk.rename(columns=lambda c: re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"))
    chunk = chunk.rename(columns=COLUMN_ALIASES)
    if "member_casual" in chunk:
        chunk["member_casual"] = (chunk["member_casual"].str.strip().str.lower()
                                  .replace({"subscriber": "member", "customer": "casual"}))
    if "rideable_type" not in chunk:
        chunk["rideable_type"] = "unknown"          # fietstype staat pas sinds 2020 in de data
    return chunk.reindex(columns=RAW_COLUMNS)


def iter_chunks(zip_path: Path, chunksize: int = 500_000, with_name: bool = False):
    """Lees alle CSV's van een (jaar- of maand-)zip in stukken van `chunksize` rijen, in het schema van vandaag."""
    with ZipFile(zip_path) as archive:
        for full, holder, name in _walk_zip(archive):
            with holder.open(name) as handle:
                for chunk in pd.read_csv(handle, dtype=str, chunksize=chunksize):
                    chunk = standardize_columns(chunk)
                    yield (full, chunk) if with_name else chunk


DATE_FORMATS = ["ISO8601", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"]


def to_datetime(values: pd.Series) -> pd.Series:
    """Datums in elk formaat dat Citi Bike ooit gebruikte: '2013-06-01 00:00:01', '1/1/2015 0:01', ...

    We proberen de formaten na elkaar en nemen het eerste dat (bijna) alles kan lezen; zo blijft het snel
    (een vast formaat) en vallen echt kapotte waarden toch op als NaT.
    """
    best = None
    filled = values.notna().sum()
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed.notna().sum() >= 0.99 * filled:
            return parsed
        if best is None or parsed.notna().sum() > best.notna().sum():
            best = parsed
    return best


def parse_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Tekst -> juiste types, plus de ritduur in minuten."""
    out = chunk.copy()
    for column in ["started_at", "ended_at"]:
        out[column] = to_datetime(out[column])
    for column in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["duration_min"] = (out["ended_at"] - out["started_at"]).dt.total_seconds() / 60
    return out


LEGACY_ID_CUTOFF = pd.Timestamp("2021-02-01")   # tot dan gehele station-id's (bv. 72), daarna 7407.13


def normalize_station_id(ids: pd.Series, legacy: pd.Series | None = None) -> pd.Series:
    """Station-id's die als getal werden opgeslagen, terugzetten naar hun echte vorm.

    - nieuw systeem (vanaf februari 2021): '3423.1' -> '3423.10' (de laatste 0 viel weg; echte id's hebben
      altijd twee decimalen);
    - oud systeem (`legacy`, ritten vóór februari 2021): '119.0' -> '119' (gehele id's die in 2018-2019 als
      kommagetal in de CSV staan).
    Tekst-id's (HB101, JC023) blijven ongemoeid.
    """
    legacy = pd.Series(False, index=ids.index) if legacy is None else legacy.fillna(False)
    one_decimal = ids.str.fullmatch(r"\d+\.\d", na=False) & ~legacy
    float_int = ids.str.fullmatch(r"\d+\.0+", na=False) & legacy
    return ids.mask(one_decimal, ids + "0").mask(float_int, ids.str.replace(r"\.0+$", "", regex=True))


def departure_mask(trips: pd.DataFrame) -> pd.Series:
    """Welke rijen tellen als vertrek? Starttijd + startstation + coördinaten binnen NYC gekend.

    De ritduur speelt hier bewust geen rol: ook een rit die (te) lang duurt, is een fiets die vertrok.
    Zie 02_eda_cleaning voor de onderbouwing.
    """
    lat_ok = trips["start_lat"].between(*NYC_BBOX["lat"])
    lng_ok = trips["start_lng"].between(*NYC_BBOX["lng"])
    return trips["started_at"].notna() & trips["start_station_id"].notna() & lat_ok & lng_ok


def csv_month(name: str) -> str | None:
    match = re.search(r"(20\d{2})(0[1-9]|1[0-2])", Path(name).name)
    return match.group(0) if match else None


def select_csvs(names: list[str]) -> tuple[list[str], list[str]]:
    """Per maand één versie van de data: (gebruikt, weggelaten).

    Sommige jaarbestanden bevatten een maand twee keer, bv. 2013: `201309-citibike-tripdata.csv` én
    `9_September/201309-citibike-tripdata_1.csv` + `_2.csv`. We houden per maand de versie in de diepste
    map (de opgesplitste delen, zoals in alle andere jaren); 02b toont dat beide versies dezelfde ritten bevatten.
    """
    by_month: dict[str, dict[str, list[str]]] = {}
    for name in names:
        parent = str(Path(name).parent)
        by_month.setdefault(csv_month(name) or "onbekend", {}).setdefault(parent, []).append(name)
    keep, dropped = [], []
    for groups in by_month.values():
        best = max(groups, key=lambda parent: (parent.count("/") + parent.count("\\"), len(groups[parent])))
        for parent, files in groups.items():
            (keep if parent == best else dropped).extend(files)
    return sorted(keep), sorted(dropped)


def _new_audit(file: str, month: str) -> dict:
    return {"file": file, "month": month, "csv_files": [], "rows": 0, "missing": {c: 0 for c in RAW_COLUMNS},
            "no_start_time": 0, "no_start_station": 0, "outside_nyc": 0, "departures": 0,
            "duration_negative": 0, "duration_over_24h": 0, "station_id_fixed": 0, "_hashes": []}


def aggregate_file(zip_path: Path, sample_frac: float = 0.002, seed: int = SEED):
    """Eén passage door een bronbestand (jaar of maand): vertrekken per station per uur, stationslijst,
    audit per maand en een ruwe steekproef. Geeft (station_hour, stations, audits, sample) terug."""
    rng = np.random.default_rng(seed)
    keep, dropped = select_csvs([name for name, _ in list_csvs(zip_path)])
    keep = set(keep)
    station_hour, stations, samples = [], [], []
    audits: dict[str, dict] = {}

    with ZipFile(zip_path) as archive:
        for full, holder, name in _walk_zip(archive):
            if full not in keep:
                continue
            month = csv_month(full) or "onbekend"
            audit = audits.setdefault(month, _new_audit(zip_path.name, month))
            audit["csv_files"].append(full)
            with holder.open(name) as handle:
                for chunk in pd.read_csv(handle, dtype=str, chunksize=500_000):
                    chunk = standardize_columns(chunk)
                    audit["rows"] += len(chunk)
                    for column in RAW_COLUMNS:
                        audit["missing"][column] += int(chunk[column].isna().sum())
                    # Duplicaten: oude data heeft geen ride_id, dus we vergelijken de hele rit.
                    key = chunk[["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"]]
                    audit["_hashes"].append(pd.util.hash_pandas_object(key, index=False).to_numpy())
                    samples.append(chunk.loc[rng.random(len(chunk)) < sample_frac].assign(source_file=zip_path.name))

                    trips = parse_chunk(chunk)
                    fixed = normalize_station_id(trips["start_station_id"], trips["started_at"] < LEGACY_ID_CUTOFF)
                    changed = trips["start_station_id"].notna() & (fixed != trips["start_station_id"])
                    audit["station_id_fixed"] += int(changed.sum())
                    trips["start_station_id"] = fixed
                    audit["no_start_time"] += int(trips["started_at"].isna().sum())
                    audit["no_start_station"] += int(trips["start_station_id"].isna().sum())
                    audit["duration_negative"] += int((trips["duration_min"] < 0).sum())
                    audit["duration_over_24h"] += int((trips["duration_min"] > 24 * 60).sum())

                    ok = departure_mask(trips)
                    known = trips["start_station_id"].notna() & trips["started_at"].notna()
                    audit["outside_nyc"] += int((known & ~ok).sum())
                    trips = trips.loc[ok]
                    audit["departures"] += len(trips)
                    trips = trips.assign(time=trips["started_at"].dt.floor("h"),
                                         member=(trips["member_casual"] == "member").astype("int32"),
                                         electric=(trips["rideable_type"] == "electric_bike").astype("int32"))
                    station_hour.append(trips.groupby(["start_station_id", "time"])
                                             .agg(trips=("time", "size"), members=("member", "sum"),
                                                  electric=("electric", "sum")).reset_index())
                    stations.append(trips.groupby("start_station_id")
                                         .agg(name=("start_station_name", "last"), lat_sum=("start_lat", "sum"),
                                              lng_sum=("start_lng", "sum"), trips=("time", "size"),
                                              first_time=("time", "min"), last_time=("time", "max")).reset_index())

    for audit in audits.values():
        hashes = np.concatenate(audit.pop("_hashes"))
        audit["duplicate_rows"] = int(len(hashes) - len(np.unique(hashes)))
    result = list(audits.values())
    if dropped:
        result.append({"file": zip_path.name, "month": "dubbele versies (weggelaten)", "csv_files": dropped})

    station_hour = (pd.concat(station_hour).groupby(["start_station_id", "time"], as_index=False)
                      [["trips", "members", "electric"]].sum())
    return station_hour, combine_stations(pd.concat(stations)), result, pd.concat(samples, ignore_index=True)


def combine_stations(parts: pd.DataFrame) -> pd.DataFrame:
    """Stationsdelen samenvoegen (som van coördinaten en ritten, eerste/laatste gebruik, laatste naam)."""
    return (parts.sort_values("last_time").groupby("start_station_id", as_index=False)
                 .agg(name=("name", "last"), lat_sum=("lat_sum", "sum"), lng_sum=("lng_sum", "sum"),
                      trips=("trips", "sum"), first_time=("first_time", "min"), last_time=("last_time", "max")))


def aggregate_all(raw_dir: Path = RAW_DIR, out_dir: Path = DATA_DIR, sample_frac: float = 0.002) -> dict:
    """Aggregeer elk bronbestand uit het manifest. Per bestand worden de resultaten apart bewaard,
    zodat het geheugen beperkt blijft en een onderbroken run kan hervatten.

    data/station_hour/<periode>.parquet  vertrekken per station per uur (enkel uren met >= 1 vertrek)
    data/stations.parquet                station-id, naam, gemiddelde coördinaten, ritten, eerste/laatste gebruik
    data/audit.json                      per maand: rijen, ontbrekende waarden, duplicaten, wat werd weggelaten
    data/trips_sample.parquet            ruwe steekproef (0,2%) van alle ritten (voor de EDA)
    """
    files = [r["file"] for r in json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))]
    parts = {name: out_dir / name for name in ["station_hour", "_stations", "_samples", "_audits"]}
    for directory in parts.values():
        directory.mkdir(parents=True, exist_ok=True)

    for file in files:
        period = file.split("-")[0]
        if (parts["_audits"] / f"{period}.json").exists():
            print(f"{file:<36} al verwerkt", flush=True)
            continue
        started = datetime.now()
        sh, st, audits, sample = aggregate_file(raw_dir / file, sample_frac)
        sh.to_parquet(parts["station_hour"] / f"{period}.parquet", index=False)
        st.to_parquet(parts["_stations"] / f"{period}.parquet", index=False)
        sample.to_parquet(parts["_samples"] / f"{period}.parquet", index=False)
        (parts["_audits"] / f"{period}.json").write_text(json.dumps(audits, indent=2, default=str), encoding="utf-8")
        rows = sum(a.get("rows", 0) for a in audits)
        deps = sum(a.get("departures", 0) for a in audits)
        print(f"{file:<36} {rows:>12,} rijen -> {deps:>12,} vertrekken ({(datetime.now() - started).seconds} s)",
              flush=True)

    stations = combine_stations(pd.concat(pd.read_parquet(p) for p in sorted(parts["_stations"].glob("*.parquet"))))
    stations["lat"] = stations.pop("lat_sum") / stations["trips"]
    stations["lng"] = stations.pop("lng_sum") / stations["trips"]
    stations = stations[["start_station_id", "name", "lat", "lng", "trips", "first_time", "last_time"]]
    stations.to_parquet(out_dir / "stations.parquet", index=False)
    pd.concat(pd.read_parquet(p) for p in sorted(parts["_samples"].glob("*.parquet"))).to_parquet(
        out_dir / "trips_sample.parquet", index=False)
    audits = [a for p in sorted(parts["_audits"].glob("*.json")) for a in json.loads(p.read_text(encoding="utf-8"))]
    (out_dir / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    return {"stations": stations, "audit": audits}


def load_station_hour(directory: Path = DATA_DIR / "station_hour", columns=None) -> pd.DataFrame:
    """Alle station-uren samen (tientallen miljoenen rijen; geef `columns` mee om geheugen te sparen)."""
    return pd.concat((pd.read_parquet(p, columns=columns) for p in sorted(directory.glob("*.parquet"))),
                     ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 3. Weer (Open-Meteo, gratis, geen API-sleutel)
# ----------------------------------------------------------------------------------------------
WEATHER_VARIABLES = {"temperature_2m": "temperature_c", "precipitation": "precipitation_mm",
                     "wind_speed_10m": "wind_kmh", "snowfall": "snowfall_cm",
                     "relative_humidity_2m": "humidity_pct", "snow_depth": "snow_depth_cm"}


def _get_with_retries(url: str, params: dict, attempts: int = 6) -> dict:
    """GET met herhaalpogingen: de gratis API weigert soms tijdelijk na veel aanvragen."""
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout):
            if attempt == attempts - 1:
                raise
            time.sleep(20 * (attempt + 1))


def download_weather(start: str = f"{YEAR}-01-01", end: str = f"{YEAR}-12-31",
                     path: Path = DATA_2024 / f"weather_{YEAR}.parquet") -> pd.DataFrame:
    """Uurlijks weer voor NYC in lokale tijd (zoals de ritdata). Wordt gecachet in data/."""
    if path.exists():
        return pd.read_parquet(path)
    hourly = _get_with_retries("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": NYC_CENTER[0], "longitude": NYC_CENTER[1], "start_date": start, "end_date": end,
        "hourly": ",".join(WEATHER_VARIABLES), "timezone": "America/New_York", "wind_speed_unit": "kmh"})["hourly"]
    weather = pd.DataFrame(hourly).rename(columns={"time": "time", **WEATHER_VARIABLES})
    weather["snow_depth_cm"] = weather["snow_depth_cm"] * 100          # de API geeft meter
    weather["time"] = pd.to_datetime(weather["time"])
    # Zomertijd: het "verdwenen" uur in maart ontbreekt, het dubbele uur in november komt twee keer voor.
    weather = weather.drop_duplicates("time").set_index("time")
    full = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h")
    weather = weather.reindex(full).interpolate(limit=2).rename_axis("time").reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path, index=False)
    return weather


def download_weather_range(start: str, end: str, path: Path = DATA_DIR / "weather.parquet") -> pd.DataFrame:
    """Weer over meerdere jaren: per jaar opgevraagd (kleinere antwoorden), samengevoegd en gecachet."""
    if path.exists():
        return pd.read_parquet(path)
    years = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        first = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        last = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
        years.append(download_weather(first, last, path=path.parent / f"_weather_{year}.parquet"))
        time.sleep(1)                               # vriendelijk voor de gratis API
    weather = pd.concat(years, ignore_index=True).drop_duplicates("time")
    weather.to_parquet(path, index=False)
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        (path.parent / f"_weather_{year}.parquet").unlink(missing_ok=True)
    return weather


# ----------------------------------------------------------------------------------------------
# 4. Zones (unsupervised: KMeans op de stationslocaties)
# ----------------------------------------------------------------------------------------------
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LNG = 111.0 * np.cos(np.radians(40.73))   # op de breedtegraad van NYC


def to_km(lat, lng) -> np.ndarray:
    """Graden -> (ongeveer) kilometers, zodat afstanden in beide richtingen even zwaar wegen."""
    return np.column_stack([np.asarray(lat) * KM_PER_DEG_LAT, np.asarray(lng) * KM_PER_DEG_LNG])


def fit_zones(stations: pd.DataFrame, k: int, seed: int = SEED):
    """KMeans op stationscoördinaten, gewogen met het aantal ritten: drukke buurten krijgen kleinere zones."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    kmeans.fit(to_km(stations["lat"], stations["lng"]), sample_weight=stations["trips"])
    return kmeans


def zone_table(stations: pd.DataFrame, kmeans) -> pd.DataFrame:
    """Per zone: id, naam (drukste station), centrum, aantal stations en ritten.

    Zones worden hernummerd van druk (0) naar rustig, zodat de nummers stabiel en betekenisvol zijn.
    """
    labels = kmeans.predict(to_km(stations["lat"], stations["lng"]))
    st = stations.assign(raw_zone=labels)
    zones = (st.sort_values("trips", ascending=False).groupby("raw_zone")
               .agg(name=("name", "first"), n_stations=("start_station_id", "size"), trips=("trips", "sum"),
                    lat=("lat", "mean"), lng=("lng", "mean"))
               .sort_values("trips", ascending=False).reset_index())
    zones["zone"] = np.arange(len(zones))
    zones["name"] = "rond " + zones["name"].astype(str)
    return zones[["zone", "raw_zone", "name", "n_stations", "trips", "lat", "lng"]]


def assign_zones(stations: pd.DataFrame, kmeans, zones: pd.DataFrame) -> pd.Series:
    """Zone-id (na hernummering) per station."""
    mapping = dict(zip(zones["raw_zone"], zones["zone"]))
    return pd.Series(kmeans.predict(to_km(stations["lat"], stations["lng"])), index=stations.index).map(mapping)


# ----------------------------------------------------------------------------------------------
# 5. Modeltabel: zone x uur + weer + kenmerken
# ----------------------------------------------------------------------------------------------
def build_zone_hour(station_hour: pd.DataFrame, station_zone: pd.Series, n_zones: int,
                    weather: pd.DataFrame, year: int = YEAR) -> pd.DataFrame:
    """Volledig rooster zone x uur (uren zonder vertrek = 0) met het weer erbij.

    `station_zone` is een Series met index start_station_id en waarde zone.
    """
    sh = station_hour.assign(zone=station_hour["start_station_id"].map(station_zone))
    sh = sh[sh["zone"].notna() & (sh["time"].dt.year == year)]
    counts = sh.groupby(["zone", "time"])[["trips", "members", "electric"]].sum()

    hours = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    grid = pd.MultiIndex.from_product([range(n_zones), hours], names=["zone", "time"])
    counts.index = counts.index.set_levels(counts.index.levels[0].astype(int), level=0)
    table = counts.reindex(grid, fill_value=0).reset_index()
    table = table.merge(weather, on="time", how="left")
    return table


def us_holidays(start, end) -> pd.DatetimeIndex:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    return USFederalHolidayCalendar().holidays(start=start, end=end)


def make_features(frame: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Kenmerken uit (zone, time, weer[, active_stations]). Dezelfde logica staat in deploy/citi_bike_demand/api.py."""
    time = pd.to_datetime(frame["time"])
    holidays = us_holidays(time.min().normalize(), time.max().normalize())
    out = pd.DataFrame(index=frame.index)
    out["zone"] = frame["zone"].astype(int)
    out["hour"] = time.dt.hour
    out["weekday"] = time.dt.dayofweek
    out["month"] = time.dt.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["is_holiday"] = time.dt.normalize().isin(holidays).astype(int)
    out["temperature_c"] = frame["temperature_c"].astype(float)
    out["precipitation_mm"] = frame["precipitation_mm"].astype(float)
    out["wind_kmh"] = frame["wind_kmh"].astype(float)
    out["year"] = time.dt.year
    if "active_stations" in frame:
        out["active_stations"] = frame["active_stations"].astype(int)
    if "snow_depth_cm" in frame:
        out["snow_depth_cm"] = frame["snow_depth_cm"].astype(float)
    return out[features]


# ----------------------------------------------------------------------------------------------
# 6. Split en evaluatie
# ----------------------------------------------------------------------------------------------
def split_labels(time: pd.Series) -> pd.Series:
    """'test' voor de laatste 7 dagen van elke maand, anders 'train'.

    Zo zit elk seizoen in de testset (een split 'laatste twee maanden' zou alleen winter testen),
    en zijn test-uren aaneengesloten blokken van een week (minder lek via naburige uren).
    """
    time = pd.to_datetime(time)
    last_week = time.dt.day > (time.dt.days_in_month - 7)
    return pd.Series(np.where(last_week, "test", "train"), index=time.index)


def split_last_months(time: pd.Series, months: int = 12) -> pd.Series:
    """'test' voor de laatste `months` maanden van de data, anders 'train' (studie op alle data).

    Met meerdere jaren testen we zoals het model gebruikt wordt: getraind op het verleden, gemeten op de
    meest recente periode. Twaalf maanden = elk seizoen één keer.
    """
    time = pd.to_datetime(time)
    cutoff = (time.max() + pd.Timedelta(hours=1)) - pd.DateOffset(months=months)
    return pd.Series(np.where(time >= cutoff, "test", "train"), index=time.index)


def backtest_folds(time: pd.Series, n_folds: int = 3, months: int = 12) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tijdsgebonden validatie ("backtesting", AWS-hoofdstuk 4) binnen de trainset.

    Fold i: trainen op alles vóór een validatieperiode van `months` maanden, valideren op die periode.
    De laatste fold valideert op de 12 maanden net vóór de testset. Geeft posities (geen labels) terug.
    """
    time = pd.to_datetime(time).reset_index(drop=True)
    end = time.max() + pd.Timedelta(hours=1)
    folds = []
    for i in range(n_folds, 0, -1):
        val_end = end - pd.DateOffset(months=months * (i - 1))
        val_start = val_end - pd.DateOffset(months=months)
        train = np.flatnonzero(time < val_start)
        val = np.flatnonzero((time >= val_start) & (time < val_end))
        folds.append((train, val))
    return folds


def cv_groups(time: pd.Series) -> pd.Series:
    """Groep voor GroupKFold: kalenderweek. Uren uit dezelfde week zitten altijd in dezelfde fold."""
    iso = pd.to_datetime(time).dt.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).rename("week")


def evaluate(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)    # Poisson-deviance vraagt > 0
    return {"mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "poisson_deviance": round(float(mean_poisson_deviance(y_true, y_pred)), 4)}


def _poisson_deviance_clipped(y_true, y_pred):
    from sklearn.metrics import mean_poisson_deviance

    return mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))


def _make_poisson_scorer():
    from sklearn.metrics import make_scorer

    # Een voorspelling van exact 0 is voor de Poisson-deviance niet toegelaten (log 0); we klemmen op 1e-6.
    return make_scorer(_poisson_deviance_clipped, greater_is_better=False)


POISSON_SCORER = _make_poisson_scorer()


def load_model_table(path: Path = DATA_DIR / "model_table.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def train_test(table: pd.DataFrame, features: list[str] = FEATURES):
    """(X_train, y_train, groups_train, X_test, y_test) volgens de split-kolom van de tabel."""
    X, y = make_features(table, features), table[TARGET]
    is_test = (table["split"] == "test").to_numpy()
    groups = cv_groups(table["time"])
    return X[~is_test], y[~is_test], groups[~is_test], X[is_test], y[is_test]


def cv_scores(model, X, y, groups=None, n_splits: int = 5, cv=None) -> dict:
    """MAE en Poisson-deviance per fold. Standaard GroupKFold op kalenderweken (studie 2024);
    met `cv=backtest_folds(...)` tijdsgebonden validatie (studie op alle data)."""
    from sklearn.model_selection import GroupKFold, cross_validate

    result = cross_validate(model, X, y, groups=groups, cv=cv if cv is not None else GroupKFold(n_splits=n_splits),
                            scoring={"mae": "neg_mean_absolute_error", "dev": POISSON_SCORER}, n_jobs=1)
    mae = (-result["test_mae"]).round(4).tolist()
    dev = (-result["test_dev"]).round(4).tolist()
    return {"cv_mae": mae, "cv_mae_mean": round(float(np.mean(mae)), 4),
            "cv_poisson_deviance": dev, "cv_fit_seconds": round(float(np.sum(result["fit_time"])), 1)}


class GroupMeanRegressor:
    """Baseline: voorspel het gemiddelde van de trainset voor dezelfde combinatie van `keys`.

    Bv. keys=["zone", "hour", "is_weekend"]: 'gemiddeld aantal vertrekken in deze zone op dit uur
    op een weekdag'. Onbekende combinaties krijgen het globale gemiddelde.
    Met `recent_months` tellen alleen de laatste maanden van de trainset mee ("hoe was het vorig jaar?");
    daarvoor zijn de kolommen `year` en `month` nodig.
    """

    def __init__(self, keys=("zone", "hour", "is_weekend"), recent_months=None):
        self.keys = keys
        self.recent_months = recent_months

    def get_params(self, deep=True):
        return {"keys": self.keys, "recent_months": self.recent_months}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        keys = list(self.keys)
        y = pd.Series(np.asarray(y), index=X.index)
        if self.recent_months:
            period = X["year"] * 12 + X["month"]
            recent = period > period.max() - self.recent_months
            X, y = X[recent], y[recent]
        self.global_mean_ = float(np.mean(y))
        self.means_ = (pd.DataFrame(X[keys]).assign(_y=y.to_numpy()).groupby(keys)["_y"].mean()) if keys else None
        return self

    def predict(self, X):
        if not self.keys:
            return np.full(len(X), self.global_mean_)
        index = pd.MultiIndex.from_frame(X[list(self.keys)]) if len(self.keys) > 1 else X[self.keys[0]]
        return self.means_.reindex(index).fillna(self.global_mean_).to_numpy()


# ----------------------------------------------------------------------------------------------
# 7. Van ruwe aggregaten naar de modeltabel (gebruikt door 04_prepare_data)
# ----------------------------------------------------------------------------------------------
N_ZONES = 30
# Dagen met een gat in de data (geen of bijna geen ritten door een ontbrekend/afgekapt bestand), zie 02b §6.
# Regel: minder dan 3% van de mediaan van de 4 weken eromheen. Het zijn allemaal zware sneeuwstormen waarbij
# Citi Bike het systeem stillegde: geen aanbod, dus "0 vertrekken" zegt niets over de vraag.
EXCLUDE_DAYS: list[str] = ["2016-01-23", "2016-01-24", "2016-01-25", "2016-01-26", "2017-02-09", "2017-03-14",
                           "2017-03-15", "2017-03-16", "2020-12-17", "2021-02-01", "2021-02-02", "2026-02-23"]
# Stations die geen echte klantenvraag zijn (depots, laadkades, testlocaties): zie 02_eda_cleaning.
NON_PUBLIC_STATION = r"(?i)depot|loading dock|warehouse|\blab\b|\bshop\b|mechanic|nycbs"


def clean_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """Alleen publieke stations binnen NYC. Geeft een kopie met een kolom `keep` en de reden."""
    st = stations.copy()
    st["reason"] = ""
    st.loc[st["name"].fillna("").str.contains(NON_PUBLIC_STATION), "reason"] = "geen publiek station"
    st.loc[st["start_station_id"].str.match(r"(?i)^SYS"), "reason"] = "geen publiek station"
    st.loc[st["start_station_id"].str.match(r"^(JC|HB)"), "reason"] = "Jersey City / Hoboken (apart systeem)"
    st["keep"] = st["reason"] == ""
    return st


def prepare_model_table(station_hour: pd.DataFrame, stations: pd.DataFrame, weather: pd.DataFrame,
                        n_zones: int = N_ZONES):
    """Stations opschonen -> zones -> rooster zone x uur + weer + split. Geeft (table, zones, stations)."""
    st = clean_stations(stations)
    public = st[st["keep"]].reset_index(drop=True)
    kmeans = fit_zones(public, n_zones)
    zones = zone_table(public, kmeans)
    public["zone"] = assign_zones(public, kmeans, zones)
    station_zone = public.set_index("start_station_id")["zone"]

    table = build_zone_hour(station_hour, station_zone, n_zones, weather)
    table["split"] = split_labels(table["time"])
    zones_out = zones.drop(columns="raw_zone")
    return table, zones_out, public


def save_results(name: str, metrics: dict, predictions: pd.DataFrame | None = None,
                 metrics_dir: Path = METRICS_DIR, pred_dir: Path = PRED_DIR) -> None:
    """Metrics naar <metrics_dir>/<name>.json; testvoorspellingen naar <pred_dir>/<name>.parquet."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    if predictions is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(pred_dir / f"{name}.parquet", index=False)


# ----------------------------------------------------------------------------------------------
# 8. Modeltabel over alle jaren (gebruikt door 04_prepare_data)
# ----------------------------------------------------------------------------------------------
def recent_station_trips(station_dir: Path, since: pd.Timestamp) -> pd.Series:
    """Vertrekken per station sinds `since` (om de zones op het netwerk van vandaag te leggen)."""
    parts = []
    for path in sorted(station_dir.glob("*.parquet")):
        sh = pd.read_parquet(path, columns=["start_station_id", "time", "trips"])
        sh = sh[sh["time"] >= since]
        if len(sh):
            parts.append(sh.groupby("start_station_id")["trips"].sum())
    return pd.concat(parts).groupby(level=0).sum()


def station_locations(stations: pd.DataFrame) -> pd.Series:
    """Locatiesleutel per station-id (coördinaten op 4 decimalen, ± 10 m): zelfde plek = zelfde station."""
    key = stations["lat"].round(4).astype(str) + "," + stations["lng"].round(4).astype(str)
    return pd.Series(key.to_numpy(), index=stations["start_station_id"])


def prepare_full_table(station_dir: Path, stations: pd.DataFrame, weather: pd.DataFrame,
                       n_zones: int = N_ZONES, recent_months: int = 12, exclude_days=()):
    """Rooster zone x uur over alle jaren. Geeft (table, zones, stations met zone) terug.

    1. Stations opschonen (zelfde regels als 2024).
    2. Zones fitten op het netwerk van de laatste `recent_months` maanden (gewogen met de recente ritten):
       het model voorspelt de toekomst, dus de zones moeten de stad van vandaag beschrijven.
    3. Elk station uit elk jaar (ook de oude id's van vóór 2021) krijgt de zone van zijn locatie.
    4. Per zone en uur: vertrekken; per zone en maand: aantal actieve stations (>= 1 vertrek), geteld als
       unieke **locaties**: een station dat van id veranderde (oud -> nieuw systeem) telt zo maar één keer.
    5. Uren waarin een zone nog geen enkel actief station had (bv. de Bronx in 2013), vallen weg:
       daar was geen aanbod, dus "0 vertrekken" zegt niets over de vraag.
    6. Dagen in `exclude_days` (gaten in de data, zie 02b) vallen weg om dezelfde reden.
    """
    st = clean_stations(stations)
    public = st[st["keep"]].reset_index(drop=True)
    last = public["last_time"].max()
    since = (last + pd.Timedelta(hours=1)) - pd.DateOffset(months=recent_months)
    recent = recent_station_trips(station_dir, since)
    current = public[public["start_station_id"].isin(recent.index)].copy()
    current["trips"] = current["start_station_id"].map(recent).astype(int)

    kmeans = fit_zones(current, n_zones)
    zones = zone_table(current, kmeans)
    public["zone"] = assign_zones(public, kmeans, zones).astype(int)
    station_zone = public.set_index("start_station_id")["zone"]
    station_location = station_locations(public)

    counts, active = [], []
    for path in sorted(station_dir.glob("*.parquet")):
        sh = pd.read_parquet(path)
        sh["zone"] = sh["start_station_id"].map(station_zone)
        sh = sh[sh["zone"].notna()]
        sh["zone"] = sh["zone"].astype(int)
        counts.append(sh.groupby(["zone", "time"])[["trips", "members", "electric"]].sum())
        active.append(sh.assign(month=sh["time"].dt.to_period("M"),
                                location=sh["start_station_id"].map(station_location))
                        [["zone", "month", "location"]].drop_duplicates())
    counts = pd.concat(counts).groupby(level=[0, 1]).sum()
    active = (pd.concat(active).drop_duplicates().groupby(["zone", "month"]).size().rename("active_stations"))

    hours = pd.date_range(counts.index.get_level_values("time").min().floor("D"),
                          counts.index.get_level_values("time").max().ceil("D") - pd.Timedelta(hours=1), freq="h")
    grid = pd.MultiIndex.from_product([range(n_zones), hours], names=["zone", "time"])
    table = counts.reindex(grid, fill_value=0).reset_index()
    table["month_period"] = table["time"].dt.to_period("M")
    table = table.merge(active.reset_index().rename(columns={"month": "month_period"}),
                        on=["zone", "month_period"], how="left")
    table["active_stations"] = table["active_stations"].fillna(0).astype(int)
    table = table[table["active_stations"] > 0].drop(columns="month_period")
    if len(exclude_days):
        table = table[~table["time"].dt.normalize().isin(pd.to_datetime(list(exclude_days)))]
    table = table.merge(weather, on="time", how="left")
    table["split"] = split_last_months(table["time"])

    latest = active.reset_index()
    latest = latest[latest["month"] == latest["month"].max()].set_index("zone")["active_stations"]
    zones["active_stations"] = zones["zone"].map(latest).fillna(0).astype(int)
    return table.reset_index(drop=True), zones.drop(columns="raw_zone"), public
