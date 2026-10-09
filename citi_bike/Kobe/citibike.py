"""Gedeelde code voor de Citi Bike-notebooks van Kobe: vraag (vertrekken) per zone per uur.

De notebooks (01 t/m 06) én train.py (dat het model voor de API schrijft) gebruiken deze functies,
zodat de data overal op exact dezelfde manier verwerkt wordt.

Overzicht:
    download_months()      zip's van de officiële S3-bucket halen (+ manifest met SHA-256)
    aggregate_month()      één maand inlezen in stukken -> vertrekken per station per uur
    download_weather()     uurlijks weer in NYC via de Open-Meteo archive API
    fit_zones()            stations groeperen tot zones (KMeans, gewogen met het aantal ritten)
    build_zone_hour()      volledig rooster zone x uur (ook uren zonder ritten = 0)
    make_features()        modelkenmerken; dezelfde code staat in deploy/citi_bike_demand/api.py
    split_labels()         vaste train/test-split (laatste 7 dagen van elke maand = test)
    evaluate()             MAE, RMSE, R² en Poisson-deviance
"""
from __future__ import annotations

import hashlib
import json
import os
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

BUCKET_URL = "https://s3.amazonaws.com/tripdata/"
YEAR = 2024
MONTHS = [f"{YEAR}{m:02d}" for m in range(1, 13)]
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
FEATURES = ["zone", "hour", "weekday", "month", "is_weekend", "is_holiday",
            "temperature_c", "precipitation_mm", "wind_kmh"]
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


def download_month(yyyymm: str, raw_dir: Path = RAW_DIR) -> dict:
    """Download één maand-zip (als die nog niet volledig aanwezig is) en geef een manifest-regel terug."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = f"{yyyymm}-citibike-tripdata.zip"
    url = BUCKET_URL + name
    target = raw_dir / name

    size = int(requests.head(url, timeout=30).headers["Content-Length"])
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

    with ZipFile(target) as archive:
        csvs = sorted(n for n in archive.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n)
    return {"month": yyyymm, "url": url, "file": name, "bytes": size, "csv_parts": csvs,
            "sha256": _sha256(target), "status": status,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def download_months(months=MONTHS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download alle maanden en schrijf data/raw/manifest.json (bron, grootte, checksum)."""
    records = []
    for yyyymm in months:
        record = download_month(yyyymm, raw_dir)
        print(f"{yyyymm}: {record['status']:<11} {record['bytes'] / 1e6:6.0f} MB, {len(record['csv_parts'])} CSV-delen")
        records.append(record)
    (raw_dir / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------------------
# 2. Inlezen en aggregeren (per maand, in stukken)
# ----------------------------------------------------------------------------------------------
def iter_chunks(zip_path: Path, chunksize: int = 500_000):
    """Lees alle CSV-delen van een maand-zip in stukken van `chunksize` rijen (allemaal tekst)."""
    with ZipFile(zip_path) as archive:
        for name in sorted(n for n in archive.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n):
            with archive.open(name) as handle:
                yield from pd.read_csv(handle, dtype=str, usecols=RAW_COLUMNS, chunksize=chunksize)


def parse_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Tekst -> juiste types, plus de ritduur in minuten."""
    out = chunk.copy()
    for column in ["started_at", "ended_at"]:
        out[column] = pd.to_datetime(out[column], format="ISO8601", errors="coerce")
    for column in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["duration_min"] = (out["ended_at"] - out["started_at"]).dt.total_seconds() / 60
    return out


def normalize_station_id(ids: pd.Series) -> pd.Series:
    """'3423.1' -> '3423.10': sommige CSV-delen bewaarden de id als getal, waardoor de laatste 0 wegviel.

    Officiële id's met een punt hebben altijd twee decimalen (bv. 7407.13); tekst-id's (HB101, JC023) blijven.
    """
    one_decimal = ids.str.fullmatch(r"\d+\.\d", na=False)
    return ids.mask(one_decimal, ids + "0")


def departure_mask(trips: pd.DataFrame) -> pd.Series:
    """Welke rijen tellen als vertrek? Starttijd + startstation + coördinaten binnen NYC gekend.

    De ritduur speelt hier bewust geen rol: ook een rit die (te) lang duurt, is een fiets die vertrok.
    Zie 02_eda_cleaning voor de onderbouwing.
    """
    lat_ok = trips["start_lat"].between(*NYC_BBOX["lat"])
    lng_ok = trips["start_lng"].between(*NYC_BBOX["lng"])
    return trips["started_at"].notna() & trips["start_station_id"].notna() & lat_ok & lng_ok


def aggregate_month(zip_path: Path, yyyymm: str, sample_frac: float = 0.01, seed: int = SEED):
    """Eén passage door een maand: vertrekken per station per uur, stationslijst, audit en steekproef.

    Geeft (station_hour, stations, audit, sample) terug. De steekproef is ruw (vóór elke filtering)
    zodat de EDA ook kan tonen wat we weggooien.
    """
    rng = np.random.default_rng(seed)
    station_hour, stations, samples, hashes = [], [], [], []
    audit = {"month": yyyymm, "rows": 0, "missing": {c: 0 for c in RAW_COLUMNS},
             "no_start_time": 0, "no_start_station": 0, "outside_nyc": 0, "departures": 0,
             "duration_negative": 0, "duration_over_24h": 0, "no_end_time": 0, "station_id_fixed": 0}

    for chunk in iter_chunks(zip_path):
        audit["rows"] += len(chunk)
        for column in RAW_COLUMNS:
            audit["missing"][column] += int(chunk[column].isna().sum())
        hashes.append(pd.util.hash_pandas_object(chunk["ride_id"], index=False).to_numpy())
        samples.append(chunk.loc[rng.random(len(chunk)) < sample_frac])

        trips = parse_chunk(chunk)
        fixed = normalize_station_id(trips["start_station_id"])
        audit["station_id_fixed"] += int((trips["start_station_id"].notna() & (fixed != trips["start_station_id"])).sum())
        trips["start_station_id"] = fixed
        audit["no_start_time"] += int(trips["started_at"].isna().sum())
        audit["no_start_station"] += int(trips["start_station_id"].isna().sum())
        audit["no_end_time"] += int(trips["ended_at"].isna().sum())
        audit["duration_negative"] += int((trips["duration_min"] < 0).sum())
        audit["duration_over_24h"] += int((trips["duration_min"] > 24 * 60).sum())

        keep = departure_mask(trips)
        audit["outside_nyc"] += int((trips["start_station_id"].notna() & trips["started_at"].notna() & ~keep).sum())
        trips = trips.loc[keep]
        audit["departures"] += len(trips)

        trips = trips.assign(time=trips["started_at"].dt.floor("h"),
                             member=(trips["member_casual"] == "member").astype("int32"),
                             electric=(trips["rideable_type"] == "electric_bike").astype("int32"))
        station_hour.append(trips.groupby(["start_station_id", "time"])
                                 .agg(trips=("ride_id", "size"), members=("member", "sum"),
                                      electric=("electric", "sum")).reset_index())
        stations.append(trips.groupby("start_station_id")
                             .agg(name=("start_station_name", "first"), lat_sum=("start_lat", "sum"),
                                  lng_sum=("start_lng", "sum"), trips=("ride_id", "size")).reset_index())

    all_hashes = np.concatenate(hashes)
    audit["duplicate_ride_ids"] = int(len(all_hashes) - len(np.unique(all_hashes)))

    station_hour = (pd.concat(station_hour).groupby(["start_station_id", "time"], as_index=False)
                      [["trips", "members", "electric"]].sum())
    stations = (pd.concat(stations).groupby("start_station_id", as_index=False)
                  .agg(name=("name", "first"), lat_sum=("lat_sum", "sum"), lng_sum=("lng_sum", "sum"),
                       trips=("trips", "sum")))
    sample = pd.concat(samples, ignore_index=True).assign(source_month=yyyymm)
    return station_hour, stations, audit, sample


def aggregate_all(months=MONTHS, raw_dir: Path = RAW_DIR, out_dir: Path = DATA_DIR) -> dict:
    """Aggregeer alle maanden en schrijf de vier tussenbestanden naar data/.

    data/station_hour.parquet   vertrekken per station per uur (enkel uren met >= 1 vertrek)
    data/stations.parquet       station-id, naam, gemiddelde coördinaten, totaal aantal ritten
    data/audit.json             per maand: rijen, ontbrekende waarden, duplicaten, wat werd weggelaten
    data/trips_sample.parquet   ruwe steekproef van 1% van alle ritten (voor de EDA)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    station_hour, stations, audits, samples = [], [], [], []
    for yyyymm in months:
        started = datetime.now()
        sh, st, audit, sample = aggregate_month(raw_dir / f"{yyyymm}-citibike-tripdata.zip", yyyymm)
        station_hour.append(sh), stations.append(st), audits.append(audit), samples.append(sample)
        print(f"{yyyymm}: {audit['rows']:>10,} rijen -> {audit['departures']:>10,} vertrekken "
              f"({(datetime.now() - started).seconds} s)")

    station_hour = (pd.concat(station_hour).groupby(["start_station_id", "time"], as_index=False)
                      [["trips", "members", "electric"]].sum())
    stations = (pd.concat(stations).groupby("start_station_id", as_index=False)
                  .agg(name=("name", "first"), lat_sum=("lat_sum", "sum"), lng_sum=("lng_sum", "sum"),
                       trips=("trips", "sum")))
    stations["lat"] = stations.pop("lat_sum") / stations["trips"]
    stations["lng"] = stations.pop("lng_sum") / stations["trips"]
    stations = stations[["start_station_id", "name", "lat", "lng", "trips"]]

    station_hour.to_parquet(out_dir / "station_hour.parquet", index=False)
    stations.to_parquet(out_dir / "stations.parquet", index=False)
    pd.concat(samples, ignore_index=True).to_parquet(out_dir / "trips_sample.parquet", index=False)
    (out_dir / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    return {"station_hour": station_hour, "stations": stations, "audit": audits}


# ----------------------------------------------------------------------------------------------
# 3. Weer (Open-Meteo, gratis, geen API-sleutel)
# ----------------------------------------------------------------------------------------------
WEATHER_VARIABLES = {"temperature_2m": "temperature_c", "precipitation": "precipitation_mm",
                     "wind_speed_10m": "wind_kmh", "snowfall": "snowfall_cm",
                     "relative_humidity_2m": "humidity_pct"}


def download_weather(start: str = f"{YEAR}-01-01", end: str = f"{YEAR}-12-31",
                     path: Path = DATA_DIR / f"weather_{YEAR}.parquet") -> pd.DataFrame:
    """Uurlijks weer voor NYC in lokale tijd (zoals de ritdata). Wordt gecachet in data/."""
    if path.exists():
        return pd.read_parquet(path)
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
        "latitude": NYC_CENTER[0], "longitude": NYC_CENTER[1], "start_date": start, "end_date": end,
        "hourly": ",".join(WEATHER_VARIABLES), "timezone": "America/New_York", "wind_speed_unit": "kmh"})
    response.raise_for_status()
    hourly = response.json()["hourly"]
    weather = pd.DataFrame(hourly).rename(columns={"time": "time", **WEATHER_VARIABLES})
    weather["time"] = pd.to_datetime(weather["time"])
    # Zomertijd: het "verdwenen" uur in maart ontbreekt, het dubbele uur in november komt twee keer voor.
    weather = weather.drop_duplicates("time").set_index("time")
    full = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h")
    weather = weather.reindex(full).interpolate(limit=2).rename_axis("time").reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path, index=False)
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


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Kenmerken uit (zone, time, weer). Exact dezelfde logica staat in deploy/citi_bike_demand/api.py."""
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
    return out[FEATURES]


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


def train_test(table: pd.DataFrame):
    """(X_train, y_train, groups_train, X_test, y_test) volgens de vaste split."""
    X, y = make_features(table), table[TARGET]
    is_test = (table["split"] == "test").to_numpy()
    groups = cv_groups(table["time"])
    return X[~is_test], y[~is_test], groups[~is_test], X[is_test], y[is_test]


def cv_scores(model, X, y, groups, n_splits: int = 5) -> dict:
    """GroupKFold op kalenderweken binnen de trainset: MAE en Poisson-deviance per fold."""
    from sklearn.model_selection import GroupKFold, cross_validate

    result = cross_validate(model, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
                            scoring={"mae": "neg_mean_absolute_error", "dev": POISSON_SCORER}, n_jobs=1)
    mae = (-result["test_mae"]).round(4).tolist()
    dev = (-result["test_dev"]).round(4).tolist()
    return {"cv_mae": mae, "cv_mae_mean": round(float(np.mean(mae)), 4),
            "cv_poisson_deviance": dev, "cv_fit_seconds": round(float(np.sum(result["fit_time"])), 1)}


class GroupMeanRegressor:
    """Baseline: voorspel het gemiddelde van de trainset voor dezelfde combinatie van `keys`.

    Bv. keys=["zone", "hour", "is_weekend"]: 'gemiddeld aantal vertrekken in deze zone op dit uur
    op een weekdag'. Onbekende combinaties krijgen het globale gemiddelde.
    """

    def __init__(self, keys=("zone", "hour", "is_weekend")):
        self.keys = keys

    def get_params(self, deep=True):
        return {"keys": self.keys}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        keys = list(self.keys)
        self.global_mean_ = float(np.mean(y))
        self.means_ = (pd.DataFrame(X[keys]).assign(_y=np.asarray(y)).groupby(keys)["_y"].mean()) if keys else None
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


def save_results(name: str, metrics: dict, predictions: pd.DataFrame | None = None) -> None:
    """Metrics naar models/metrics/<name>.json; testvoorspellingen naar models/predictions/<name>.parquet."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    if predictions is not None:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(PRED_DIR / f"{name}.parquet", index=False)
