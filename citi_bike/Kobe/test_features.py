"""Controle: de API berekent exact dezelfde kenmerken als de notebooks/train.py.

make_features() staat twee keer in de repo (citibike.py en deploy/citi_bike_demand/api.py), omdat alleen
de map deploy/ naar de VM gaat. Deze test faalt zodra de twee uit elkaar lopen.
Uitvoeren:  python -m pytest citi_bike/Kobe/test_features.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import citibike as cb  # noqa: E402

API_PATH = HERE.parent.parent / "deploy" / "citi_bike_demand" / "api.py"


def load_api():
    spec = importlib.util.spec_from_file_location("citi_bike_demand_api", API_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)       # laadt ook model.pkl
    return module


def example_frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 366 * 24, n), unit="h")
    return pd.DataFrame({"zone": rng.integers(0, 5, n), "time": times,
                         "temperature_c": rng.normal(15, 8, n), "precipitation_mm": rng.exponential(0.5, n),
                         "wind_kmh": rng.uniform(0, 40, n), "active_stations": rng.integers(1, 120, n),
                         "snow_depth_cm": rng.exponential(2, n)})


def test_api_features_match_training_features():
    api = load_api()
    frame = example_frame()
    pd.testing.assert_frame_equal(api.make_features(frame), cb.make_features(frame, api.FEATURES), check_dtype=False)


def test_holidays_are_flagged():
    frame = pd.DataFrame({"zone": [0, 0], "time": ["2024-07-04 12:00", "2024-07-05 12:00"],
                          "temperature_c": [25, 25], "precipitation_mm": [0, 0], "wind_kmh": [5, 5]})
    assert cb.make_features(frame, cb.FEATURES_2024)["is_holiday"].tolist() == [1, 0]


def test_split_last_months():
    times = pd.Series(pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h"))
    labels = cb.split_last_months(times)
    assert labels[times < "2025-01-01"].eq("train").all() and labels[times >= "2025-01-01"].eq("test").all()


def test_backtest_folds_never_train_on_the_future():
    times = pd.Series(pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h"))
    for train, val in cb.backtest_folds(times):
        assert times.iloc[train].max() < times.iloc[val].min()


def test_split_puts_last_week_of_each_month_in_test():
    times = pd.Series(pd.to_datetime(["2024-02-22", "2024-02-23", "2024-12-24", "2024-12-25"]))
    assert cb.split_labels(times).tolist() == ["train", "test", "train", "test"]


def test_bundle_matches_training_features():
    api = load_api()
    assert list(api.FEATURES) == cb.FEATURES
