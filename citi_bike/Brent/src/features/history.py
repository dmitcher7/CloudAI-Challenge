"""Leakage-safe historical duration features for stations and planned routes."""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    "route_typical_minutes",
    "route_history_log_count",
    "start_typical_minutes",
    "end_typical_minutes",
]
SMOOTHING = 20.0


def _clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("unknown").astype(str)


def _route_key(frame: pd.DataFrame) -> pd.Series:
    return _clean_key(frame["start_station_id"]) + "\x1f" + _clean_key(frame["end_station_id"])


def _expanding_estimate(
    keys: pd.Series, target: pd.Series, global_prior: pd.Series, smoothing: float
) -> tuple[pd.Series, pd.Series]:
    helper = pd.DataFrame({"key": keys.to_numpy(), "target": target.to_numpy()})
    grouped = helper.groupby("key", sort=False, dropna=False)["target"]
    prior_count = grouped.cumcount().astype(float)
    prior_sum = grouped.cumsum() - helper["target"]
    estimate = (prior_sum + smoothing * global_prior.to_numpy()) / (prior_count + smoothing)
    return estimate, prior_count


def fit_transform_history(
    train: pd.DataFrame, target_column: str, smoothing: float = SMOOTHING
) -> tuple[pd.DataFrame, dict]:
    """Create past-only train features and fit mappings for future rows."""
    result = train.copy()
    target = pd.to_numeric(result[target_column], errors="raise").reset_index(drop=True)
    result = result.reset_index(drop=True)
    global_prior = target.expanding().mean().shift().fillna(10.0)

    route_estimate, route_count = _expanding_estimate(
        _route_key(result), target, global_prior, smoothing
    )
    start_estimate, _ = _expanding_estimate(
        _clean_key(result["start_station_id"]), target, global_prior, smoothing
    )
    end_estimate, _ = _expanding_estimate(
        _clean_key(result["end_station_id"]), target, global_prior, smoothing
    )
    result["route_typical_minutes"] = route_estimate
    result["route_history_log_count"] = np.log1p(route_count)
    result["start_typical_minutes"] = start_estimate
    result["end_typical_minutes"] = end_estimate

    route_keys = _route_key(result)
    start_keys = _clean_key(result["start_station_id"])
    end_keys = _clean_key(result["end_station_id"])

    def aggregate(keys: pd.Series) -> dict[str, tuple[float, int]]:
        stats = pd.DataFrame({"key": keys, "target": target}).groupby("key").target.agg(
            ["mean", "count"]
        )
        return {
            str(key): (float(row["mean"]), int(row["count"]))
            for key, row in stats.iterrows()
        }

    history = {
        "global_mean": float(target.mean()),
        "smoothing": float(smoothing),
        "route": aggregate(route_keys),
        "start": aggregate(start_keys),
        "end": aggregate(end_keys),
    }
    return result, history


def apply_history(frame: pd.DataFrame, history: dict | None) -> pd.DataFrame:
    """Apply fitted train-only history to validation or inference rows."""
    result = frame.copy()
    if not history:
        for name in HISTORY_FEATURES:
            result[name] = np.nan
        return result

    global_mean = float(history["global_mean"])
    smoothing = float(history.get("smoothing", SMOOTHING))

    def map_estimate(keys: pd.Series, mapping: dict) -> tuple[np.ndarray, np.ndarray]:
        records = keys.map(mapping)
        means = records.map(lambda value: value[0] if isinstance(value, tuple) else global_mean)
        counts = records.map(lambda value: value[1] if isinstance(value, tuple) else 0).astype(float)
        estimate = (means.to_numpy() * counts.to_numpy() + smoothing * global_mean) / (
            counts.to_numpy() + smoothing
        )
        return estimate, counts.to_numpy()

    route_estimate, route_count = map_estimate(_route_key(result), history["route"])
    start_estimate, _ = map_estimate(_clean_key(result["start_station_id"]), history["start"])
    end_estimate, _ = map_estimate(_clean_key(result["end_station_id"]), history["end"])
    result["route_typical_minutes"] = route_estimate
    result["route_history_log_count"] = np.log1p(route_count)
    result["start_typical_minutes"] = start_estimate
    result["end_typical_minutes"] = end_estimate
    return result
