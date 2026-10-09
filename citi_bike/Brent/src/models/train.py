"""Train, compare and serialize leakage-safe duration models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    MAX_DURATION_MINUTES,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
    WEATHER_PATH,
)
from src.data.weather import WEATHER_FEATURES, attach_hourly_weather
from src.features.build import make_features
from src.models.transforms import inverse_log_duration

CATEGORICAL = ["rideable_type", "member_casual", "start_station_id", "end_station_id"]
NUMERIC = [
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "direct_distance_km",
    "delta_lat",
    "delta_lng",
    "start_hour_sin",
    "start_hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    *WEATHER_FEATURES,
]
def temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    """Split chronologically so future records never inform past evaluation."""
    ordered = frame.sort_values("started_at").reset_index(drop=True)
    split_at = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction))))
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def _ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categories",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
            ("numbers", SimpleImputer(strategy="median"), NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models(quick: bool = False) -> dict[str, object]:
    """Models chosen for complementary bias/variance and compute profiles."""
    one_hot = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                CATEGORICAL,
            ),
            (
                "numbers",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
        ]
    )
    estimators: dict[str, object] = {
        "median_baseline": DummyRegressor(strategy="median"),
        "ridge": Pipeline([("preprocess", one_hot), ("model", Ridge(alpha=10.0))]),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _ordinal_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.08,
                        max_iter=60 if quick else 250,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocess", _ordinal_preprocessor()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=30 if quick else 250,
                        min_samples_leaf=5,
                        max_features=0.8,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    # A log target reduces domination by the right tail; inverse predictions remain minutes.
    return {
        name: TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=inverse_log_duration,
            check_inverse=False,
        )
        for name, estimator in estimators.items()
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "median_ae_minutes": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cap_rows_across_time(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Deterministically thin large data while retaining the full time range."""
    if not max_rows or len(frame) <= max_rows:
        return frame
    indices = np.linspace(0, len(frame) - 1, max_rows, dtype=int)
    return frame.iloc[indices].copy()


def train_and_compare(
    data_path: Path,
    model_path: Path = MODEL_DIR / "duration_model.joblib",
    metrics_path: Path = REPORT_DIR / "metrics.csv",
    max_rows: int | None = 500_000,
    quick: bool = False,
    weather_path: Path = WEATHER_PATH,
) -> tuple[dict, pd.DataFrame]:
    columns = [
        "started_at",
        "rideable_type",
        "member_casual",
        "start_station_id",
        "end_station_id",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        TARGET,
    ]
    available_columns = set(pq.read_schema(data_path).names)
    columns.extend(name for name in WEATHER_FEATURES if name in available_columns)
    frame = pd.read_parquet(data_path, columns=columns).sort_values("started_at")
    frame = cap_rows_across_time(frame, max_rows)
    frame = attach_hourly_weather(frame, weather_path)
    if len(frame) < 100:
        raise ValueError("At least 100 valid rows are needed for model comparison")

    train, test = temporal_split(frame)
    x_train, x_test = make_features(train), make_features(test)
    y_train, y_test = train[TARGET], test[TARGET]
    results = []
    fitted = {}

    for name, model in candidate_models(quick=quick).items():
        print(f"Training {name} on {len(train):,} rows …", flush=True)
        model.fit(x_train, y_train)
        predictions = np.clip(model.predict(x_test), 0, MAX_DURATION_MINUTES)
        fitted[name] = model
        results.append(
            {
                "model": name,
                **regression_metrics(y_test, predictions),
                "train_rows": len(train),
                "test_rows": len(test),
                "test_start": str(test["started_at"].min()),
                "test_end": str(test["started_at"].max()),
            }
        )

    metrics = pd.DataFrame(results).sort_values("mae_minutes").reset_index(drop=True)
    winner_name = str(metrics.iloc[0]["model"])
    bundle = {
        "model": fitted[winner_name],
        "model_name": winner_name,
        "features": list(x_train.columns),
        "target": TARGET,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_period": [str(train["started_at"].min()), str(train["started_at"].max())],
        "holdout_period": [str(test["started_at"].min()), str(test["started_at"].max())],
        "metrics": metrics.iloc[0].to_dict(),
        "limitations": (
            "Destination is treated as planned input; weather is hourly and city-level. "
            "Events, traffic, route choice and bike availability are unavailable."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics.to_csv(metrics_path, index=False)
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps({key: value for key, value in bundle.items() if key != "model"}, indent=2),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"Selected {winner_name}; wrote {model_path}")
    return bundle, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "trips.parquet")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "duration_model.joblib")
    parser.add_argument("--metrics", type=Path, default=REPORT_DIR / "metrics.csv")
    parser.add_argument("--weather", type=Path, default=WEATHER_PATH)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="maximum rows sampled evenly across the full time range; 0 means all rows",
    )
    parser.add_argument("--quick", action="store_true", help="small estimators for CI/smoke tests")
    args = parser.parse_args()
    train_and_compare(
        args.data,
        args.model,
        args.metrics,
        max_rows=args.max_rows or None,
        quick=args.quick,
        weather_path=args.weather,
    )


if __name__ == "__main__":
    main()
