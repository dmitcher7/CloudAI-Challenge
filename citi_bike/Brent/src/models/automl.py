"""Optional FLAML comparison using the same chronological holdout.

FLAML is included in `requirements.txt`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import PROCESSED_DIR, TARGET, WEATHER_PATH
from src.data.weather import attach_hourly_weather
from src.data.sample import systematic_parquet_sample
from src.features.build import make_features
from src.features.history import apply_history, fit_transform_history
from src.models.train import temporal_split


def run_automl(
    data_path: Path,
    seconds: int = 300,
    max_rows: int = 100_000,
) -> dict:
    """Run FLAML on a memory-safe, time-ordered sample of the Parquet data.

    The complete Citi Bike dataset is too large to materialize as a pandas
    frame on a normal laptop. The sample is spread over the complete Parquet
    file, then sorted and split chronologically, so FLAML still sees the full
    time range without exhausting RAM. The selected configuration can be
    retrained on the full development set afterwards.
    """
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise RuntimeError("Install the project requirements to enable FLAML AutoML") from exc

    frame = systematic_parquet_sample(data_path, max_rows=max_rows).sort_values(
        "started_at"
    )
    frame = attach_hourly_weather(frame, WEATHER_PATH)
    train, test = temporal_split(frame)
    train, route_history = fit_transform_history(train, TARGET)
    test = apply_history(test, route_history)
    x_train, x_test = make_features(train), make_features(test)
    # FLAML handles category dtype efficiently and avoids an enormous station one-hot matrix.
    for column in ["rideable_type", "member_casual", "start_station_id", "end_station_id"]:
        x_train[column] = x_train[column].astype("category")
        x_test[column] = x_test[column].astype("category")

    automl = AutoML()
    automl.fit(
        X_train=x_train,
        y_train=train[TARGET],
        task="regression",
        metric="mae",
        time_budget=seconds,
        # Keep the notebook stable on Windows/Python 3.13. FLAML's default
        # list may select LightGBM/XGBoost/CatBoost, whose native binaries can
        # terminate the kernel with an access violation on some installations.
        estimator_list=["extra_tree"],
        n_jobs=1,
        eval_method="holdout",
        split_type="time",
        seed=42,
        verbose=2,
    )
    prediction = automl.predict(x_test)
    result = {
        "estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "validation_loss": automl.best_loss,
        "temporal_holdout_mae_minutes": float(mean_absolute_error(test[TARGET], prediction)),
        "time_budget_seconds": seconds,
        "sample_rows": int(len(frame)),
    }
    print(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "trips.parquet")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=100_000)
    args = parser.parse_args()
    run_automl(args.data, args.seconds, args.max_rows)


if __name__ == "__main__":
    main()
