"""Fair FLAML comparison on the same nested chronological split as manual models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    MAX_DURATION_MINUTES,
    PROCESSED_DIR,
    REPORT_DIR,
    TARGET,
    WEATHER_PATH,
)
from src.data.sample import systematic_parquet_sample
from src.data.weather import WEATHER_FEATURES, attach_hourly_weather
from src.features.build import make_features
from src.features.history import apply_history, fit_transform_history
from src.models.train import regression_metrics, temporal_split


# These learners were smoke-tested on the project's Windows/Python 3.13 setup.
# LightGBM is intentionally not a default: its native wheel raises an access
# violation in this runtime and would abort the complete multi-model search.
DEFAULT_ESTIMATORS = ("histgb", "extra_tree", "rf", "catboost", "xgboost")
CATEGORICAL_FEATURES = ("rideable_type", "member_casual", "start_station_id", "end_station_id")


def _comparison_data(
    data_path: Path,
    max_rows: int | None,
    weather_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Build the exact selection train/validation partitions used by manual models."""
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
    if max_rows:
        frame = systematic_parquet_sample(data_path, max_rows=max_rows, columns=columns)
    else:
        frame = pd.read_parquet(data_path, columns=columns)
    frame = attach_hourly_weather(frame.sort_values("started_at"), weather_path)
    if len(frame) < 100:
        raise ValueError("At least 100 valid rows are needed for AutoML comparison")

    # This mirrors train_and_compare exactly. The final holdout is deliberately
    # not returned to FLAML and therefore cannot influence AutoML selection.
    development, final_holdout = temporal_split(frame)
    selection_train, selection_validation = temporal_split(development)
    selection_train, route_history = fit_transform_history(selection_train, TARGET)
    selection_validation = apply_history(selection_validation, route_history)
    x_train = make_features(selection_train)
    x_validation = make_features(selection_validation)
    for column in CATEGORICAL_FEATURES:
        x_train[column] = x_train[column].astype("category")
        x_validation[column] = x_validation[column].astype("category")

    split = {
        "sample_rows": int(len(frame)),
        "train_rows": int(len(selection_train)),
        "validation_rows": int(len(selection_validation)),
        "validation_start": str(selection_validation["started_at"].min()),
        "validation_end": str(selection_validation["started_at"].max()),
        "reserved_final_holdout_rows": int(len(final_holdout)),
        "reserved_final_holdout_start": str(final_holdout["started_at"].min()),
        "reserved_final_holdout_end": str(final_holdout["started_at"].max()),
    }
    return (
        x_train,
        selection_train[TARGET],
        x_validation,
        selection_validation[TARGET],
        split,
    )


def run_automl(
    data_path: Path,
    seconds: int = 1_800,
    max_rows: int | None = 1_000_000,
    estimators: Sequence[str] = DEFAULT_ESTIMATORS,
    weather_path: Path = WEATHER_PATH,
    metrics_path: Path = REPORT_DIR / "automl_metrics.csv",
    result_path: Path = REPORT_DIR / "automl_result.json",
    n_jobs: int = 1,
    trial_seconds: int = 180,
) -> dict:
    """Tune multiple learners against the manual comparison's validation window.

    FLAML receives an explicit chronological validation set. It never sees the
    final holdout used for the project's one-time final model evaluation.
    """
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise RuntimeError("Install the project requirements to enable FLAML AutoML") from exc

    estimator_list = list(dict.fromkeys(estimators))
    if not estimator_list:
        raise ValueError("At least one FLAML estimator is required")
    x_train, y_train, x_validation, y_validation, split = _comparison_data(
        data_path, max_rows, weather_path
    )

    automl = AutoML()
    automl.fit(
        X_train=x_train,
        y_train=y_train,
        X_val=x_validation,
        y_val=y_validation,
        task="regression",
        metric="mae",
        time_budget=seconds,
        train_time_limit=trial_seconds,
        estimator_list=estimator_list,
        n_jobs=n_jobs,
        eval_method="holdout",
        retrain_full=False,
        seed=42,
        verbose=2,
    )
    if not automl.best_estimator:
        raise RuntimeError("FLAML did not finish a valid trial within the time budget")

    prediction = np.clip(automl.predict(x_validation), 0, MAX_DURATION_MINUTES)
    winner_metrics = regression_metrics(y_validation, prediction)
    configurations = automl.best_config_per_estimator
    losses = automl.best_loss_per_estimator
    rows = []
    for estimator in estimator_list:
        config = configurations.get(estimator)
        loss = losses.get(estimator)
        tried = config is not None and loss is not None and np.isfinite(loss)
        rows.append(
            {
                "model": f"flaml_{estimator}",
                "mae_minutes": float(loss) if tried else np.nan,
                "tried": bool(tried),
                "selected": estimator == automl.best_estimator,
                "train_rows": split["train_rows"],
                "test_rows": split["validation_rows"],
                "test_start": split["validation_start"],
                "test_end": split["validation_end"],
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["tried", "mae_minutes"], ascending=[False, True], na_position="last"
    )

    result = {
        "estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "selection_metrics": winner_metrics,
        "best_validation_mae_per_estimator": {
            name: (float(losses[name]) if configurations.get(name) is not None else None)
            for name in estimator_list
        },
        "best_config_per_estimator": configurations,
        "estimators_requested": estimator_list,
        "estimators_not_tried": [
            name for name in estimator_list if configurations.get(name) is None
        ],
        "runtime_exclusions": {
            "lgbm": (
                "LightGBM 4.7.0 is excluded by default because its native Windows wheel "
                "raises an access violation under the project's Python 3.13 runtime."
            )
        },
        "time_budget_seconds": seconds,
        "max_trial_seconds": trial_seconds,
        **split,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"Wrote {metrics_path} and {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR / "trips.parquet")
    parser.add_argument("--seconds", type=int, default=1_800)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_000_000,
        help="maximum rows sampled evenly over time; 0 means all rows",
    )
    parser.add_argument(
        "--estimators",
        default=",".join(DEFAULT_ESTIMATORS),
        help="comma-separated FLAML estimator names",
    )
    parser.add_argument("--weather", type=Path, default=WEATHER_PATH)
    parser.add_argument("--metrics", type=Path, default=REPORT_DIR / "automl_metrics.csv")
    parser.add_argument("--result", type=Path, default=REPORT_DIR / "automl_result.json")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--trial-seconds",
        type=int,
        default=180,
        help="maximum training time for one configuration",
    )
    args = parser.parse_args()
    run_automl(
        args.data,
        seconds=args.seconds,
        max_rows=args.max_rows or None,
        estimators=[name.strip() for name in args.estimators.split(",") if name.strip()],
        weather_path=args.weather,
        metrics_path=args.metrics,
        result_path=args.result,
        n_jobs=args.n_jobs,
        trial_seconds=args.trial_seconds,
    )


if __name__ == "__main__":
    main()
