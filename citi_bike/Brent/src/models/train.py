"""Train, compare and serialize leakage-safe duration models."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor
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
from src.features.history import HISTORY_FEATURES, apply_history, fit_transform_history
from src.models.transforms import inverse_log_duration

CATEGORICAL = ["rideable_type", "member_casual", "start_station_id", "end_station_id"]
MAE_SELECTION_TOLERANCE = 0.10
NUMERIC = [
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "direct_distance_km",
    "grid_distance_km",
    "delta_lat",
    "delta_lng",
    "bearing_sin",
    "bearing_cos",
    *HISTORY_FEATURES,
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
    log_estimators: dict[str, object] = {
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
    models = {
        name: TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=inverse_log_duration,
            check_inverse=False,
        )
        for name, estimator in log_estimators.items()
    }
    models["hist_gradient_boosting_raw"] = Pipeline(
        [
            ("preprocess", _ordinal_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.06,
                    max_iter=80 if quick else 350,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    models["hist_gradient_boosting_absolute"] = Pipeline(
        [
            ("preprocess", _ordinal_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.06,
                    max_iter=60 if quick else 250,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    models["catboost"] = CatBoostRegressor(
        iterations=80 if quick else 600,
        depth=8,
        learning_rate=0.06,
        loss_function="RMSE",
        eval_metric="MAE",
        l2_leaf_reg=8.0,
        random_strength=0.5,
        cat_features=CATEGORICAL,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    return models


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
    max_rows: int | None = 1_000_000,
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

    development, test = temporal_split(frame)
    selection_train, selection_validation = temporal_split(development)
    selection_train, selection_history = fit_transform_history(selection_train, TARGET)
    selection_validation = apply_history(selection_validation, selection_history)
    x_selection_train = make_features(selection_train)
    x_selection_validation = make_features(selection_validation)
    y_selection_train = selection_train[TARGET]
    y_selection_validation = selection_validation[TARGET]
    results = []
    candidates = candidate_models(quick=quick)
    for name in list(candidates):
        model = candidates.pop(name)
        print(f"Selecting {name} on {len(selection_train):,} rows …", flush=True)
        model.fit(x_selection_train, y_selection_train)
        predictions = np.clip(
            model.predict(x_selection_validation), 0, MAX_DURATION_MINUTES
        )
        model_metrics = regression_metrics(y_selection_validation, predictions)
        results.append(
            {
                "model": name,
                **model_metrics,
                "train_rows": len(selection_train),
                "test_rows": len(selection_validation),
                "test_start": str(selection_validation["started_at"].min()),
                "test_end": str(selection_validation["started_at"].max()),
            }
        )
        del model
        gc.collect()

    metrics = pd.DataFrame(results).sort_values("mae_minutes").reset_index(drop=True)
    best_validation_mae = float(metrics["mae_minutes"].min())
    eligible = metrics.loc[
        metrics["mae_minutes"] <= best_validation_mae + MAE_SELECTION_TOLERANCE
    ]
    winner_name = str(
        eligible.sort_values(["r2", "mae_minutes"], ascending=[False, True]).iloc[0]["model"]
    )
    metrics["selected"] = metrics["model"].eq(winner_name)

    development, route_history = fit_transform_history(development, TARGET)
    test = apply_history(test, route_history)
    x_train, x_test = make_features(development), make_features(test)
    y_train, y_test = development[TARGET], test[TARGET]
    best_model = candidate_models(quick=quick)[winner_name]
    print(f"Refitting selected {winner_name} on {len(development):,} rows …", flush=True)
    best_model.fit(x_train, y_train)
    best_predictions = np.clip(best_model.predict(x_test), 0, MAX_DURATION_MINUTES)
    final_metrics = {
        "model": winner_name,
        **regression_metrics(y_test, best_predictions),
        "train_rows": len(development),
        "test_rows": len(test),
        "test_start": str(test["started_at"].min()),
        "test_end": str(test["started_at"].max()),
    }
    train_predictions = np.clip(best_model.predict(x_train), 0, MAX_DURATION_MINUTES)
    train_metrics = regression_metrics(y_train, train_predictions)
    monthly_rows = []
    test_month = pd.to_datetime(test["started_at"]).dt.to_period("M").astype(str)
    for month in sorted(test_month.unique()):
        mask = test_month == month
        if int(mask.sum()) < 2:
            continue
        monthly_rows.append(
            {
                "month": month,
                "rows": int(mask.sum()),
                **regression_metrics(y_test.loc[mask], best_predictions[mask.to_numpy()]),
            }
        )
    monthly_metrics = pd.DataFrame(monthly_rows)
    bundle = {
        "model": best_model,
        "model_name": winner_name,
        "features": list(x_train.columns),
        "target": TARGET,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_period": [
            str(development["started_at"].min()),
            str(development["started_at"].max()),
        ],
        "holdout_period": [str(test["started_at"].min()), str(test["started_at"].max())],
        "metrics": final_metrics,
        "selection_rule": (
            f"highest validation R2 among models within {MAE_SELECTION_TOLERANCE:.2f} "
            "minutes of the best validation MAE"
        ),
        "selection_metrics": metrics.loc[metrics.model == winner_name].iloc[0].to_dict(),
        "training_metrics": train_metrics,
        "monthly_holdout_metrics": monthly_rows,
        "route_history": route_history,
        "route_history_summary": {
            "routes": len(route_history["route"]),
            "start_stations": len(route_history["start"]),
            "end_stations": len(route_history["end"]),
            "smoothing": route_history["smoothing"],
        },
        "limitations": (
            "Destination is treated as planned input; weather is hourly and city-level. "
            "Events, traffic, route choice and bike availability are unavailable."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame([final_metrics]).to_csv(
        metrics_path.with_name("final_holdout_metrics.csv"), index=False
    )
    monthly_metrics.to_csv(metrics_path.with_name("metrics_by_month.csv"), index=False)
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {key: value for key, value in bundle.items() if key not in {"model", "route_history"}},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Selection validation metrics:")
    print(metrics.to_string(index=False))
    print("Final separate holdout:")
    print(pd.DataFrame([final_metrics]).to_string(index=False))
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
        default=1_000_000,
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
