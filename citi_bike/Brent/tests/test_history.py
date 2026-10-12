import pandas as pd

from src.features.history import apply_history, fit_transform_history


def test_history_features_use_only_prior_rows():
    frame = pd.DataFrame(
        {
            "start_station_id": ["A", "A", "A"],
            "end_station_id": ["B", "B", "B"],
            "duration_minutes": [10.0, 20.0, 999.0],
        }
    )
    transformed, _ = fit_transform_history(frame, "duration_minutes")
    assert transformed.loc[0, "route_typical_minutes"] == 10.0
    assert transformed.loc[1, "route_typical_minutes"] == 10.0


def test_future_mapping_falls_back_safely_for_unseen_route():
    train = pd.DataFrame(
        {
            "start_station_id": ["A", "A", "C"],
            "end_station_id": ["B", "B", "D"],
            "duration_minutes": [10.0, 14.0, 20.0],
        }
    )
    _, history = fit_transform_history(train, "duration_minutes")
    future = pd.DataFrame({"start_station_id": ["X"], "end_station_id": ["Y"]})
    result = apply_history(future, history)
    assert result.loc[0, "route_history_log_count"] == 0.0
    assert result.loc[0, "route_typical_minutes"] == history["global_mean"]
