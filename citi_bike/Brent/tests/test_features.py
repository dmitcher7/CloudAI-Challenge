import pandas as pd

from src.features.build import make_features


def test_post_trip_information_cannot_change_features():
    base = pd.DataFrame(
        [
            {
                "started_at": "2024-06-15 09:30",
                "ended_at": "2024-06-15 09:40",
                "rideable_type": "classic_bike",
                "member_casual": "member",
                "start_station_id": "A",
                "start_lat": 40.72,
                "start_lng": -73.99,
                "end_station_id": "B",
            }
        ]
    )
    changed_future = base.assign(ended_at="2024-06-16", end_station_id="Z")
    pd.testing.assert_frame_equal(make_features(base), make_features(changed_future))


def test_cyclic_features_repeat_after_full_period():
    rows = pd.DataFrame(
        {
            "started_at": ["2024-01-01 08:00", "2024-01-08 08:00"],
            "rideable_type": ["classic_bike"] * 2,
            "member_casual": ["member"] * 2,
            "start_station_id": ["A"] * 2,
            "start_lat": [40.72] * 2,
            "start_lng": [-73.99] * 2,
        }
    )
    features = make_features(rows)
    assert features.loc[0, "start_hour_sin"] == features.loc[1, "start_hour_sin"]
    assert features.loc[0, "weekday_cos"] == features.loc[1, "weekday_cos"]

