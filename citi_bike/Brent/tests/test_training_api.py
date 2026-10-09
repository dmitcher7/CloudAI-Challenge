from pathlib import Path

from fastapi.testclient import TestClient

import app.main as web
from scripts.make_demo_data import make_demo_data
from src.models.train import train_and_compare


def test_training_and_prediction_roundtrip(tmp_path: Path, monkeypatch):
    data_path = tmp_path / "demo.parquet"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.csv"
    make_demo_data(data_path, rows=400)
    bundle, metrics = train_and_compare(
        data_path, model_path, metrics_path, max_rows=None, quick=True
    )
    assert model_path.exists()
    assert metrics["mae_minutes"].notna().all()
    assert bundle["model_name"] in set(metrics["model"])

    monkeypatch.setattr(web, "MODEL_PATH", model_path)
    web.model_bundle.cache_clear()
    client = TestClient(web.app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post(
        "/predict",
        json={
            "started_at": "2024-06-15T09:30:00",
            "rideable_type": "classic_bike",
            "member_casual": "casual",
            "start_station_id": "HB101",
            "start_lat": 40.7359,
            "start_lng": -74.0303,
        },
    )
    assert response.status_code == 200
    assert response.json()["predicted_duration_minutes"] > 0


def test_api_rejects_impossible_coordinates():
    client = TestClient(web.app)
    response = client.post(
        "/predict",
        json={
            "started_at": "2024-06-15T09:30:00",
            "rideable_type": "classic_bike",
            "member_casual": "member",
            "start_station_id": "A",
            "start_lat": 0,
            "start_lng": 0,
        },
    )
    assert response.status_code == 422

