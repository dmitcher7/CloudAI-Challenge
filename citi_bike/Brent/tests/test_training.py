from pathlib import Path

from scripts.make_demo_data import make_demo_data
from src.models.predict import load_bundle, predict_rows
from src.models.train import train_and_compare


def test_training_and_prediction_roundtrip(tmp_path: Path):
    data_path = tmp_path / "demo.parquet"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.csv"
    frame = make_demo_data(data_path, rows=400)
    bundle, metrics = train_and_compare(
        data_path, model_path, metrics_path, max_rows=None, quick=True
    )

    assert model_path.exists()
    assert metrics["mae_minutes"].notna().all()
    assert bundle["model_name"] in set(metrics["model"])
    assert metrics["selected"].sum() == 1
    assert metrics.loc[metrics["selected"], "model"].iloc[0] == bundle["model_name"]
    assert bundle["metrics"]["test_rows"] > 0
    assert bundle["route_history_summary"]["routes"] > 0

    loaded = load_bundle(model_path)
    prediction = predict_rows(loaded, frame.iloc[[-1]])
    assert prediction.shape == (1,)
    assert 0 <= prediction[0] <= 180
