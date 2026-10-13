import json

from scripts.make_demo_data import make_demo_data
from src.models.automl import run_automl


def test_automl_uses_explicit_nested_time_split(tmp_path):
    data_path = tmp_path / "demo.parquet"
    metrics_path = tmp_path / "automl_metrics.csv"
    result_path = tmp_path / "automl_result.json"
    make_demo_data(data_path, rows=500)

    result = run_automl(
        data_path,
        seconds=2,
        max_rows=500,
        estimators=["histgb"],
        metrics_path=metrics_path,
        result_path=result_path,
        n_jobs=1,
    )

    assert result["sample_rows"] == 500
    assert result["train_rows"] == 320
    assert result["validation_rows"] == 80
    assert result["reserved_final_holdout_rows"] == 100
    assert result["estimators_not_tried"] == []
    assert result["estimator"] == "histgb"
    assert metrics_path.exists()
    assert json.loads(result_path.read_text())["estimator"] == "histgb"
