from pathlib import Path

import pandas as pd

from src.data.sample import systematic_parquet_sample


def test_systematic_parquet_sample_is_bounded_and_spans_file(tmp_path: Path):
    path = tmp_path / "rows.parquet"
    pd.DataFrame({"value": range(1000), "unused": "x"}).to_parquet(path, index=False)
    sample = systematic_parquet_sample(path, max_rows=100, columns=["value"])
    assert list(sample.columns) == ["value"]
    assert len(sample) <= 100
    assert sample["value"].min() == 0
    assert sample["value"].max() >= 990


def test_systematic_parquet_sample_rejects_nonpositive_limit(tmp_path: Path):
    path = tmp_path / "rows.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    try:
        systematic_parquet_sample(path, max_rows=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
