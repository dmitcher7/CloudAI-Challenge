"""SageMaker entrypoint; artifacts are copied to SM_MODEL_DIR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.models.train import train_and_compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    train_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    data_candidates = [path for path in train_dir.glob("*.parquet") if "weather" not in path.name]
    weather_candidates = [path for path in train_dir.glob("*.parquet") if "weather" in path.name]
    if not data_candidates:
        raise FileNotFoundError(f"No Parquet input in {train_dir}")
    train_and_compare(
        data_candidates[0],
        model_path=model_dir / "duration_model.joblib",
        metrics_path=model_dir / "metrics.csv",
        max_rows=args.max_rows,
        weather_path=weather_candidates[0] if weather_candidates else train_dir / "missing-weather.parquet",
    )


if __name__ == "__main__":
    main()
