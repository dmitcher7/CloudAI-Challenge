"""Load a trained bundle and predict duration for start-time inputs."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build import make_features


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    required = {"model", "model_name", "features", "metrics"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid model bundle at {path}")
    return bundle


def predict_rows(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    features = make_features(frame)
    return np.maximum(0, bundle["model"].predict(features))

