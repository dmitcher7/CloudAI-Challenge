"""Load a trained bundle and predict duration for start-time inputs."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MAX_DURATION_MINUTES
from src.features.build import make_features
from src.features.history import apply_history


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    required = {"model", "model_name", "features", "metrics"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid model bundle at {path}")
    return bundle


def features_for_bundle(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    enriched = apply_history(frame, bundle.get("route_history"))
    return make_features(enriched)


def predict_rows(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    features = features_for_bundle(bundle, frame)
    return np.clip(bundle["model"].predict(features), 0, MAX_DURATION_MINUTES)
