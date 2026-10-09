"""Stable, serializable transformations shared by training and inference."""

from __future__ import annotations

import numpy as np

from src.config import MAX_DURATION_MINUTES


def inverse_log_duration(values):
    """Invert log1p safely within the documented target range."""
    return np.expm1(np.clip(values, 0, np.log1p(MAX_DURATION_MINUTES)))
