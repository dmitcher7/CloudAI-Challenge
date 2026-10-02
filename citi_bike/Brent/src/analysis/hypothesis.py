"""Pre-registered member/casual duration comparison with effect sizes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def cliffs_delta(x: np.ndarray, y: np.ndarray, max_pairs: int = 2_000_000, seed: int = 42) -> float:
    """Estimate P(X>Y)-P(X<Y), subsampling pairs to control memory."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pairs = min(max_pairs, max(len(x), len(y), 1))
    xi = rng.choice(x, size=pairs, replace=True)
    yi = rng.choice(y, size=pairs, replace=True)
    return float(np.mean(xi > yi) - np.mean(xi < yi))


def bootstrap_median_difference(
    casual: np.ndarray,
    member: np.ndarray,
    repetitions: int = 2_000,
    sample_size: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI for median(casual) - median(member)."""
    rng = np.random.default_rng(seed)
    size_c = min(sample_size, len(casual))
    size_m = min(sample_size, len(member))
    differences = np.empty(repetitions)
    for index in range(repetitions):
        c = rng.choice(casual, size=size_c, replace=True)
        m = rng.choice(member, size=size_m, replace=True)
        differences[index] = np.median(c) - np.median(m)
    return tuple(float(value) for value in np.quantile(differences, [0.025, 0.975]))


def test_member_difference(frame: pd.DataFrame, max_test_rows: int = 200_000) -> dict:
    """Run the planned one-sided test and practical-effect estimates."""
    casual_all = frame.loc[frame["member_casual"] == "casual", "duration_minutes"].dropna().to_numpy()
    member_all = frame.loc[frame["member_casual"] == "member", "duration_minutes"].dropna().to_numpy()
    if not len(casual_all) or not len(member_all):
        raise ValueError("Both casual and member observations are required")

    rng = np.random.default_rng(42)
    casual_test = rng.choice(casual_all, min(len(casual_all), max_test_rows), replace=False)
    member_test = rng.choice(member_all, min(len(member_all), max_test_rows), replace=False)
    statistic, p_value = mannwhitneyu(casual_test, member_test, alternative="greater")
    ci_low, ci_high = bootstrap_median_difference(casual_all, member_all)
    p_value_float = float(p_value)
    return {
        "hypothesis": "casual duration tends to be greater than member duration",
        "n_casual": int(len(casual_all)),
        "n_member": int(len(member_all)),
        "median_casual": float(np.median(casual_all)),
        "median_member": float(np.median(member_all)),
        "median_difference": float(np.median(casual_all) - np.median(member_all)),
        "median_difference_ci95": [ci_low, ci_high],
        "mann_whitney_u": float(statistic),
        "one_sided_p_value": p_value_float,
        "p_value_interpretation": (
            "below floating-point precision (reported as 0.0)"
            if p_value_float == 0.0
            else "reported directly"
        ),
        "cliffs_delta_casual_minus_member": cliffs_delta(casual_all, member_all),
        "note": "p-value uses capped deterministic samples; effect sizes use all rows or pair subsampling",
    }
