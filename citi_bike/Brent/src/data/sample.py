"""Memory-efficient sampling helpers for exploratory notebooks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def systematic_parquet_sample(
    path: Path, max_rows: int = 300_000, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read at most ``max_rows`` spread evenly over a Parquet file's row order."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    if total_rows <= max_rows:
        return pd.read_parquet(path, columns=columns)

    step = math.ceil(total_rows / max_rows)
    batches: list[pa.RecordBatch] = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        first = (-offset) % step
        indices = np.arange(first, len(batch), step, dtype=np.int64)
        if len(indices):
            batches.append(batch.take(pa.array(indices)))
        offset += len(batch)
    return pa.Table.from_batches(batches).to_pandas()
