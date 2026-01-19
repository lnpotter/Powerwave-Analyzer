from __future__ import annotations

import io
from typing import IO, Union

import numpy as np
import pandas as pd


FileInput = Union[str, io.IOBase, IO[str], IO[bytes]]


def load_waveform_csv(
    filepath: FileInput,
    time_column: str = "time",
    signal_column: str = "voltage",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a time-domain waveform from a CSV file (or file-like object).

    The function reads only `time_column` and `signal_column`, converts both to
    float64, and returns NumPy arrays.

    Args:
        filepath: Path-like or a file-like object (e.g., BytesIO/StringIO).
        time_column: Column name for time values (seconds).
        signal_column: Column name for the waveform (voltage/current).

    Returns:
        (time, signal) as NumPy float64 arrays.

    Raises:
        ValueError: If columns are missing or values cannot be parsed as numbers.
    """
    try:
        df = pd.read_csv(filepath, usecols=[time_column, signal_column])
    except ValueError as exc:
        msg = str(exc)
        if "Usecols do not match columns" in msg:
            cols = list(pd.read_csv(filepath, nrows=0).columns)
            raise ValueError(
                f"CSV is missing required columns. "
                f"Required: '{time_column}', '{signal_column}'. Found: {cols}"
            ) from exc
        raise

    time = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    signal = pd.to_numeric(df[signal_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    bad = ~(np.isfinite(time) & np.isfinite(signal))
    if bad.any():
        bad_count = int(bad.sum())
        raise ValueError(f"Non-numeric or invalid values found in CSV ({bad_count} rows).")

    return time, signal