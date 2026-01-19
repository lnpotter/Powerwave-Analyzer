from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ColumnGuess:
    """Result of a heuristic column guess."""
    name: str
    reason: str
    score: float


_TIME_NAME_KEYWORDS = ("time", "tempo", "timestamp", "t")
_SIGNAL_NAME_KEYWORDS = (
    "voltage",
    "current",
    "tensao",
    "corrente",
    "ch1",
    "ch2",
    "ch3",
    "v",
    "i",
)


def _numeric_finite(series: pd.Series, min_size: int) -> Optional[np.ndarray]:
    """
    Convert a Series to a finite float64 NumPy array.

    Returns None when conversion yields fewer than `min_size` finite samples.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    values = values[np.isfinite(values)]
    return values if values.size >= min_size else None


def _name_boost(name: str, keywords: Sequence[str], weight: float = 2.0) -> float:
    """
    Compute a name-based score boost by counting keyword matches.
    """
    lower = name.lower()
    return weight * sum(kw in lower for kw in keywords)


def _score_time_column(name: str, series: pd.Series) -> float:
    """
    Heuristic score for a potential time column.

    Signals used:
    - name keyword matches
    - (mostly) strictly increasing values
    - relatively stable step size (regular sampling)
    """
    score = _name_boost(name, _TIME_NAME_KEYWORDS)

    values = _numeric_finite(series, min_size=3)
    if values is None:
        return score

    a, b = values[:-1], values[1:]
    inc = b > a

    if inc.all():
        score += 3.0
    elif inc.mean() > 0.9:
        score += 1.5

    if values.size > 6:
        diffs = b - a
        step_mean = float(diffs.mean())
        if step_mean > 0:
            step_std = float(diffs.std())
            if (step_std / step_mean) < 0.2:
                score += 1.0

    return score


def _score_signal_column(name: str, series: pd.Series) -> float:
    """
    Heuristic score for a potential signal column (voltage/current).

    Signals used:
    - name keyword matches
    - non-constant numeric content
    - presence of both positive and negative values (typical AC waveform)
    """
    score = _name_boost(name, _SIGNAL_NAME_KEYWORDS)

    values = _numeric_finite(series, min_size=5)
    if values is None:
        return score

    if float(values.std()) > 0.0:
        score += 1.0
    if (values > 0).any() and (values < 0).any():
        score += 1.0

    return score


def guess_time_column(df: pd.DataFrame, candidates: Optional[Sequence[str]] = None) -> Optional[ColumnGuess]:
    """
    Guess the most likely time column in a DataFrame.

    Returns:
        ColumnGuess if a candidate scores > 0, otherwise None.
    """
    cols = list(candidates) if candidates is not None else list(df.columns)
    if not cols:
        return None

    best_name = None
    best_score = float("-inf")

    for name in cols:
        if name not in df.columns:
            continue
        score = _score_time_column(name, df[name])
        if score > best_score:
            best_name, best_score = name, score

    return ColumnGuess(best_name, "heuristic", best_score) if best_name and best_score > 0 else None


def guess_signal_column(df: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> Optional[ColumnGuess]:
    """
    Guess the most likely signal column (voltage/current) in a DataFrame.

    Args:
        df: Input DataFrame.
        exclude: Column names that must not be considered (e.g., the time column).

    Returns:
        ColumnGuess if a candidate scores > 0, otherwise None.
    """
    excluded = set(exclude or ())
    best_name = None
    best_score = float("-inf")

    for name in df.columns:
        if name in excluded:
            continue
        score = _score_signal_column(name, df[name])
        if score > best_score:
            best_name, best_score = name, score

    return ColumnGuess(best_name, "heuristic", best_score) if best_name and best_score > 0 else None
