from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_rms(signal: np.ndarray) -> float:
    """
    Compute RMS (root-mean-square) of a 1D signal.

    Returns 0.0 for empty input.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    return float(np.sqrt(np.dot(x, x) / n))


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Compute crest factor = peak / RMS.

    Returns 0.0 when RMS is zero or signal is empty.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return 0.0

    rms = compute_rms(x)
    if rms == 0.0:
        return 0.0

    peak = float(np.max(np.abs(x)))
    return peak / rms


def _peak_in_band(freqs: np.ndarray, mag: np.ndarray, f0: float, frac: float) -> tuple[float, float]:
    """
    Find the maximum magnitude peak within a +/- (frac * f0) band around f0.

    Returns:
        (f_peak, mag_peak)

    Raises:
        ValueError if the band contains no bins.
    """
    df = abs(f0) * float(frac)
    lo, hi = (f0 - df), (f0 + df)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        raise ValueError(f"No spectrum bins found near {f0} Hz.")

    idx = np.flatnonzero(mask)
    j = idx[np.argmax(mag[idx])]
    return float(freqs[j]), float(mag[j])


def compute_harmonics(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    fundamental_freq: float,
    max_harmonic_order: int = 40,
    search_fraction: float = 0.05,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fundamental and harmonic peaks from a magnitude spectrum.

    Args:
        freqs: Frequency bins (Hz).
        magnitude: Spectrum magnitude (linear units, not dB).
        fundamental_freq: Expected fundamental frequency (Hz).
        max_harmonic_order: Highest harmonic order to analyze (>= 2).
        search_fraction: Search band half-width as a fraction of expected frequency.

    Returns:
        f1_detected: Detected fundamental frequency (Hz).
        v1: Fundamental magnitude.
        harmonic_orders: Array [2..max_harmonic_order].
        harmonic_freqs: Detected harmonic peak frequencies (Hz), 0 when missing.
        harmonic_magnitudes: Detected harmonic magnitudes, 0 when missing.
        harmonic_percent: Harmonic magnitudes as percent of v1 (0 when v1 == 0).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    mag = np.asarray(magnitude, dtype=np.float64)

    if freqs.size == 0 or mag.size == 0:
        raise ValueError("Empty spectrum.")
    if freqs.size != mag.size:
        raise ValueError("freqs and magnitude must have the same length.")
    if fundamental_freq <= 0:
        raise ValueError("fundamental_freq must be > 0.")
    if max_harmonic_order < 2:
        raise ValueError("max_harmonic_order must be >= 2.")
    if not (0.0 < float(search_fraction) < 1.0):
        raise ValueError("search_fraction must be in (0, 1).")

    f1_detected, v1 = _peak_in_band(freqs, mag, float(fundamental_freq), float(search_fraction))

    orders = np.arange(2, max_harmonic_order + 1, dtype=np.int32)
    h_freqs = np.zeros(orders.size, dtype=np.float64)
    h_mags = np.zeros(orders.size, dtype=np.float64)

    for i, h in enumerate(orders):
        fh = float(h) * f1_detected
        try:
            fpk, mpk = _peak_in_band(freqs, mag, fh, float(search_fraction))
        except ValueError:
            continue
        h_freqs[i] = fpk
        h_mags[i] = mpk

    h_percent = (100.0 * (h_mags / v1)) if v1 > 0 else np.zeros_like(h_mags)

    return f1_detected, v1, orders, h_freqs, h_mags, h_percent


def compute_thd(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    fundamental_freq: float,
    max_harmonic_order: int = 40,
    search_fraction: float = 0.05,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute total harmonic distortion (THD_F) from a magnitude spectrum.

    THD_F = sqrt(sum_{h=2..H} Vh^2) / V1
    """
    _f1, v1, orders, _hf, hm, _hp = compute_harmonics(
        freqs=freqs,
        magnitude=magnitude,
        fundamental_freq=fundamental_freq,
        max_harmonic_order=max_harmonic_order,
        search_fraction=search_fraction,
    )

    if v1 <= 0:
        return 0.0, orders, hm

    thd = float(np.sqrt(np.dot(hm, hm)) / v1)
    return thd * 100.0, orders, hm
