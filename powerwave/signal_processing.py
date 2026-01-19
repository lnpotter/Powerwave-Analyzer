from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq


def estimate_sampling_rate_from_time(time: np.ndarray) -> float:
    """
    Estimate sampling rate from a time vector (seconds) using median positive dt.

    This is robust to small jitter because it uses the median of finite, positive
    time differences.

    Args:
        time: Time samples in seconds.

    Returns:
        Estimated sampling rate in Hz.

    Raises:
        ValueError: If sampling rate cannot be estimated (too few samples, not increasing,
        or invalid dt).
    """
    t = np.asarray(time, dtype=np.float64)
    if t.size < 3:
        raise ValueError("Time vector too small to estimate sampling rate.")

    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

    if diffs.size < 2:
        raise ValueError("Time vector is not strictly increasing; cannot estimate sampling rate.")

    dt = float(np.median(diffs))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Invalid dt computed from time vector.")

    fs = 1.0 / dt
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("Invalid sampling rate estimated from time vector.")

    return float(fs)


def compute_fft_spectrum(
    signal: np.ndarray,
    sampling_rate: float,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided magnitude spectrum of a real-valued signal.

    Notes:
        - Uses rFFT (real FFT), returning only non-negative frequency bins.
        - If `window=True`, applies a Hann window to reduce spectral leakage. [web:121]
        - Applies a coherent-gain correction so a sine wave's peak magnitude is preserved
          approximately under windowing.

    Args:
        signal: Time-domain samples (real-valued).
        sampling_rate: Sampling frequency in Hz (> 0).
        window: If True, apply a Hann window before FFT.

    Returns:
        freqs: Frequency bins in Hz (non-negative).
        magnitude: Single-sided magnitude spectrum (linear units).

    Raises:
        ValueError: If sampling_rate is invalid.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    fs = float(sampling_rate)
    if fs <= 0.0 or not np.isfinite(fs):
        raise ValueError("sampling_rate must be a finite value > 0.")

    if window:
        w = np.hanning(n)
        cg = float(w.mean())  # coherent gain for amplitude correction
        if cg <= 0.0 or not np.isfinite(cg):
            cg = 1.0
        x = x * w
    else:
        cg = 1.0

    X = rfft(x)
    freqs = rfftfreq(n, d=1.0 / fs)

    mag = (2.0 / (n * cg)) * np.abs(X)

    if mag.size:
        mag[0] *= 0.5  # DC should not be doubled
    if (n % 2 == 0) and (mag.size > 1):
        mag[-1] *= 0.5  # Nyquist should not be doubled for even n

    return freqs.astype(np.float64, copy=False), mag.astype(np.float64, copy=False)


def _peak_in_band(freqs: np.ndarray, magnitude: np.ndarray, center_hz: float, half_width_hz: float) -> tuple[float, float]:
    """
    Find the peak magnitude within a frequency band [center-half_width, center+half_width].

    Returns:
        (f_peak_hz, mag_peak)

    Raises:
        ValueError: If there are no bins inside the band.
    """
    f = np.asarray(freqs, dtype=np.float64)
    m = np.asarray(magnitude, dtype=np.float64)

    lo = float(center_hz) - float(half_width_hz)
    hi = float(center_hz) + float(half_width_hz)
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        raise ValueError("No frequency bins found in the requested band.")

    idx = np.flatnonzero(mask)
    j = idx[np.argmax(m[idx])]
    return float(f[j]), float(m[j])


def detect_fundamental_from_spectrum(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    expected_f0: float,
    search_bandwidth: float = 5.0,
) -> Tuple[float, float]:
    """
    Detect the fundamental frequency and its magnitude from a magnitude spectrum.

    Args:
        freqs: Frequency bins (Hz).
        magnitude: Magnitude spectrum (linear units).
        expected_f0: Expected fundamental frequency in Hz (e.g., 50 or 60).
        search_bandwidth: +/- bandwidth around expected_f0 (Hz).

    Returns:
        f0_detected: Detected fundamental frequency (Hz).
        mag_f0: Magnitude at the detected fundamental.

    Raises:
        ValueError: If inputs are invalid or no bins exist inside the search band.
    """
    if expected_f0 <= 0.0 or not np.isfinite(expected_f0):
        raise ValueError("expected_f0 must be a finite value > 0.")
    bw = float(search_bandwidth)
    if bw <= 0.0 or not np.isfinite(bw):
        raise ValueError("search_bandwidth must be a finite value > 0.")

    return _peak_in_band(freqs, magnitude, center_hz=float(expected_f0), half_width_hz=bw)
