from typing import Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_fft_spectrum(
    signal: np.ndarray,
    sampling_rate: float,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided magnitude spectrum of a real-valued signal.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain samples of the signal.
    sampling_rate : float
        Sampling frequency in Hz.
    window : bool
        If True, apply a Hann window before FFT to reduce spectral leakage.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins (Hz) for the single-sided spectrum.
    magnitude : np.ndarray
        Magnitude spectrum (same units as signal).
    """
    signal = np.asarray(signal, dtype=float)
    n_samples = signal.size

    if window:
        hann_window = np.hanning(n_samples)
        windowed = signal * hann_window
        coherent_gain = np.sum(hann_window) / n_samples
    else:
        windowed = signal
        coherent_gain = 1.0

    fft_values = rfft(windowed)
    freqs = rfftfreq(n_samples, d=1.0 / sampling_rate)

    magnitude = (2.0 / (n_samples * coherent_gain)) * np.abs(fft_values)

    if magnitude.size > 0:
        magnitude[0] /= 2.0
    if n_samples % 2 == 0 and magnitude.size > 1:
        magnitude[-1] /= 2.0

    return freqs, magnitude


def detect_fundamental_from_spectrum(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    expected_f0: float,
    search_bandwidth: float = 5.0,
) -> Tuple[float, float]:
    """
    Detect the fundamental frequency and its magnitude from the spectrum.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins (Hz).
    magnitude : np.ndarray
        Magnitude spectrum.
    expected_f0 : float
        Expected fundamental frequency in Hz (e.g., 50 or 60).
    search_bandwidth : float
        +/- range around expected_f0 to search (Hz).

    Returns
    -------
    f0_detected : float
        Detected fundamental frequency (Hz).
    mag_f0 : float
        Magnitude at the fundamental frequency.
    """
    freqs = np.asarray(freqs)
    magnitude = np.asarray(magnitude)

    lower = expected_f0 - search_bandwidth
    upper = expected_f0 + search_bandwidth

    mask = (freqs >= lower) & (freqs <= upper)
    if not np.any(mask):
        raise ValueError("No frequency bins found in the fundamental search region.")

    idx_region = np.where(mask)[0]
    idx_max = idx_region[np.argmax(magnitude[idx_region])]

    f0_detected = float(freqs[idx_max])
    mag_f0 = float(magnitude[idx_max])

    return f0_detected, mag_f0