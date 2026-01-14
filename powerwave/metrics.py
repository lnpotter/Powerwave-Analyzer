from typing import Tuple

import numpy as np

def compute_rms(signal: np.ndarray) -> float:
    """
    Compute RMS value of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain samples.

    Returns
    -------
    float
        RMS value.
    """
    signal = np.asarray(signal, dtype=float)
    return float(np.sqrt(np.mean(signal**2)))


def compute_thd(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    fundamental_freq: float,
    max_harmonic_order: int = 40,
    search_fraction: float = 0.05,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Total Harmonic Distortion (THD) of a waveform spectrum.

    THD is defined as:
        THD = sqrt( sum_{h=2}^{Hmax} (V_h^2) ) / V_1
    """
    freqs = np.asarray(freqs)
    magnitude = np.asarray(magnitude)

    f1 = fundamental_freq
    df1 = f1 * search_fraction
    mask_f1 = (freqs >= f1 - df1) & (freqs <= f1 + df1)
    if not np.any(mask_f1):
        raise ValueError("Fundamental frequency not found in spectrum.")

    idx_f1_region = np.where(mask_f1)[0]
    idx_f1 = idx_f1_region[np.argmax(magnitude[idx_f1_region])]
    v1 = float(magnitude[idx_f1])

    harmonic_orders = np.arange(2, max_harmonic_order + 1)
    harmonic_magnitudes = np.zeros_like(harmonic_orders, dtype=float)

    for i, h in enumerate(harmonic_orders):
        fh = h * f1
        dfh = fh * search_fraction
        mask_h = (freqs >= fh - dfh) & (freqs <= fh + dfh)
        if not np.any(mask_h):
            continue
        idx_h_region = np.where(mask_h)[0]
        idx_h = idx_h_region[np.argmax(magnitude[idx_h_region])]
        harmonic_magnitudes[i] = magnitude[idx_h]

    numerator = float(np.sqrt(np.sum(harmonic_magnitudes**2)))
    thd = numerator / v1 if v1 > 0 else 0.0
    thd_percent = thd * 100.0

    return thd_percent, harmonic_orders, harmonic_magnitudes