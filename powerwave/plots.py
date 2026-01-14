from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def save_all_plots(
    filepath: str,
    time,
    signal,
    freqs,
    magnitude,
    harmonic_orders,
    harmonic_magnitudes,
    plot_max_freq: float = 2000.0,
):
    """
    Create a single figure with all plots (time-series, spectrum, harmonics)
    and save it to an image file.

    Parameters
    ----------
    filepath : str
        Output image path (e.g., 'plots/report.png').
    time : np.ndarray
        Time vector (s).
    signal : np.ndarray
        Waveform samples.
    freqs : np.ndarray
        Frequency bins (Hz).
    magnitude : np.ndarray
        Magnitude spectrum.
    harmonic_orders : np.ndarray
        Harmonic orders.
    harmonic_magnitudes : np.ndarray
        Harmonic magnitudes.
    plot_max_freq : float
        Max frequency to display in the spectrum subplot (Hz).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # Time-domain waveform
    ax_time = axes[0]
    ax_time.plot(time, signal)
    ax_time.set_title("Time-domain waveform")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)

    # Magnitude spectrum
    ax_spec = axes[1]
    ax_spec.plot(freqs, magnitude)
    ax_spec.set_title("Magnitude spectrum")
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("Magnitude")
    ax_spec.set_xlim(0, plot_max_freq)
    ax_spec.grid(True)

    # Harmonic bar chart
    ax_harm = axes[2]
    orders = np.asarray(harmonic_orders)
    mags = np.asarray(harmonic_magnitudes)
    mask = orders <= 25
    ax_harm.bar(orders[mask], mags[mask], width=0.8)
    ax_harm.set_title("Harmonic magnitudes")
    ax_harm.set_xlabel("Harmonic order")
    ax_harm.set_ylabel("Magnitude")
    ax_harm.grid(True, axis="y")

    fig.suptitle("PowerWave Analyzer report", fontsize=14)

    fig.savefig(filepath, dpi=150)
    plt.close(fig)