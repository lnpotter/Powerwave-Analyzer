import argparse

import numpy as np

from powerwave.io import load_waveform_csv
from powerwave.signal_processing import compute_fft_spectrum
from powerwave.metrics import compute_rms, compute_thd
from powerwave.plots import save_all_plots


def analyze_file(
    filepath: str,
    signal_column: str,
    sampling_rate: float,
    fundamental_freq: float,
    time_column: str = "time",
    plot_max_freq: float = 2000.0,
    save_report_path: str | None = None,
):
    """
    High-level analysis routine for a single CSV file.

    This version does not open interactive windows; it only prints
    numerical results and optionally saves a report image.
    """
    time, signal = load_waveform_csv(
        filepath=filepath,
        time_column=time_column,
        signal_column=signal_column,
    )

    signal_np = np.asarray(signal, dtype=float)

    print("=== PowerWave Analyzer ===")
    print(f"File: {filepath}")
    print(f"Samples: {signal_np.size}")
    print(f"Sampling rate: {sampling_rate:.1f} Hz")
    print(f"Fundamental frequency (assumed): {fundamental_freq:.1f} Hz")

    rms_value = compute_rms(signal_np)
    print(f"RMS value: {rms_value:.4f}")

    freqs, magnitude = compute_fft_spectrum(
        signal_np,
        sampling_rate=sampling_rate,
        window=True,
    )

    thd_percent, harmonic_orders, harmonic_magnitudes = compute_thd(
        freqs=freqs,
        magnitude=magnitude,
        fundamental_freq=fundamental_freq,
        max_harmonic_order=40,
        search_fraction=0.05,
    )

    print(f"THD: {thd_percent:.2f} %")

    # Save combined report image (required in this non-interactive mode)
    if save_report_path is not None:
        save_all_plots(
            filepath=save_report_path,
            time=time,
            signal=signal_np,
            freqs=freqs,
            magnitude=magnitude,
            harmonic_orders=harmonic_orders,
            harmonic_magnitudes=harmonic_magnitudes,
            plot_max_freq=plot_max_freq,
        )
        print(f"Saved report image to: {save_report_path}")
    else:
        print("No --save-report path provided; no image saved.")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for CLI.
    """
    parser = argparse.ArgumentParser(
        description="PowerWave Analyzer - offline waveform analysis (FFT, THD, harmonics)."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to CSV file containing waveform.",
    )
    parser.add_argument(
        "--column",
        type=str,
        required=True,
        help="Name of signal column in CSV (e.g., voltage, current).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        required=True,
        help="Sampling rate in Hz.",
    )
    parser.add_argument(
        "--fundamental",
        type=float,
        required=True,
        help="Fundamental frequency in Hz (e.g., 50 or 60).",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="time",
        help="Name of time column in CSV (default: time).",
    )
    parser.add_argument(
        "--plot-max-freq",
        type=float,
        default=2000.0,
        help="Max frequency to show in spectrum plot (Hz).",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Path to save a combined report image (e.g., plots/report.png).",
    )
    return parser


def main():
    """
    Entry point for the CLI.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    analyze_file(
        filepath=args.file,
        signal_column=args.column,
        sampling_rate=args.sampling_rate,
        fundamental_freq=args.fundamental,
        time_column=args.time_column,
        plot_max_freq=args.plot_max_freq,
        save_report_path=args.save_report,
    )


if __name__ == "__main__":
    main()