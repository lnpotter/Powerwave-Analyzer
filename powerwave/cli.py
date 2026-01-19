from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from powerwave.io import load_waveform_csv
from powerwave.metrics import compute_crest_factor, compute_rms, compute_thd
from powerwave.plots import save_all_plots
from powerwave.signal_processing import compute_fft_spectrum
from powerwave.i18n import get_string, LanguageCode


def analyze_file(
    filepath: str,
    signal_column: str,
    sampling_rate: float,
    fundamental_freq: float,
    time_column: str = "time",
    plot_max_freq: float = 2000.0,
    save_report_path: Optional[str] = None,
    lang: LanguageCode = "en",
) -> None:
    """
    Analyze a waveform CSV file and print metrics (RMS, crest factor, THD).

    This routine is non-interactive: it prints results to stdout and optionally
    saves a report image with plots.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be > 0.")
    if fundamental_freq <= 0:
        raise ValueError("fundamental_freq must be > 0.")

    def msg(key: str, **kwargs) -> None:
        print(get_string(lang, key, **kwargs))

    time, signal = load_waveform_csv(
        filepath=filepath,
        time_column=time_column,
        signal_column=signal_column,
    )
    signal_np = np.asarray(signal, dtype=np.float64)

    msg("cli_header")
    msg("cli_file", file=filepath)
    msg("cli_samples", samples=signal_np.size)
    msg("cli_sampling_rate", fs=float(sampling_rate))
    msg("cli_fundamental", f0=float(fundamental_freq))

    rms_value = compute_rms(signal_np)
    crest = compute_crest_factor(signal_np)
    msg("cli_rms", rms=float(rms_value))
    msg("cli_crest", crest=float(crest))

    freqs, magnitude = compute_fft_spectrum(
        signal_np,
        sampling_rate=float(sampling_rate),
        window=True,
    )

    thd_percent, harmonic_orders, harmonic_magnitudes = compute_thd(
        freqs=freqs,
        magnitude=magnitude,
        fundamental_freq=float(fundamental_freq),
        max_harmonic_order=40,
        search_fraction=0.05,
    )
    msg("cli_thd", thd=float(thd_percent))

    if save_report_path:
        save_all_plots(
            filepath=save_report_path,
            time=time,
            signal=signal_np,
            freqs=freqs,
            magnitude=magnitude,
            harmonic_orders=harmonic_orders,
            harmonic_magnitudes=harmonic_magnitudes,
            plot_max_freq=float(plot_max_freq),
            lang=lang,
        )
        msg("cli_saved_report", path=save_report_path)
    else:
        msg("cli_no_report")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="PowerWave Analyzer - offline waveform analysis (FFT, THD, harmonics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file containing waveform.")
    parser.add_argument("--column", type=str, required=True, help="Name of signal column in CSV (e.g., voltage).")
    parser.add_argument("--sampling-rate", type=float, required=True, help="Sampling rate in Hz.")
    parser.add_argument("--fundamental", type=float, required=True, help="Fundamental frequency in Hz (e.g., 50/60).")
    parser.add_argument("--time-column", type=str, default="time", help="Name of time column in CSV.")
    parser.add_argument("--plot-max-freq", type=float, default=2000.0, help="Max frequency in spectrum plot (Hz).")
    parser.add_argument("--save-report", type=str, default=None, help="Path to save a combined report image.")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "pt"], help="Language for CLI messages.")
    return parser


def main() -> None:
    """
    CLI entrypoint.
    """
    args = build_arg_parser().parse_args()
    lang: LanguageCode = "pt" if args.lang == "pt" else "en"

    analyze_file(
        filepath=args.file,
        signal_column=args.column,
        sampling_rate=float(args.sampling_rate),
        fundamental_freq=float(args.fundamental),
        time_column=args.time_column,
        plot_max_freq=float(args.plot_max_freq),
        save_report_path=args.save_report,
        lang=lang,
    )


if __name__ == "__main__":
    main()
