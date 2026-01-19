from .io import load_waveform_csv
from .signal_processing import compute_fft_spectrum, detect_fundamental_from_spectrum
from .metrics import compute_rms, compute_thd, compute_crest_factor
from .plots import save_all_plots, build_report_figure
from .i18n import get_string, LanguageCode
from .csv_detect import guess_time_column, guess_signal_column

try:
    from .plotly_plots import build_interactive_report_figure
except Exception:
    build_interactive_report_figure = None

__all__ = [
    "load_waveform_csv",
    "compute_fft_spectrum",
    "detect_fundamental_from_spectrum",
    "compute_rms",
    "compute_thd",
    "compute_crest_factor",
    "save_all_plots",
    "build_report_figure",
    "get_string",
    "LanguageCode",
    "guess_time_column",
    "guess_signal_column",
    "build_interactive_report_figure",
]
