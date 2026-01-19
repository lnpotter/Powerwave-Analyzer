from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .i18n import get_string, LanguageCode


_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": "#000000",
    "grid.alpha": 0.12,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
}


def _as_f64(x: np.ndarray) -> np.ndarray:
    """Return `x` as a float64 NumPy array."""
    return np.asarray(x, dtype=np.float64)


def _config_axes(ax: plt.Axes) -> None:
    """Apply consistent axes configuration (minor ticks + soft grid)."""
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.12)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.08)


def _robust_ylim_db(mag_db: np.ndarray) -> tuple[float, float]:
    """
    Compute robust y-limits for a dB magnitude curve using percentiles.

    Returns:
        (ymin, ymax)
    """
    x = _as_f64(mag_db)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -60.0, 10.0

    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 99.5))
    if (hi - lo) < 10.0:
        hi = lo + 10.0
    return lo - 6.0, hi + 3.0


def build_report_figure(
    time: np.ndarray,
    signal: np.ndarray,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    harmonic_orders: np.ndarray,
    harmonic_magnitudes: np.ndarray,
    plot_max_freq: float = 2000.0,
    lang: LanguageCode = "en",
    fundamental_freq_detected: float | None = None,
    harmonic_percent: np.ndarray | None = None,
    title_suffix: str | None = None,
) -> plt.Figure:
    """
    Build a static Matplotlib report figure (3 rows): time, spectrum (dB), harmonics.

    Args:
        time, signal: Time-domain waveform arrays.
        freqs, magnitude: FFT spectrum arrays (linear magnitude).
        harmonic_orders, harmonic_magnitudes: Harmonic peak results.
        plot_max_freq: Upper frequency limit for spectrum plot (Hz).
        lang: i18n language code.
        fundamental_freq_detected: Optional f0 marker (Hz) in the spectrum plot.
        harmonic_percent: Optional harmonic magnitudes as % of fundamental for annotations.
        title_suffix: Optional suffix for a global figure title.

    Returns:
        A Matplotlib Figure (caller can save/show/close).
    """
    s = lambda key, **kw: get_string(lang, key, **kw)

    t = _as_f64(time)
    y = _as_f64(signal)
    f = _as_f64(freqs)
    m = _as_f64(magnitude)
    h_ord = np.asarray(harmonic_orders)
    h_mag = _as_f64(harmonic_magnitudes)
    h_pct = None if harmonic_percent is None else _as_f64(harmonic_percent)

    with mpl.rc_context(_STYLE):  # temporary style without global side effects [web:106]
        fig, (ax_time, ax_spec, ax_harm) = plt.subplots(
            3, 1, figsize=(8.4, 6.0), sharex=False, constrained_layout=True
        )

        # --- Time domain
        ax_time.plot(t, y, color="#1f77b4", linewidth=1.2)
        ax_time.set_title(s("plot_title_time"), loc="left")
        ax_time.set_xlabel(s("plot_xlabel_time_s"))
        ax_time.set_ylabel(s("plot_ylabel_signal"))
        _config_axes(ax_time)
        if t.size > 1 and np.isfinite(t).all():
            ax_time.set_xlim(float(t.min()), float(t.max()))

        # --- Spectrum (dB)
        ax_spec.set_title(s("plot_title_spectrum"), loc="left")
        ax_spec.set_xlabel(s("plot_xlabel_freq_hz"))
        ax_spec.set_ylabel(s("plot_ylabel_mag_db"))
        _config_axes(ax_spec)

        if f.size and m.size:
            max_f = float(plot_max_freq)
            mask = (f >= 0.0) & (f <= max_f)
            fplot = f[mask] if mask.any() else f
            mplot = m[mask] if mask.any() else m

            mag_db = 20.0 * np.log10(np.maximum(mplot, 1e-12))
            ax_spec.plot(fplot, mag_db, color="#ff7f0e", linewidth=1.2)
            ymin, ymax = _robust_ylim_db(mag_db)
            ax_spec.set_ylim(ymin, ymax)

            if fundamental_freq_detected is not None and np.isfinite(fundamental_freq_detected):
                f0 = float(fundamental_freq_detected)
                ax_spec.axvline(f0, color="#d62728", linewidth=1.0, alpha=0.9)
                ax_spec.annotate(
                    f"f0 = {f0:.2f} Hz",
                    xy=(f0, ymax),
                    xytext=(6, -6),
                    textcoords="offset points",
                    va="top",
                    ha="left",
                    fontsize=9,
                    color="#d62728",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", alpha=0.9),
                )
        else:
            ax_spec.text(0.5, 0.5, "No spectrum data", transform=ax_spec.transAxes, ha="center", va="center")

        # --- Harmonics
        ax_harm.set_title(s("plot_title_harmonics"), loc="left")
        ax_harm.set_xlabel(s("plot_xlabel_order"))
        ax_harm.set_ylabel(s("plot_ylabel_mag"))
        _config_axes(ax_harm)

        if h_ord.size and h_mag.size:
            xh = h_ord.astype(np.float64, copy=False)
            yh = h_mag.astype(np.float64, copy=False)

            valid = np.isfinite(xh) & np.isfinite(yh) & (yh > 0)
            xh = xh[valid]
            yh = yh[valid]
            hp = h_pct[valid] if (h_pct is not None and h_pct.size == h_mag.size) else None

            if xh.size:
                ax_harm.vlines(xh, 0.0, yh, color="#2ca02c", linewidth=1.0, alpha=0.85)
                ax_harm.plot(xh, yh, "o", color="#2ca02c", markersize=4.5)
                ax_harm.set_xlim(0.5, float(xh.max()) + 0.5)

                if hp is not None and hp.size:
                    idx = np.argsort(-np.nan_to_num(hp, nan=-1e18))[: min(5, hp.size)]
                    for i in idx:
                        ax_harm.annotate(
                            f"{hp[i]:.1f}%",
                            xy=(float(xh[i]), float(yh[i])),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="#2ca02c",
                        )
        else:
            ax_harm.text(0.5, 0.5, "No harmonic data", transform=ax_harm.transAxes, ha="center", va="center")

        if title_suffix:
            fig.suptitle(f"PowerWave Analyzer — {title_suffix}", y=1.02, fontsize=12, fontweight="semibold")

        return fig


def save_all_plots(
    filepath: str,
    time: np.ndarray,
    signal: np.ndarray,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    harmonic_orders: np.ndarray,
    harmonic_magnitudes: np.ndarray,
    plot_max_freq: float = 2000.0,
    lang: LanguageCode = "en",
    fundamental_freq_detected: float | None = None,
    harmonic_percent: np.ndarray | None = None,
    title_suffix: Optional[str] = None,
) -> None:
    """
    Save the static report figure to disk and close it.
    """
    fig = build_report_figure(
        time=time,
        signal=signal,
        freqs=freqs,
        magnitude=magnitude,
        harmonic_orders=harmonic_orders,
        harmonic_magnitudes=harmonic_magnitudes,
        plot_max_freq=plot_max_freq,
        lang=lang,
        fundamental_freq_detected=fundamental_freq_detected,
        harmonic_percent=harmonic_percent,
        title_suffix=title_suffix,
    )
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
