from __future__ import annotations

import numpy as np

from .i18n import get_string, LanguageCode

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover
    go = None
    make_subplots = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _robust_ylim_db(mag_db: np.ndarray) -> tuple[float, float]:
    """
    Compute robust y-limits for a dB magnitude curve using percentiles.

    Returns:
        (ymin, ymax)
    """
    x = np.asarray(mag_db, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -60.0, 10.0

    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 99.5))
    if (hi - lo) < 10.0:
        hi = lo + 10.0
    return lo - 6.0, hi + 3.0


def _scatter_cls(n_points: int):
    """
    Choose a Plotly scatter class depending on point count.

    Uses Scattergl for large series when available.
    """
    if go is None:
        return None
    return go.Scattergl if n_points >= 5000 else go.Scatter


def build_interactive_report_figure(
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
):
    """
    Build a clean interactive Plotly report figure (3 rows).

    Rows:
        1) Time-domain waveform
        2) Magnitude spectrum (dB)
        3) Harmonic magnitudes (stem-like lines + markers)
    """
    if go is None or make_subplots is None:
        raise ImportError(f"Plotly is not available: {_IMPORT_ERROR}")

    s = lambda key, **kw: get_string(lang, key, **kw)

    t = np.asarray(time, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    m = np.asarray(magnitude, dtype=np.float64)

    h_ord = np.asarray(harmonic_orders)
    h_mag = np.asarray(harmonic_magnitudes, dtype=np.float64)
    h_pct = None if harmonic_percent is None else np.asarray(harmonic_percent, dtype=np.float64)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        row_heights=[0.34, 0.33, 0.33],
    )

    # --- Row 1: Time-domain ---
    ScatterT = _scatter_cls(t.size)
    fig.add_trace(
        ScatterT(
            x=t,
            y=y,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name=s("plot_legend_signal"),
            hovertemplate="t=%{x:.6f}s<br>y=%{y:.6g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=s("plot_xlabel_time_s"), title_standoff=14, row=1, col=1)
    fig.update_yaxes(title_text=s("plot_ylabel_signal"), title_standoff=14, row=1, col=1)

    # --- Row 2: Spectrum (dB) ---
    if f.size and m.size:
        max_f = float(plot_max_freq)
        mask = (f >= 0.0) & (f <= max_f)
        fplot = f[mask] if mask.any() else f
        mplot = m[mask] if mask.any() else m

        mag_db = 20.0 * np.log10(np.maximum(mplot, 1e-12))
        ymin, ymax = _robust_ylim_db(mag_db)

        ScatterF = _scatter_cls(fplot.size)
        fig.add_trace(
            ScatterF(
                x=fplot,
                y=mag_db,
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
                name=s("plot_legend_spectrum"),
                hovertemplate="f=%{x:.3f}Hz<br>mag=%{y:.2f} dB<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[ymin, ymax], row=2, col=1)

        if fundamental_freq_detected is not None and np.isfinite(fundamental_freq_detected):
            f0 = float(fundamental_freq_detected)
            fig.add_vline(x=f0, line_width=2, line_color="#d62728", opacity=0.9, row=2, col=1)
            fig.add_annotation(
                x=f0,
                y=ymax,
                xref="x2",
                yref="y2",
                text=f"f0 = {f0:.2f} Hz",
                showarrow=False,
                yanchor="top",
                xanchor="left",
                font=dict(color="#d62728", size=12),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#d62728",
                borderwidth=1,
            )

    fig.update_xaxes(title_text=s("plot_xlabel_freq_hz"), title_standoff=14, row=2, col=1)
    fig.update_yaxes(title_text=s("plot_ylabel_mag_db"), title_standoff=14, row=2, col=1)

    # --- Row 3: Harmonics (stem-like with one line trace) ---
    if h_ord.size and h_mag.size:
        xh = h_ord.astype(np.float64, copy=False)
        yh = h_mag.astype(np.float64, copy=False)

        valid = np.isfinite(xh) & np.isfinite(yh) & (yh > 0)
        xh = xh[valid]
        yh = yh[valid]
        hp = h_pct[valid] if (h_pct is not None and h_pct.size == h_mag.size) else None

        if xh.size:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for xi, yi in zip(xh, yh):
                xs.extend([float(xi), float(xi), None])
                ys.extend([0.0, float(yi), None])

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#2ca02c", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

            custom = None if hp is None else hp.reshape(-1, 1)
            fig.add_trace(
                go.Scatter(
                    x=xh,
                    y=yh,
                    mode="markers",
                    marker=dict(color="#2ca02c", size=9),
                    name=s("plot_legend_harmonics"),
                    customdata=custom,
                    hovertemplate=(
                        "h=%{x:.0f}<br>mag=%{y:.6g}"
                        + ("<br>%fund=%{customdata[0]:.2f}%" if custom is not None else "")
                        + "<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

            if hp is not None and hp.size:
                idx = np.argsort(-np.nan_to_num(hp, nan=-1e18))[: min(5, hp.size)]
                for i in idx:
                    fig.add_annotation(
                        x=float(xh[i]),
                        y=float(yh[i]),
                        xref="x3",
                        yref="y3",
                        text=f"{hp[i]:.1f}%",
                        showarrow=False,
                        yanchor="bottom",
                        font=dict(color="#2ca02c", size=12),
                    )

            fig.update_xaxes(range=[0.5, float(np.max(xh)) + 0.5], row=3, col=1)

    fig.update_xaxes(title_text=s("plot_xlabel_order"), title_standoff=14, row=3, col=1)
    fig.update_yaxes(title_text=s("plot_ylabel_mag"), title_standoff=14, row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        height=760,
        margin=dict(l=70, r=25, t=55, b=55),
        font=dict(family="Arial, sans-serif", size=12, color="#222222"),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    for r in (1, 2, 3):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", row=r, col=1)

    return fig
