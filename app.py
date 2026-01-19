from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from powerwave import (
    load_waveform_csv,
    compute_fft_spectrum,
    compute_rms,
    compute_thd,
    compute_crest_factor,
    build_report_figure,
    build_interactive_report_figure,
    get_string,
    LanguageCode,
    guess_time_column,
    guess_signal_column,
)

MAX_STEP = 5

st.set_page_config(page_title="PowerWave Analyzer", page_icon="⚡", layout="wide")


def init_state() -> None:
    """Initialize Streamlit session state defaults."""
    ss = st.session_state
    ss.setdefault("wizard_step", 1)
    ss.setdefault("uploaded_file_bytes", None)
    ss.setdefault("time_column", None)
    ss.setdefault("signal_column", None)
    ss.setdefault("sampling_rate", 10_000.0)
    ss.setdefault("fundamental_freq", 60.0)
    ss.setdefault("plot_max_freq", 2000)
    ss.setdefault("results", None)
    ss.setdefault("lang", "en")


def clamp_step(step: int) -> int:
    """Clamp wizard step between 1 and MAX_STEP."""
    return max(1, min(int(step), MAX_STEP))


def go_next() -> None:
    """Advance wizard step."""
    st.session_state.wizard_step = clamp_step(st.session_state.wizard_step + 1)


def go_prev() -> None:
    """Go back one wizard step."""
    st.session_state.wizard_step = clamp_step(st.session_state.wizard_step - 1)


def set_uploaded_bytes(file_bytes: Optional[bytes]) -> None:
    """Set uploaded bytes and reset dependent state when the file changes."""
    ss = st.session_state
    if file_bytes == ss.uploaded_file_bytes:
        return
    ss.uploaded_file_bytes = file_bytes
    ss.time_column = None
    ss.signal_column = None
    ss.results = None


@st.cache_data(show_spinner=False)
def load_csv_preview(file_bytes: bytes, nrows: int = 200) -> pd.DataFrame:
    """
    Load a small CSV preview from uploaded bytes.

    This is used only for showing a preview and listing columns.
    """
    return pd.read_csv(io.BytesIO(file_bytes), nrows=int(nrows))


def _fundamental_mag_in_band(freqs: np.ndarray, magnitude: np.ndarray, f0: float, frac: float) -> Optional[float]:
    """
    Estimate the fundamental magnitude by taking the max within +/- (frac * f0).

    Returns None if no bins exist in the band or if inputs are empty/invalid.
    """
    if freqs.size == 0 or magnitude.size == 0:
        return None
    if not np.isfinite(f0) or f0 <= 0:
        return None

    df = abs(float(f0)) * float(frac)
    mask = (freqs >= (f0 - df)) & (freqs <= (f0 + df))
    if not mask.any():
        return None

    idx = np.flatnonzero(mask)
    return float(magnitude[idx[np.argmax(magnitude[idx])]])


def run_analysis() -> None:
    """
    Run the analysis once (triggered by a button) and store results in session state.

    Results are stored as a plain dict (instead of a dataclass) to avoid type-mismatch
    issues across Streamlit reruns.
    """
    ss = st.session_state
    file_bytes: Optional[bytes] = ss.uploaded_file_bytes

    if not file_bytes or not ss.time_column or not ss.signal_column:
        return

    try:
        time, signal = load_waveform_csv(
            filepath=io.BytesIO(file_bytes),
            time_column=str(ss.time_column),
            signal_column=str(ss.signal_column),
        )
        time = np.asarray(time, dtype=np.float64)
        signal_np = np.asarray(signal, dtype=np.float64)
    except Exception as exc:
        ss.results = {"error": f"Error parsing CSV: {exc}"}
        ss.wizard_step = 5
        return

    sampling_rate = float(ss.sampling_rate)
    fundamental_freq = float(ss.fundamental_freq)
    plot_max_freq = float(ss.plot_max_freq)

    rms_value = float(compute_rms(signal_np))
    crest_factor = float(compute_crest_factor(signal_np))

    freqs, magnitude = compute_fft_spectrum(signal_np, sampling_rate=sampling_rate, window=True)

    try:
        thd_percent, harmonic_orders, harmonic_magnitudes = compute_thd(
            freqs=freqs,
            magnitude=magnitude,
            fundamental_freq=fundamental_freq,
            max_harmonic_order=40,
            search_fraction=0.05,
        )
        thd_error = None
        thd_percent_out: Optional[float] = float(thd_percent)
    except ValueError as exc:
        thd_percent_out = None
        harmonic_orders = np.array([], dtype=np.int32)
        harmonic_magnitudes = np.array([], dtype=np.float64)
        thd_error = str(exc)

    v1 = _fundamental_mag_in_band(freqs, magnitude, fundamental_freq, frac=0.05)
    harmonic_percent = None
    if v1 and v1 > 0 and harmonic_magnitudes.size:
        harmonic_percent = 100.0 * (harmonic_magnitudes / float(v1))

    # Generate a single PNG once, reuse it for display + download.
    fig = build_report_figure(
        time=time,
        signal=signal_np,
        freqs=freqs,
        magnitude=magnitude,
        harmonic_orders=harmonic_orders,
        harmonic_magnitudes=harmonic_magnitudes,
        plot_max_freq=plot_max_freq,
        lang=ss.lang,
        fundamental_freq_detected=fundamental_freq,
        harmonic_percent=harmonic_percent,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)

    ss.results = {
        "time": time,
        "signal": signal_np,
        "sampling_rate": sampling_rate,
        "fundamental_freq": fundamental_freq,
        "plot_max_freq": plot_max_freq,
        "rms_value": rms_value,
        "crest_factor": crest_factor,
        "thd_percent": thd_percent_out,
        "thd_error": thd_error,
        "freqs": freqs,
        "magnitude": magnitude,
        "harmonic_orders": harmonic_orders,
        "harmonic_magnitudes": harmonic_magnitudes,
        "harmonic_percent": harmonic_percent,
        "report_png_bytes": buf.getvalue(),
    }
    ss.wizard_step = 5


def render_nav_top(lang: LanguageCode, current_step: int, labels: dict[int, str]) -> None:
    """Render the top step indicator and a Back button when applicable."""
    st.markdown(f"**{labels[current_step]}**  ({current_step}/5)")
    if current_step > 1:
        st.button("⬅ Back" if lang == "en" else "⬅ Voltar", key="top_back", on_click=go_prev)


def render_step_1(s) -> None:
    """Step 1: analysis settings."""
    ss = st.session_state
    st.markdown("### " + s("step1_title"))

    c1, c2 = st.columns(2)
    with c1:
        ss.sampling_rate = st.number_input(
            s("sampling_rate_label"),
            min_value=100.0,
            max_value=1_000_000.0,
            value=float(ss.sampling_rate),
            step=100.0,
            help=s("sampling_rate_help"),
        )
    with c2:
        ss.fundamental_freq = st.number_input(
            s("fundamental_label"),
            min_value=1.0,
            max_value=1000.0,
            value=float(ss.fundamental_freq),
            step=1.0,
            help=s("fundamental_help"),
        )

    ss.plot_max_freq = st.slider(
        s("max_freq_label"),
        min_value=100,
        max_value=50_000,
        value=int(ss.plot_max_freq),
        step=100,
        help=s("max_freq_help"),
    )

    st.caption(s("fft_note"))
    st.markdown("---")
    st.button("Next ➜" if ss.lang == "en" else "Próximo ➜", type="primary", key="step1_next", on_click=go_next)


def render_step_2(s) -> None:
    """Step 2: upload CSV and show preview."""
    ss = st.session_state
    st.markdown("### " + s("step2_title"))

    uploaded = st.file_uploader(
        s("upload_label"),
        type=["csv"],
        help=s("upload_help"),
        key="file_uploader_widget",
    )
    if uploaded is not None:
        set_uploaded_bytes(uploaded.getvalue())

    if ss.uploaded_file_bytes:
        try:
            df_preview = load_csv_preview(ss.uploaded_file_bytes, nrows=200)
            st.subheader(s("preview_title"))
            st.dataframe(df_preview, width="stretch")
            st.caption(s("csv_detected_columns", columns=", ".join(map(str, df_preview.columns))))
        except Exception:
            st.info(s("no_csv_warning"))
    else:
        st.info(s("no_csv_warning"))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅ Back" if ss.lang == "en" else "⬅ Voltar", key="step2_back", on_click=go_prev)
    with c2:
        disabled = not bool(ss.uploaded_file_bytes)
        st.button(
            "Next ➜" if ss.lang == "en" else "Próximo ➜",
            type="primary",
            key="step2_next",
            disabled=disabled,
            on_click=None if disabled else go_next,
        )


def render_step_3(s) -> None:
    """Step 3: select time/signal columns with heuristics."""
    ss = st.session_state
    st.markdown("### " + s("step3_title"))

    if not ss.uploaded_file_bytes:
        st.warning(s("no_csv_warning"))
        st.markdown("---")
        st.button("⬅ Back" if ss.lang == "en" else "⬅ Voltar", key="step3_back", on_click=go_prev)
        return

    try:
        df_preview = load_csv_preview(ss.uploaded_file_bytes, nrows=200)
    except Exception:
        st.error(s("csv_no_columns"))
        return

    columns = list(map(str, df_preview.columns))
    if not columns:
        st.error(s("csv_no_columns"))
        return

    time_guess = guess_time_column(df_preview)
    time_default_idx = columns.index(time_guess.name) if time_guess and time_guess.name in columns else 0

    signal_guess = guess_signal_column(df_preview, exclude=[time_guess.name] if time_guess else [])
    signal_default_idx = (
        columns.index(signal_guess.name)
        if signal_guess and signal_guess.name in columns
        else (1 if len(columns) > 1 else 0)
    )

    c1, c2 = st.columns(2)
    with c1:
        ss.time_column = st.selectbox(
            s("time_column_label"),
            options=columns,
            index=time_default_idx,
            help=(f"Suggested automatically: '{time_guess.name}' (score={time_guess.score:.1f})" if time_guess else None),
            key="time_column_select",
        )
    with c2:
        ss.signal_column = st.selectbox(
            s("signal_column_label"),
            options=columns,
            index=signal_default_idx,
            help=(
                f"Suggested automatically: '{signal_guess.name}' (score={signal_guess.score:.1f})"
                if signal_guess
                else None
            ),
            key="signal_column_select",
        )

    st.markdown("---")
    c1b, c2b = st.columns(2)
    with c1b:
        st.button("⬅ Back" if ss.lang == "en" else "⬅ Voltar", key="step3_back", on_click=go_prev)
    with c2b:
        disabled = not (ss.time_column and ss.signal_column and ss.uploaded_file_bytes)
        st.button(
            "Next ➜" if ss.lang == "en" else "Próximo ➜",
            type="primary",
            key="step3_next",
            disabled=disabled,
            on_click=None if disabled else go_next,
        )


def render_step_4(s) -> None:
    """Step 4: run analysis."""
    ss = st.session_state
    st.markdown("### " + s("step4_title"))

    if not ss.uploaded_file_bytes:
        st.warning(s("no_csv_warning"))
    else:
        st.write(f"Time column: {ss.time_column}")
        st.write(f"Signal column: {ss.signal_column}")
        st.button(s("run_analysis_button"), type="primary", key="run_analysis_btn", on_click=run_analysis)

    st.markdown("---")
    st.button("⬅ Back" if ss.lang == "en" else "⬅ Voltar", key="step4_back", on_click=go_prev)


def render_step_5(s) -> None:
    """Step 5: show results and plots."""
    ss = st.session_state
    st.markdown("### " + s("step5_title"))

    res = ss.results
    if res is None:
        st.warning(
            "No results available. Please run the analysis in the previous step."
            if ss.lang == "en"
            else "Nenhum resultado disponível. Execute a análise na etapa anterior."
        )
        return

    if isinstance(res, dict) and "error" in res:
        st.error(str(res["error"]))
        return

    if not isinstance(res, dict):
        st.error("Invalid results object.")
        return

    required_keys = {"time", "signal", "sampling_rate", "fundamental_freq", "rms_value", "crest_factor", "freqs", "magnitude"}
    if not required_keys.issubset(res.keys()):
        st.error("Invalid results object.")
        return

    st.subheader(s("results_title"))

    samples = int(np.asarray(res["signal"]).size)
    c1, c2 = st.columns(2)

    with c1:
        st.metric(s("metric_samples"), f"{samples}")
        st.metric(s("metric_sampling_rate"), f"{float(res['sampling_rate']):.1f}")
        st.metric(s("metric_fundamental"), f"{float(res['fundamental_freq']):.1f}")

    with c2:
        st.metric(s("metric_rms"), f"{float(res['rms_value']):.4f}")
        st.metric(s("metric_crest"), f"{float(res['crest_factor']):.3f}")
        thd_percent = res.get("thd_percent", None)
        st.metric(s("metric_thd"), "N/A" if thd_percent is None else f"{float(thd_percent):.2f}")

    thd_error = res.get("thd_error", None)
    if thd_error:
        st.warning(s("thd_not_computed", error=str(thd_error)))

    with st.expander(s("export_json_title"), expanded=False):
        st.json(
            {
                "samples": samples,
                "sampling_rate_hz": float(res["sampling_rate"]),
                "fundamental_hz": float(res["fundamental_freq"]),
                "rms_value": float(res["rms_value"]),
                "crest_factor": float(res["crest_factor"]),
                "thd_percent": None if thd_percent is None else float(thd_percent),
            }
        )

    st.subheader(s("plots_title"))

    plotly_ok = build_interactive_report_figure is not None
    if plotly_ok:
        try:
            fig_plotly = build_interactive_report_figure(
                time=np.asarray(res["time"]),
                signal=np.asarray(res["signal"]),
                freqs=np.asarray(res["freqs"]),
                magnitude=np.asarray(res["magnitude"]),
                harmonic_orders=np.asarray(res.get("harmonic_orders", np.array([]))),
                harmonic_magnitudes=np.asarray(res.get("harmonic_magnitudes", np.array([]))),
                plot_max_freq=float(res.get("plot_max_freq", 2000.0)),
                lang=ss.lang,
                fundamental_freq_detected=float(res["fundamental_freq"]),
                harmonic_percent=res.get("harmonic_percent", None),
            )
            st.plotly_chart(fig_plotly, width="stretch", config={"displaylogo": False})
        except Exception:
            plotly_ok = False

    if not plotly_ok:
        # Fallback: show pre-rendered PNG created during analysis
        png = res.get("report_png_bytes", b"")
        if png:
            st.image(png, width="stretch")
        else:
            st.warning("No plot data available.")

    png = res.get("report_png_bytes", b"")
    if png:
        st.download_button(
            label=s("download_report_button"),
            data=png,
            file_name="powerwave_report.png",
            mime="image/png",
            width="stretch",
        )

    st.markdown("---")
    st.button("⬅ Back" if ss.lang == "en" else "⬅ Voltar", key="step5_back", on_click=go_prev)

    with st.expander(s("about_title"), expanded=False):
        st.markdown(s("about_body", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def main() -> None:
    """Streamlit app entrypoint."""
    init_state()
    ss = st.session_state

    lang_choice = st.sidebar.selectbox(
        "Language / Idioma",
        options=[("en", "English"), ("pt", "Português")],
        format_func=lambda x: x[1],
    )
    ss.lang = lang_choice[0]
    lang: LanguageCode = ss.lang

    s = lambda key, **kw: get_string(lang, key, **kw)

    st.title(s("app_title"))
    st.caption(s("app_tagline"))
    st.markdown("---")

    step_labels = {
        1: s("step1_title"),
        2: s("step2_title"),
        3: s("step3_title"),
        4: s("step4_title"),
        5: s("step5_title"),
    }

    ss.wizard_step = clamp_step(ss.wizard_step)
    render_nav_top(lang, ss.wizard_step, step_labels)

    if ss.wizard_step == 1:
        render_step_1(s)
    elif ss.wizard_step == 2:
        render_step_2(s)
    elif ss.wizard_step == 3:
        render_step_3(s)
    elif ss.wizard_step == 4:
        render_step_4(s)
    else:
        render_step_5(s)


if __name__ == "__main__":
    main()
