from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal, Mapping

LanguageCode = Literal["en", "pt"]

_EN: Final[dict[str, str]] = {
    "app_title": "PowerWave Analyzer",
    "app_tagline": "Interactive offline analysis of electrical waveforms stored as CSV files.",
    "step1_title": "Step 1 · Analysis settings",
    "step2_title": "Step 2 · Upload waveform CSV",
    "step3_title": "Step 3 · Select time and signal columns",
    "step4_title": "Step 4 · Run analysis",
    "step5_title": "Step 5 · Results",
    "sampling_rate_label": "Sampling rate [Hz]",
    "sampling_rate_help": "Sampling frequency used to acquire the waveform.",
    "fundamental_label": "Fundamental frequency [Hz]",
    "fundamental_help": "Expected fundamental (e.g., 50 or 60 Hz for power systems).",
    "max_freq_label": "Max spectrum frequency [Hz]",
    "max_freq_help": "Upper frequency limit for spectrum and harmonic plots.",
    "upload_label": "Choose a CSV file (time column + signal column)",
    "upload_help": "CSV with at least a time column and one signal column.",
    "preview_title": "Preview of uploaded CSV",
    "no_csv_warning": "Upload a CSV file to continue to the next step.",
    "csv_no_columns": "CSV has no columns.",
    "csv_detected_columns": "Detected columns: {columns}",
    "time_column_label": "Time column",
    "signal_column_label": "Signal column (voltage/current)",
    "run_analysis_button": "▶ Analyze waveform",
    "results_title": "Numerical metrics",
    "metric_samples": "Samples",
    "metric_sampling_rate": "Sampling rate [Hz]",
    "metric_fundamental": "Fundamental [Hz]",
    "metric_rms": "RMS value",
    "metric_crest": "Crest factor",
    "metric_thd": "THD [%]",
    "plots_title": "Plots",
    "download_report_button": "💾 Download report (PNG)",
    "export_json_title": "Export results (JSON)",
    "about_title": "ℹ About this tool",
    "about_body": (
        "- Developed as a study project for power quality and signal processing.\n"
        "- Core stack: **NumPy**, **SciPy**, **Pandas**, **Matplotlib**, **Streamlit**.\n"
        "- Computes RMS, crest factor, FFT spectrum and total harmonic distortion (THD).\n"
        "- Source code: [lnpotter/Powerwave-Analyzer]"
        "(https://github.com/lnpotter/Powerwave-Analyzer)\n"
        "- Session started: `{timestamp}`"
    ),
    "thd_not_computed": "THD could not be computed: {error}",
    "fft_note": (
        "A Hann window is applied before FFT to reduce spectral leakage. "
        "THD is computed as the RMS ratio between harmonic content and the fundamental."
    ),
    "cli_header": "=== PowerWave Analyzer ===",
    "cli_file": "File: {file}",
    "cli_samples": "Samples: {samples}",
    "cli_sampling_rate": "Sampling rate: {fs:.1f} Hz",
    "cli_fundamental": "Fundamental frequency (assumed): {f0:.1f} Hz",
    "cli_rms": "RMS value: {rms:.4f}",
    "cli_crest": "Crest factor: {crest:.4f}",
    "cli_thd": "THD: {thd:.2f} %",
    "cli_saved_report": "Saved report image to: {path}",
    "cli_no_report": "No --save-report path provided; no image saved.",
    "plot_title_time": "Time-domain waveform",
    "plot_xlabel_time_s": "Time [s]",
    "plot_ylabel_signal": "Signal amplitude",
    "plot_title_spectrum": "Magnitude spectrum",
    "plot_xlabel_freq_hz": "Frequency [Hz]",
    "plot_ylabel_mag_db": "Magnitude [dB]",
    "plot_title_harmonics": "Harmonic magnitudes",
    "plot_xlabel_order": "Harmonic order",
    "plot_ylabel_mag": "Magnitude",
    "plot_legend_signal": "Signal",
    "plot_legend_spectrum": "Spectrum",
    "plot_legend_harmonics": "Harmonics",
    "synth_header": "=== Synthetic CSV Generator ===",
    "synth_preset": "Preset: {preset}",
    "synth_params": "Parameters: fs={fs:.1f} Hz, duration={duration:.3f} s, f0={f0:.2f} Hz, Vrms={vrms:.2f} V",
    "synth_generating": "Generating synthetic waveform CSV: {path}",
    "synth_saved": "Synthetic CSV saved to: {path}",
    "synth_error": "Failed to generate synthetic waveform: {error}",
}

_PT: Final[dict[str, str]] = {
    "app_title": "PowerWave Analyzer",
    "app_tagline": "Análise offline interativa de formas de onda elétricas em arquivos CSV.",
    "step1_title": "Etapa 1 · Configurações de análise",
    "step2_title": "Etapa 2 · Enviar CSV da forma de onda",
    "step3_title": "Etapa 3 · Selecionar colunas de tempo e sinal",
    "step4_title": "Etapa 4 · Executar análise",
    "step5_title": "Etapa 5 · Resultados",
    "sampling_rate_label": "Taxa de amostragem [Hz]",
    "sampling_rate_help": "Frequência de amostragem usada para adquirir a forma de onda.",
    "fundamental_label": "Frequência fundamental [Hz]",
    "fundamental_help": "Fundamental esperada (por exemplo, 50 ou 60 Hz em sistemas de potência).",
    "max_freq_label": "Frequência máxima do espectro [Hz]",
    "max_freq_help": "Limite superior de frequência para o espectro e os harmônicos.",
    "upload_label": "Escolha um arquivo CSV (coluna de tempo + coluna de sinal)",
    "upload_help": "CSV com pelo menos uma coluna de tempo e uma coluna de sinal.",
    "preview_title": "Pré-visualização do CSV enviado",
    "no_csv_warning": "Envie um arquivo CSV para continuar para a próxima etapa.",
    "csv_no_columns": "O CSV não possui colunas.",
    "csv_detected_columns": "Colunas detectadas: {columns}",
    "time_column_label": "Coluna de tempo",
    "signal_column_label": "Coluna de sinal (tensão/corrente)",
    "run_analysis_button": "▶ Analisar forma de onda",
    "results_title": "Métricas numéricas",
    "metric_samples": "Amostras",
    "metric_sampling_rate": "Taxa de amostragem [Hz]",
    "metric_fundamental": "Fundamental [Hz]",
    "metric_rms": "Valor eficaz (RMS)",
    "metric_crest": "Fator de crista",
    "metric_thd": "Distorção harmônica total (THD) [%]",
    "plots_title": "Gráficos",
    "download_report_button": "💾 Baixar relatório (PNG)",
    "export_json_title": "Exportar resultados (JSON)",
    "about_title": "ℹ Sobre esta ferramenta",
    "about_body": (
        "- Desenvolvido como projeto de estudo em qualidade de energia e processamento de sinais.\n"
        "- Stack principal: **NumPy**, **SciPy**, **Pandas**, **Matplotlib**, **Streamlit**.\n"
        "- Calcula RMS, fator de crista, espectro via FFT e distorção harmônica total (THD).\n"
        "- Código-fonte: [lnpotter/Powerwave-Analyzer]"
        "(https://github.com/lnpotter/Powerwave-Analyzer)\n"
        "- Sessão iniciada em: `{timestamp}`"
    ),
    "thd_not_computed": "Não foi possível calcular a THD: {error}",
    "fft_note": (
        "Uma janela de Hann é aplicada antes da FFT para reduzir vazamento espectral. "
        "A THD é calculada como a razão em RMS entre o conteúdo harmônico e a fundamental."
    ),
    "cli_header": "=== PowerWave Analyzer ===",
    "cli_file": "Arquivo: {file}",
    "cli_samples": "Amostras: {samples}",
    "cli_sampling_rate": "Taxa de amostragem: {fs:.1f} Hz",
    "cli_fundamental": "Frequência fundamental (assumida): {f0:.1f} Hz",
    "cli_rms": "Valor eficaz (RMS): {rms:.4f}",
    "cli_crest": "Fator de crista: {crest:.4f}",
    "cli_thd": "THD: {thd:.2f} %",
    "cli_saved_report": "Relatório salvo em: {path}",
    "cli_no_report": "Nenhum caminho em --save-report; nenhuma imagem foi salva.",
    "plot_title_time": "Forma de onda no tempo",
    "plot_xlabel_time_s": "Tempo [s]",
    "plot_ylabel_signal": "Amplitude do sinal",
    "plot_title_spectrum": "Espectro em magnitude",
    "plot_xlabel_freq_hz": "Frequência [Hz]",
    "plot_ylabel_mag_db": "Magnitude [dB]",
    "plot_title_harmonics": "Magnitudes harmônicas",
    "plot_xlabel_order": "Ordem harmônica",
    "plot_ylabel_mag": "Magnitude",
    "plot_legend_signal": "Sinal",
    "plot_legend_spectrum": "Espectro",
    "plot_legend_harmonics": "Harmônicos",
    "synth_header": "=== Gerador de CSV Sintético ===",
    "synth_preset": "Preset: {preset}",
    "synth_params": "Parâmetros: fs={fs:.1f} Hz, duração={duration:.3f} s, f0={f0:.2f} Hz, Vrms={vrms:.2f} V",
    "synth_generating": "Gerando CSV de forma de onda sintética: {path}",
    "synth_saved": "CSV sintético salvo em: {path}",
    "synth_error": "Falha ao gerar forma de onda sintética: {error}",
}

_STRINGS: Final[Mapping[LanguageCode, Mapping[str, str]]] = MappingProxyType(
    {
        "en": MappingProxyType(_EN),
        "pt": MappingProxyType(_PT),
    }
)


def get_string(lang: LanguageCode, key: str, **kwargs) -> str:
    """
    Retrieve a localized string with English fallback.

    Args:
        lang: Language code ('en' or 'pt').
        key: String identifier.
        **kwargs: Optional format parameters for the template.

    Returns:
        The localized string. If the language or key is missing, falls back to English.
        If formatting fails, returns the unformatted template.
    """
    en = _STRINGS["en"]
    strings = _STRINGS.get(lang, en)

    text = strings.get(key) or en.get(key) or key
    if not kwargs:
        return text

    try:
        return text.format(**kwargs)
    except (KeyError, ValueError):
        return text
