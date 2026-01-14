import pandas as pd

def load_waveform_csv(
    filepath: str,
    time_column: str = "time",
    signal_column: str = "voltage",
) -> tuple[pd.Series, pd.Series]:
    """
    Load time-domain waveform from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    time_column : str
        Column name for time values (seconds).
    signal_column : str
        Column name for the waveform (voltage/current).

    Returns
    -------
    time : pd.Series
        Time vector in seconds.
    signal : pd.Series
        Waveform values (floats).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(filepath)

    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in CSV.")
    if signal_column not in df.columns:
        raise ValueError(f"Signal column '{signal_column}' not found in CSV.")

    time = df[time_column].astype(float)
    signal = df[signal_column].astype(float)

    return time, signal