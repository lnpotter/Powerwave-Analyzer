import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_waveform(
    filepath: str,
    sampling_rate: float = 10000.0,
    duration: float = 0.2,
    fundamental_freq: float = 60.0,
):
    n_samples = int(sampling_rate * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    v1_peak = 325.0
    v1 = v1_peak * np.sin(2.0 * np.pi * fundamental_freq * t)

    v3_peak = 0.1 * v1_peak
    v5_peak = 0.05 * v1_peak
    v3 = v3_peak * np.sin(2.0 * np.pi * 3 * fundamental_freq * t)
    v5 = v5_peak * np.sin(2.0 * np.pi * 5 * fundamental_freq * t)

    noise = 5.0 * np.random.randn(n_samples)

    voltage = v1 + v3 + v5 + noise

    df = pd.DataFrame({"time": t, "voltage": voltage})
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    generate_synthetic_waveform("examples/grid_pure.csv")
    print("Synthetic CSV saved to examples/grid_pure.csv")