from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from powerwave.i18n import get_string, LanguageCode


@dataclass(frozen=True, slots=True)
class HarmonicSpec:
    """Specification for a harmonic component."""
    order: int
    percent_of_fundamental_peak: float
    phase_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class SagSpec:
    """Voltage sag envelope definition."""
    start_s: float
    duration_s: float
    depth_percent: float


@dataclass(frozen=True, slots=True)
class SwellSpec:
    """Voltage swell envelope definition."""
    start_s: float
    duration_s: float
    rise_percent: float


def parse_harmonics(spec: str) -> list[HarmonicSpec]:
    """
    Parse harmonic specification string.

    Supported formats:
    - "3:10,5:5"
    - "3:10@20,5:5@-15"

    Percent values are relative to fundamental PEAK.
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    out: list[HarmonicSpec] = []
    for token in (p.strip() for p in spec.split(",") if p.strip()):
        left, phase_str = (token.split("@", 1) + ["0"])[:2]
        if ":" not in left:
            raise ValueError(
                f"Invalid harmonic token '{token}'. Expected 'order:percent' or 'order:percent@phase'."
            )

        order_str, perc_str = left.split(":", 1)
        order = int(order_str.strip())
        perc = float(perc_str.strip())
        phase_deg = float(phase_str.strip())

        if order < 2:
            raise ValueError("Harmonic order must be >= 2.")
        if perc < 0:
            raise ValueError("Harmonic percent must be >= 0.")

        out.append(HarmonicSpec(order=order, percent_of_fundamental_peak=perc, phase_deg=phase_deg))

    return out


def step_envelope(t: np.ndarray, start_s: float, duration_s: float, factor: float) -> np.ndarray:
    """
    Build a piecewise-constant multiplicative envelope.

    Returns an array of ones, with `factor` applied for t in [start_s, start_s + duration_s).
    """
    start = float(start_s)
    end = start + float(duration_s)

    env = np.ones_like(t)
    if duration_s > 0:
        mask = (t >= start) & (t < end)
        env[mask] = float(factor)
    return env


def build_sag_envelope(t: np.ndarray, sag: Optional[SagSpec]) -> np.ndarray:
    """
    Return the sag multiplicative envelope (1 outside sag, (1 - depth) during sag).
    """
    if sag is None:
        return np.ones_like(t)

    depth = np.clip(float(sag.depth_percent) / 100.0, 0.0, 1.0)
    return step_envelope(t, sag.start_s, sag.duration_s, 1.0 - depth)


def build_swell_envelope(t: np.ndarray, swell: Optional[SwellSpec]) -> np.ndarray:
    """
    Return the swell multiplicative envelope (1 outside swell, (1 + rise) during swell).
    """
    if swell is None:
        return np.ones_like(t)

    rise = np.clip(float(swell.rise_percent) / 100.0, 0.0, 5.0)
    return step_envelope(t, swell.start_s, swell.duration_s, 1.0 + rise)


def generate_synthetic_waveform(
    filepath: str,
    sampling_rate: float = 10_000.0,
    duration: float = 0.2,
    fundamental_freq: float = 60.0,
    fundamental_vrms: float = 230.0,
    fundamental_phase_deg: float = 0.0,
    harmonics: Optional[Sequence[HarmonicSpec]] = None,
    dc_offset: float = 0.0,
    noise_std: float = 2.0,
    clip_peak: Optional[float] = None,
    sag: Optional[SagSpec] = None,
    swell: Optional[SwellSpec] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic voltage waveform and save it as CSV.

    Output columns:
    - time: seconds
    - voltage: volts
    """
    fs = float(sampling_rate)
    dur = float(duration)
    f0 = float(fundamental_freq)
    vrms = float(fundamental_vrms)

    if fs <= 0:
        raise ValueError("sampling_rate must be > 0.")
    if dur <= 0:
        raise ValueError("duration must be > 0.")
    if f0 <= 0:
        raise ValueError("fundamental_freq must be > 0.")
    if vrms <= 0:
        raise ValueError("fundamental_vrms must be > 0.")

    n_samples = int(round(fs * dur))
    if n_samples < 8:
        raise ValueError("Too few samples. Increase duration or sampling_rate.")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / fs

    v1_peak = vrms * np.sqrt(2.0)
    phi1 = np.deg2rad(float(fundamental_phase_deg))
    omega0 = 2.0 * np.pi * f0

    v = v1_peak * np.sin(omega0 * t + phi1)

    for h in (harmonics or ()):
        vh_peak = (float(h.percent_of_fundamental_peak) / 100.0) * v1_peak
        phih = np.deg2rad(float(h.phase_deg))
        v += vh_peak * np.sin((omega0 * float(h.order)) * t + phih)

    v *= build_sag_envelope(t, sag) * build_swell_envelope(t, swell)

    if dc_offset:
        v += float(dc_offset)

    if noise_std and float(noise_std) > 0:
        v += float(noise_std) * rng.standard_normal(n_samples)

    if clip_peak is not None:
        cp = float(clip_peak)
        if cp > 0:
            v = np.clip(v, -cp, cp)

    df = pd.DataFrame({"time": t, "voltage": v})

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def preset_to_params(preset: str) -> dict:
    """
    Map preset name to waveform parameters.
    """
    name = (preset or "").strip().lower()

    if name == "grid_pure":
        return dict(
            fundamental_vrms=230.0,
            harmonics=[],
            noise_std=1.0,
            dc_offset=0.0,
            clip_peak=None,
            sag=None,
            swell=None,
        )

    if name == "grid_distorted":
        return dict(
            fundamental_vrms=230.0,
            harmonics=parse_harmonics("3:10@10,5:5@-15,7:3@25"),
            noise_std=2.0,
            dc_offset=0.0,
            clip_peak=None,
            sag=None,
            swell=None,
        )

    if name == "sag_30":
        return dict(
            fundamental_vrms=230.0,
            harmonics=parse_harmonics("3:8,5:4"),
            noise_std=2.0,
            dc_offset=0.0,
            clip_peak=None,
            sag=SagSpec(start_s=0.06, duration_s=0.06, depth_percent=30.0),
            swell=None,
        )

    if name == "swell_20":
        return dict(
            fundamental_vrms=230.0,
            harmonics=parse_harmonics("3:8,5:4"),
            noise_std=2.0,
            dc_offset=0.0,
            clip_peak=None,
            sag=None,
            swell=SwellSpec(start_s=0.06, duration_s=0.06, rise_percent=20.0),
        )

    if name == "clipped":
        return dict(
            fundamental_vrms=230.0,
            harmonics=parse_harmonics("3:12,5:6"),
            noise_std=1.5,
            dc_offset=0.0,
            clip_peak=260.0,
            sag=None,
            swell=None,
        )

    raise ValueError(f"Unknown preset '{preset}'.")


def build_argparser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.
    """
    p = argparse.ArgumentParser(description="PowerWave synthetic waveform CSV generator.")
    p.add_argument("--out", type=str, default="examples/grid_distorted.csv", help="Output CSV path.")
    p.add_argument("--lang", type=str, choices=["en", "pt"], default="en", help="Language for messages.")

    p.add_argument(
        "--preset",
        type=str,
        default="grid_distorted",
        help="Preset: grid_pure, grid_distorted, sag_30, swell_20, clipped.",
    )
    p.add_argument("--sampling-rate", type=float, default=10_000.0, help="Sampling rate [Hz].")
    p.add_argument("--duration", type=float, default=0.2, help="Duration [s].")
    p.add_argument("--fundamental", type=float, default=60.0, help="Fundamental frequency [Hz].")
    p.add_argument("--vrms", type=float, default=230.0, help="Fundamental RMS voltage [V].")
    p.add_argument("--phase-deg", type=float, default=0.0, help="Fundamental phase [deg].")

    p.add_argument(
        "--harmonics",
        type=str,
        default="",
        help="Harmonics: '3:10@20,5:5@-15' (percent of fundamental PEAK).",
    )
    p.add_argument("--noise-std", type=float, default=2.0, help="Gaussian noise standard deviation.")
    p.add_argument("--dc", type=float, default=0.0, help="DC offset.")
    p.add_argument("--clip-peak", type=float, default=None, help="If set and >0, clip waveform to +/- clip_peak.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible noise.")

    p.add_argument("--sag-start", type=float, default=None, help="Sag start [s].")
    p.add_argument("--sag-duration", type=float, default=None, help="Sag duration [s].")
    p.add_argument("--sag-depth", type=float, default=None, help="Sag depth [%].")

    p.add_argument("--swell-start", type=float, default=None, help="Swell start [s].")
    p.add_argument("--swell-duration", type=float, default=None, help="Swell duration [s].")
    p.add_argument("--swell-rise", type=float, default=None, help="Swell rise [%].")
    return p


def maybe_override_sag(args: argparse.Namespace, base: Optional[SagSpec]) -> Optional[SagSpec]:
    """
    Override preset sag if all sag CLI values are provided.
    """
    if args.sag_start is None or args.sag_duration is None or args.sag_depth is None:
        return base
    return SagSpec(start_s=float(args.sag_start), duration_s=float(args.sag_duration), depth_percent=float(args.sag_depth))


def maybe_override_swell(args: argparse.Namespace, base: Optional[SwellSpec]) -> Optional[SwellSpec]:
    """
    Override preset swell if all swell CLI values are provided.
    """
    if args.swell_start is None or args.swell_duration is None or args.swell_rise is None:
        return base
    return SwellSpec(
        start_s=float(args.swell_start),
        duration_s=float(args.swell_duration),
        rise_percent=float(args.swell_rise),
    )


def main() -> None:
    """
    CLI entrypoint.
    """
    args = build_argparser().parse_args()
    lang: LanguageCode = "pt" if args.lang == "pt" else "en"

    try:
        preset_params = preset_to_params(args.preset)

        harmonics = parse_harmonics(args.harmonics) if args.harmonics.strip() else preset_params["harmonics"]
        sag = maybe_override_sag(args, preset_params["sag"])
        swell = maybe_override_swell(args, preset_params["swell"])

        clip_peak = args.clip_peak if args.clip_peak is not None else preset_params["clip_peak"]

        print(get_string(lang, "synth_header"))
        print(get_string(lang, "synth_preset", preset=args.preset))
        print(
            get_string(
                lang,
                "synth_params",
                fs=float(args.sampling_rate),
                duration=float(args.duration),
                f0=float(args.fundamental),
                vrms=float(args.vrms),
            )
        )
        print(get_string(lang, "synth_generating", path=args.out))

        generate_synthetic_waveform(
            filepath=args.out,
            sampling_rate=float(args.sampling_rate),
            duration=float(args.duration),
            fundamental_freq=float(args.fundamental),
            fundamental_vrms=float(args.vrms),
            fundamental_phase_deg=float(args.phase_deg),
            harmonics=harmonics,
            dc_offset=float(args.dc),
            noise_std=float(args.noise_std),
            clip_peak=clip_peak,
            sag=sag,
            swell=swell,
            seed=args.seed,
        )

        print(get_string(lang, "synth_saved", path=args.out))

    except Exception as exc:
        print(get_string(lang, "synth_error", error=str(exc)))
        raise


if __name__ == "__main__":
    main()