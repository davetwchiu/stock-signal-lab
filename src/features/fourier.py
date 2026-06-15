"""Rolling Fourier features with explicit trailing-window semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_source(series: pd.Series, input_mode: str) -> pd.Series:
    if input_mode == "returns":
        return series.pct_change()
    if input_mode == "log_price":
        return np.log(series)
    if input_mode == "log_returns":
        return np.log(series).diff()
    raise ValueError("input_mode must be one of: returns, log_returns, log_price")


def _spectral_entropy(power: np.ndarray) -> float:
    total = power.sum()
    if total <= 0:
        return np.nan
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * np.log(probabilities)).sum()
    return float(entropy / np.log(len(power))) if len(power) > 1 else 0.0


def rolling_fourier_features(
    series: pd.Series,
    window: int = 60,
    n_components: int = 3,
    input_mode: str = "returns",
    prefix: str = "fourier",
) -> pd.DataFrame:
    """Extract dominant frequencies and energy measures from trailing windows.

    For row `t`, the FFT is computed only on observations ending at `t`.
    """

    source = _prepare_source(series.astype(float), input_mode)
    columns = [
        f"{prefix}_energy_concentration",
        f"{prefix}_spectral_entropy",
        f"{prefix}_cycle_clarity",
        f"{prefix}_cycle_strength",
        f"{prefix}_noise_diffusion",
    ]
    for rank in range(1, n_components + 1):
        columns.extend(
            [
                f"{prefix}_freq_{rank}",
                f"{prefix}_period_{rank}",
                f"{prefix}_amp_{rank}",
            ]
        )

    output = pd.DataFrame(index=series.index, columns=columns, dtype=float)
    values = source.to_numpy(dtype=float)

    for end in range(window - 1, len(values)):
        window_values = values[end - window + 1 : end + 1]
        if not np.isfinite(window_values).all():
            continue

        demeaned = window_values - window_values.mean()
        spectrum = np.fft.rfft(demeaned)
        frequencies = np.fft.rfftfreq(window)
        amplitudes = np.abs(spectrum)
        power = amplitudes**2

        nonzero = np.arange(1, len(frequencies))
        if len(nonzero) == 0 or power[nonzero].sum() <= 0:
            continue

        ranked = nonzero[np.argsort(power[nonzero])[::-1]]
        top = ranked[:n_components]
        total_energy = power[nonzero].sum()
        energy_concentration = power[top].sum() / total_energy
        spectral_entropy = _spectral_entropy(power[nonzero])
        dominant_amplitude = amplitudes[top[0]]

        output.iat[end, output.columns.get_loc(f"{prefix}_energy_concentration")] = (
            energy_concentration
        )
        output.iat[end, output.columns.get_loc(f"{prefix}_spectral_entropy")] = spectral_entropy
        output.iat[end, output.columns.get_loc(f"{prefix}_cycle_clarity")] = (
            energy_concentration / (1.0 + spectral_entropy)
            if np.isfinite(spectral_entropy)
            else np.nan
        )
        output.iat[end, output.columns.get_loc(f"{prefix}_cycle_strength")] = (
            dominant_amplitude * energy_concentration
            if np.isfinite(dominant_amplitude)
            else np.nan
        )
        output.iat[end, output.columns.get_loc(f"{prefix}_noise_diffusion")] = (
            spectral_entropy / (1.0 + energy_concentration)
            if np.isfinite(spectral_entropy)
            else np.nan
        )

        for rank, idx in enumerate(top, start=1):
            freq = frequencies[idx]
            output.iat[end, output.columns.get_loc(f"{prefix}_freq_{rank}")] = freq
            output.iat[end, output.columns.get_loc(f"{prefix}_period_{rank}")] = (
                1.0 / freq if freq else np.nan
            )
            output.iat[end, output.columns.get_loc(f"{prefix}_amp_{rank}")] = amplitudes[idx]

    return output
