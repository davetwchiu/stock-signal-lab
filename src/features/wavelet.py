"""Rolling wavelet features.

PyWavelets is used when installed. The function returns NaN feature columns if
the optional dependency is unavailable, keeping the app functional while making
the missing advanced feature explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _empty_wavelet_frame(
    index: pd.Index,
    level: int,
    prefix: str,
    available: float,
) -> pd.DataFrame:
    columns = [f"{prefix}_available", f"{prefix}_trend_return", f"{prefix}_short_noise_intensity"]
    columns.extend(f"{prefix}_energy_scale_{scale}" for scale in range(1, level + 1))
    output = pd.DataFrame(index=index, columns=columns, dtype=float)
    output[f"{prefix}_available"] = available
    return output


def rolling_wavelet_features(
    series: pd.Series,
    window: int = 64,
    wavelet: str = "db4",
    level: int = 3,
    prefix: str = "wavelet",
) -> pd.DataFrame:
    """Extract trailing-window wavelet scale energy and trend/noise measures."""

    try:
        import pywt
    except ImportError:
        return _empty_wavelet_frame(series.index, level, prefix, available=0.0)

    output = _empty_wavelet_frame(series.index, level, prefix, available=1.0)
    values = np.log(series.astype(float)).to_numpy()

    for end in range(window - 1, len(values)):
        window_values = values[end - window + 1 : end + 1]
        if not np.isfinite(window_values).all():
            continue

        max_level = pywt.dwt_max_level(window, pywt.Wavelet(wavelet).dec_len)
        active_level = min(level, max_level)
        if active_level < 1:
            continue

        coeffs = pywt.wavedec(window_values, wavelet=wavelet, level=active_level, mode="periodization")
        energies = [float(np.sum(coeff**2)) for coeff in coeffs[1:]]
        total_energy = float(sum(energies) + np.sum(coeffs[0] ** 2))

        trend_coeffs = [coeffs[0]] + [np.zeros_like(coeff) for coeff in coeffs[1:]]
        trend = pywt.waverec(trend_coeffs, wavelet=wavelet, mode="periodization")[:window]
        output.at[series.index[end], f"{prefix}_trend_return"] = float(np.exp(trend[-1] - trend[0]) - 1.0)

        detail_by_short_scale = list(reversed(energies))
        for scale, energy in enumerate(detail_by_short_scale[:level], start=1):
            output.at[series.index[end], f"{prefix}_energy_scale_{scale}"] = (
                energy / total_energy if total_energy > 0 else np.nan
            )

        short_energy = detail_by_short_scale[0] if detail_by_short_scale else np.nan
        output.at[series.index[end], f"{prefix}_short_noise_intensity"] = (
            short_energy / total_energy if total_energy > 0 else np.nan
        )

    return output

