from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from .config import FeatureConfig


SPECTRAL_COLUMN_PATTERN = re.compile(r"^R_([0-9.]+)nm$")
FEATURE_COLUMNS = [
    "center_peak_wavelength_nm",
    "center_peak_power_db",
    "left_sidelobe_wavelength_nm",
    "left_sidelobe_power_db",
    "right_sidelobe_wavelength_nm",
    "right_sidelobe_power_db",
]


def spectral_columns_and_axis(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    pairs: list[tuple[str, float]] = []
    for column in frame.columns:
        match = SPECTRAL_COLUMN_PATTERN.match(str(column))
        if match:
            pairs.append((column, float(match.group(1))))
    if not pairs:
        raise ValueError("No spectral columns matching R_<wavelength>nm were found.")
    pairs.sort(key=lambda item: item[1])
    return [item[0] for item in pairs], np.asarray([item[1] for item in pairs])


def _quadratic_peak(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= len(y) - 1:
        return float(x[index]), float(y[index])
    y0, y1, y2 = y[index - 1], y[index], y[index + 1]
    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) < 1e-15:
        return float(x[index]), float(y[index])
    offset = 0.5 * (y0 - y2) / denominator
    step = x[index + 1] - x[index]
    peak_x = x[index] + offset * step
    peak_y = y1 - 0.25 * (y0 - y2) * offset
    return float(peak_x), float(peak_y)


def _side_peak(
    wavelengths: np.ndarray,
    power_db: np.ndarray,
    center_nm: float,
    direction: int,
    config: FeatureConfig,
) -> tuple[float, float, bool]:
    if direction < 0:
        mask = (wavelengths <= center_nm - config.sidelobe_near_nm) & (
            wavelengths >= center_nm - config.sidelobe_far_nm
        )
    else:
        mask = (wavelengths >= center_nm + config.sidelobe_near_nm) & (
            wavelengths <= center_nm + config.sidelobe_far_nm
        )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return float("nan"), float("nan"), True
    local = power_db[indices]
    peaks, _ = find_peaks(local, prominence=0.05)
    used_fallback = peaks.size == 0
    local_index = int(np.argmax(local) if used_fallback else peaks[np.argmax(local[peaks])])
    global_index = int(indices[local_index])
    peak_wavelength, peak_power = _quadratic_peak(wavelengths, power_db, global_index)
    return peak_wavelength, peak_power, used_fallback


def _fwhm_nm(wavelengths: np.ndarray, reflectivity: np.ndarray) -> float:
    peak = float(np.max(reflectivity))
    above = np.flatnonzero(reflectivity >= peak / 2.0)
    if above.size < 2:
        return float("nan")
    return float(wavelengths[above[-1]] - wavelengths[above[0]])


def extract_six_features(
    frame: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    spectral_columns, wavelengths = spectral_columns_and_axis(frame)
    spectra = frame[spectral_columns].to_numpy(dtype=float)
    metadata_columns = [
        name
        for name in [
            "dataset",
            "spectrum_id",
            "condition_id",
            "channel",
            "source_file",
            "replicate",
            "strain_nominal_ue",
            "temperature_nominal_F",
            "temperature_nominal_C",
            "strain_actual_ue",
            "temperature_actual_C",
        ]
        if name in frame.columns
    ]
    results: list[dict[str, float | str]] = []
    window = min(config.savgol_window, spectra.shape[1] - (1 - spectra.shape[1] % 2))
    if window % 2 == 0:
        window -= 1
    window = max(window, config.savgol_order + 3)
    if window % 2 == 0:
        window += 1
    for row in spectra:
        clipped = np.clip(row, 1e-10, None)
        smooth = savgol_filter(clipped, window, config.savgol_order, mode="interp")
        power_db = 10.0 * np.log10(np.clip(smooth, 1e-10, None))
        center_index = int(np.argmax(smooth))
        center_wavelength, center_power = _quadratic_peak(
            wavelengths, power_db, center_index
        )
        left_wavelength, left_power, left_fallback = _side_peak(
            wavelengths, power_db, center_wavelength, -1, config
        )
        right_wavelength, right_power, right_fallback = _side_peak(
            wavelengths, power_db, center_wavelength, 1, config
        )
        warnings = []
        if left_fallback:
            warnings.append("left_fallback")
        if right_fallback:
            warnings.append("right_fallback")
        results.append(
            {
                "center_peak_wavelength_nm": center_wavelength,
                "center_peak_power_db": center_power,
                "left_sidelobe_wavelength_nm": left_wavelength,
                "left_sidelobe_power_db": left_power,
                "right_sidelobe_wavelength_nm": right_wavelength,
                "right_sidelobe_power_db": right_power,
                "halfmax_width_nm": _fwhm_nm(wavelengths, smooth),
                "feature_warning": ";".join(warnings),
            }
        )
    return pd.concat(
        [frame[metadata_columns].reset_index(drop=True), pd.DataFrame(results)], axis=1
    )

