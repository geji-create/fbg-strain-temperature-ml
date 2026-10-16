from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .config import SimulationConfig


def fahrenheit_to_celsius(value_f: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value_f) - 32.0) * 5.0 / 9.0


def wavelength_axis_nm(config: SimulationConfig) -> np.ndarray:
    count = int(
        round(
            (config.wavelength_max_nm - config.wavelength_min_nm)
            / config.wavelength_step_nm
        )
    )
    return config.wavelength_min_nm + np.arange(count + 1) * config.wavelength_step_nm


def temperature_levels(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    values_f = np.linspace(
        config.temperature_min_f,
        config.temperature_max_f,
        config.temperature_count,
    )
    return values_f, fahrenheit_to_celsius(values_f)


def bragg_wavelength_nm(
    strain_ue: float, temperature_c: float, config: SimulationConfig
) -> float:
    strain = strain_ue * 1e-6
    delta_t = temperature_c - config.reference_temperature_c
    fractional_shift = (
        (1.0 - config.photoelastic_coefficient) * strain
        + (config.thermal_expansion_per_c + config.thermo_optic_per_c) * delta_t
    )
    return config.reference_bragg_nm * (1.0 + fractional_shift)


def uniform_fbg_reflectivity(
    wavelengths_nm: np.ndarray,
    bragg_nm: float,
    strain_ue: float,
    temperature_c: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Closed-form coupled-mode reflectivity of a uniform, lossless FBG."""
    wavelengths_m = wavelengths_nm * 1e-9
    bragg_m = bragg_nm * 1e-9
    delta_t = temperature_c - config.reference_temperature_c
    modulation_scale = (
        1.0
        + config.coupling_temperature_coefficient_per_c * delta_t
        + config.coupling_strain_coefficient_per_ue * strain_ue
    )
    delta_n = max(config.index_modulation * modulation_scale, 1e-8)
    kappa = np.pi * delta_n / bragg_m
    detuning = 2.0 * np.pi * config.effective_index * (
        1.0 / wavelengths_m - 1.0 / bragg_m
    )
    gamma = np.sqrt((kappa * kappa - detuning * detuning).astype(complex))
    gamma_safe = np.where(np.abs(gamma) < 1e-18, 1e-18 + 0j, gamma)
    sinh_term = np.sinh(gamma_safe * config.grating_length_m)
    cosh_term = np.cosh(gamma_safe * config.grating_length_m)
    numerator = 1j * kappa * sinh_term
    denominator = gamma_safe * cosh_term - 1j * detuning * sinh_term
    reflectivity = np.abs(numerator / denominator) ** 2
    return np.clip(np.real(reflectivity), 0.0, 1.0)


def _instrument_and_noise(
    clean: np.ndarray,
    wavelengths_nm: np.ndarray,
    center_nm: float,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    sigma_points = (
        config.instrument_resolution_fwhm_nm
        / 2.354820045
        / config.wavelength_step_nm
    )
    resolved = gaussian_filter1d(clean, sigma=max(sigma_points, 0.0), mode="nearest")
    power_scale_db = rng.normal(0.0, config.power_scale_std_db)
    baseline_slope = rng.normal(0.0, config.baseline_slope_std_db_per_nm)
    ripple_amplitude = abs(rng.normal(0.0, config.ripple_amplitude_std_db))
    phase = rng.uniform(0.0, 2.0 * np.pi)
    power_db = 10.0 * np.log10(np.clip(resolved, 1e-8, None))
    power_db += power_scale_db
    power_db += baseline_slope * (wavelengths_nm - center_nm)
    power_db += ripple_amplitude * np.sin(
        2.0 * np.pi * (wavelengths_nm - wavelengths_nm[0]) / 0.42 + phase
    )
    power_db += rng.normal(0.0, config.power_noise_std_db, size=power_db.size)
    noisy = np.clip(10.0 ** (power_db / 10.0), 1e-8, 1.0)
    return noisy, {
        "power_scale_db": power_scale_db,
        "baseline_slope_db_per_nm": baseline_slope,
        "ripple_amplitude_db": ripple_amplitude,
    }


def generate_spectra(config: SimulationConfig) -> pd.DataFrame:
    wavelengths_nm = wavelength_axis_nm(config)
    temperatures_f, temperatures_c = temperature_levels(config)
    condition_count = len(config.strain_levels_ue) * len(temperatures_c)
    sample_count = condition_count * config.replicates_per_condition
    spectra = np.empty((sample_count, wavelengths_nm.size), dtype=np.float64)
    metadata: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(config.random_seed)
    row = 0
    condition_id = 0
    for strain_nominal in config.strain_levels_ue:
        for temperature_f, temperature_nominal in zip(temperatures_f, temperatures_c):
            condition_id += 1
            for replicate in range(1, config.replicates_per_condition + 1):
                strain_actual = strain_nominal + rng.normal(0.0, config.strain_jitter_std_ue)
                temperature_actual = temperature_nominal + rng.normal(
                    0.0, config.temperature_jitter_std_c
                )
                wavelength_offset = rng.normal(0.0, config.wavelength_offset_std_nm)
                center_nm = bragg_wavelength_nm(strain_actual, temperature_actual, config)
                measured_center_nm = center_nm + wavelength_offset
                clean = uniform_fbg_reflectivity(
                    wavelengths_nm,
                    measured_center_nm,
                    strain_actual,
                    temperature_actual,
                    config,
                )
                noisy, noise_meta = _instrument_and_noise(
                    clean, wavelengths_nm, measured_center_nm, rng, config
                )
                spectra[row] = noisy
                metadata.append(
                    {
                        "spectrum_id": f"S{row + 1:05d}",
                        "condition_id": condition_id,
                        "replicate": replicate,
                        "strain_nominal_ue": float(strain_nominal),
                        "temperature_nominal_F": float(temperature_f),
                        "temperature_nominal_C": float(temperature_nominal),
                        "strain_actual_ue": float(strain_actual),
                        "temperature_actual_C": float(temperature_actual),
                        "wavelength_offset_nm": float(wavelength_offset),
                        "bragg_center_model_nm": float(center_nm),
                        **noise_meta,
                    }
                )
                row += 1
    spectral_columns = [f"R_{value:.3f}nm" for value in wavelengths_nm]
    return pd.concat(
        [pd.DataFrame(metadata), pd.DataFrame(spectra, columns=spectral_columns)],
        axis=1,
    )


def config_as_dict(config: SimulationConfig) -> dict[str, object]:
    return asdict(config)

