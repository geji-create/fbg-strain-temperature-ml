"""Generate an expanded, leakage-safe synthetic FBG dataset.

The dataset follows the physical setup used in the earlier "模拟均匀FBG反射谱"
work: a uniform-FBG coupled-mode model and an independent SimPlus-like transfer
model are evaluated on the same temperature/strain conditions.  Unlike the
original 10 x 7 grid, this script uses Latin-hypercube sampling so that adding
rows really adds new operating conditions instead of only adding noisy copies.

Outputs
-------
expanded_conditions.csv
    One row per independent temperature/strain condition.
expanded_six_features.csv
    One row per condition and solver, containing the six peak/side-lobe
    features used by the machine-learning pipeline.
expanded_spectra.npz
    Compressed float32 spectra for later 1-D CNN experiments.
expanded_dataset_metadata.json
    Generation parameters and array shapes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REFERENCE_TEMPERATURE_C = 23.0
REFERENCE_BRAGG_NM = 1548.097
TEMPERATURE_MIN_C = (46.0 - 32.0) * 5.0 / 9.0
TEMPERATURE_MAX_C = (105.0 - 32.0) * 5.0 / 9.0
STRAIN_MIN_UE = 0.0
STRAIN_MAX_UE = 900.0


def latin_hypercube(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Return an n x dimensions reproducible Latin-hypercube design."""
    result = np.empty((n, dimensions), dtype=np.float64)
    for column in range(dimensions):
        bins = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        result[:, column] = bins[rng.permutation(n)]
    return result


def uniform_fbg_reflectivity(
    wavelength_nm: np.ndarray,
    bragg_nm: float,
    effective_index: float,
    grating_length_m: float,
    index_modulation: float,
) -> np.ndarray:
    """Closed-form reflectivity of a lossless, uniform FBG."""
    wavelength_m = wavelength_nm * 1e-9
    bragg_m = bragg_nm * 1e-9
    kappa = np.pi * index_modulation / bragg_m
    detuning = 2.0 * np.pi * effective_index * (1.0 / wavelength_m - 1.0 / bragg_m)
    discriminant = kappa * kappa - detuning * detuning
    reflectivity = np.empty_like(wavelength_nm, dtype=np.float64)

    positive = discriminant > 1e-12
    if np.any(positive):
        gamma = np.sqrt(discriminant[positive])
        sinh_term = np.sinh(gamma * grating_length_m)
        cosh_term = np.cosh(gamma * grating_length_m)
        numerator = (kappa * sinh_term) ** 2
        denominator = (gamma * cosh_term) ** 2 + (
            detuning[positive] * sinh_term
        ) ** 2
        reflectivity[positive] = numerator / np.maximum(denominator, 1e-30)

    negative = discriminant < -1e-12
    if np.any(negative):
        gamma = np.sqrt(-discriminant[negative])
        sin_term = np.sin(gamma * grating_length_m)
        cos_term = np.cos(gamma * grating_length_m)
        numerator = (kappa * sin_term) ** 2
        denominator = (gamma * cos_term) ** 2 + (
            detuning[negative] * sin_term
        ) ** 2
        reflectivity[negative] = numerator / np.maximum(denominator, 1e-30)

    boundary = ~(positive | negative)
    if np.any(boundary):
        kl = kappa * grating_length_m
        dl = detuning[boundary] * grating_length_m
        reflectivity[boundary] = kl * kl / (1.0 + dl * dl)

    return np.clip(reflectivity, 1e-10, 0.999999)


def segmented_fbg_reflectivity(
    wavelength_nm: np.ndarray,
    local_bragg_nm: np.ndarray,
    effective_index: float,
    grating_length_m: float,
    local_index_modulation: np.ndarray,
) -> np.ndarray:
    """Transfer-matrix solution for an FBG with a longitudinal strain profile."""
    wavelength_m = wavelength_nm * 1e-9
    segment_length_m = grating_length_m / len(local_bragg_nm)
    total_11 = np.ones(wavelength_nm.size, dtype=np.complex128)
    total_12 = np.zeros(wavelength_nm.size, dtype=np.complex128)
    total_21 = np.zeros(wavelength_nm.size, dtype=np.complex128)
    total_22 = np.ones(wavelength_nm.size, dtype=np.complex128)

    for bragg_nm, index_modulation in zip(local_bragg_nm, local_index_modulation):
        bragg_m = float(bragg_nm) * 1e-9
        kappa = np.pi * float(index_modulation) / bragg_m
        detuning = 2.0 * np.pi * effective_index * (
            1.0 / wavelength_m - 1.0 / bragg_m
        )
        gamma = np.sqrt((kappa * kappa - detuning * detuning).astype(np.complex128))
        gamma_safe = np.where(np.abs(gamma) < 1e-14, 1e-14 + 0j, gamma)
        sinh_term = np.sinh(gamma * segment_length_m)
        cosh_term = np.cosh(gamma * segment_length_m)
        ratio = sinh_term / gamma_safe
        segment_11 = cosh_term - 1j * detuning * ratio
        segment_12 = -1j * kappa * ratio
        segment_21 = 1j * kappa * ratio
        segment_22 = cosh_term + 1j * detuning * ratio

        next_11 = total_11 * segment_11 + total_12 * segment_21
        next_12 = total_11 * segment_12 + total_12 * segment_22
        next_21 = total_21 * segment_11 + total_22 * segment_21
        next_22 = total_21 * segment_12 + total_22 * segment_22
        total_11, total_12 = next_11, next_12
        total_21, total_22 = next_21, next_22

    reflection_amplitude = total_21 / np.where(
        np.abs(total_11) < 1e-20, 1e-20 + 0j, total_11
    )
    return np.clip(np.abs(reflection_amplitude) ** 2, 1e-10, 0.999999)


def gaussian_kernel(sigma_points: float = 1.0) -> np.ndarray:
    radius = max(2, int(np.ceil(4.0 * sigma_points)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_points) ** 2)
    return kernel / kernel.sum()


def smooth_same(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parabolic_peak(wavelength_nm: np.ndarray, values: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= len(values) - 1:
        return float(wavelength_nm[index]), float(values[index])
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    offset = 0.0 if abs(denominator) < 1e-15 else 0.5 * (left - right) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    step = wavelength_nm[index + 1] - wavelength_nm[index]
    peak_wavelength = wavelength_nm[index] + offset * step
    peak_value = center - 0.25 * (left - right) * offset
    return float(peak_wavelength), float(max(peak_value, 1e-10))


def six_features(wavelength_nm: np.ndarray, reflectivity: np.ndarray) -> list[float]:
    """Main peak and nearest physically meaningful left/right side lobes."""
    extraction_kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    extraction_kernel /= extraction_kernel.sum()
    values = smooth_same(reflectivity, extraction_kernel)
    main_index = int(np.argmax(values))
    local_maxima = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    exclusion_points = max(4, int(round(0.035 / (wavelength_nm[1] - wavelength_nm[0]))))
    left_candidates = local_maxima[local_maxima < main_index - exclusion_points]
    right_candidates = local_maxima[local_maxima > main_index + exclusion_points]

    if left_candidates.size:
        left_index = int(left_candidates[np.argmax(values[left_candidates])])
    else:
        left_index = int(np.argmax(values[: max(main_index - exclusion_points, 1)]))
    if right_candidates.size:
        right_index = int(right_candidates[np.argmax(values[right_candidates])])
    else:
        start = min(main_index + exclusion_points, len(values) - 1)
        right_index = start + int(np.argmax(values[start:]))

    main_wavelength, main_power = parabolic_peak(wavelength_nm, values, main_index)
    left_wavelength, left_power = parabolic_peak(wavelength_nm, values, left_index)
    right_wavelength, right_power = parabolic_peak(wavelength_nm, values, right_index)
    return [
        main_wavelength,
        10.0 * np.log10(main_power),
        left_wavelength,
        10.0 * np.log10(left_power),
        right_wavelength,
        10.0 * np.log10(right_power),
    ]


def make_spectrum(
    wavelength_nm: np.ndarray,
    temperature_c: float,
    strain_ue: float,
    solver: str,
    nuisance: dict[str, float],
    rng: np.random.Generator,
    physics_profile: str = "uniform",
) -> np.ndarray:
    delta_temperature = temperature_c - REFERENCE_TEMPERATURE_C
    if solver == "uniform_cmt":
        effective_photoelastic = 0.2200
        fractional_temperature_coefficient = 6.50e-6
        effective_index = 1.4470
        index_modulation = 2.40e-4
        solver_offset_nm = 0.0
    elif solver == "simplus_like":
        p11, p12, poisson = 0.121, 0.270, 0.17
        effective_index = 1.4465
        effective_photoelastic = (effective_index**2 / 2.0) * (
            p12 - poisson * (p11 + p12)
        )
        fractional_temperature_coefficient = 6.65e-6
        index_modulation = 2.25e-4
        solver_offset_nm = 0.012
    else:
        raise ValueError(f"Unknown solver: {solver}")

    fractional_shift = (
        (1.0 - effective_photoelastic) * strain_ue * 1e-6
        + fractional_temperature_coefficient * delta_temperature
    )
    bragg_nm = (
        REFERENCE_BRAGG_NM * (1.0 + fractional_shift)
        + solver_offset_nm
        + nuisance["wavelength_zero_nm"]
    )
    grating_length_m = 0.010 * nuisance["length_scale"] * (
        1.0 + strain_ue * 1e-6 + 0.55e-6 * delta_temperature
    )
    # Small, physics-directed coupling changes provide a weak second observable
    # in addition to the Bragg shift. They are deliberately kept below 1%.
    coupling_scale = (
        1.0
        - 1.5e-4 * delta_temperature
        - 3.0e-6 * strain_ue
    )
    if solver == "simplus_like":
        coupling_scale *= 1.0 - 7.0e-5 * delta_temperature + 1.0e-6 * strain_ue
    index_modulation *= nuisance["index_modulation_scale"] * coupling_scale

    if physics_profile in {"uniform", "uniform_same_device"}:
        spectrum = uniform_fbg_reflectivity(
            wavelength_nm,
            bragg_nm,
            effective_index,
            grating_length_m,
            index_modulation,
        )
    elif physics_profile in {"strain_gradient", "strain_gradient_same_device"}:
        segment_positions = np.linspace(-0.5, 0.5, 25, dtype=np.float64)
        # The label is the mean strain. A mounting/load-transfer gradient creates
        # an end-to-end variation that grows with applied strain. Temperature is
        # nearly uniform, with only a small longitudinal residual gradient.
        end_to_end_gradient_ue = (
            nuisance["gradient_sign"]
            * nuisance["strain_gradient_scale"]
            * (5.0 + 0.10 * strain_ue)
        )
        curvature_ue = 0.018 * strain_ue
        local_strain_ue = (
            strain_ue
            + end_to_end_gradient_ue * segment_positions
            + curvature_ue
            * (segment_positions**2 - np.mean(segment_positions**2))
        )
        local_temperature_c = (
            temperature_c
            + nuisance["temperature_gradient_C"] * segment_positions
        )
        local_fractional_shift = (
            (1.0 - effective_photoelastic) * local_strain_ue * 1e-6
            + fractional_temperature_coefficient
            * (local_temperature_c - REFERENCE_TEMPERATURE_C)
        )
        local_bragg_nm = (
            REFERENCE_BRAGG_NM * (1.0 + local_fractional_shift)
            + solver_offset_nm
            + nuisance["wavelength_zero_nm"]
        )
        local_modulation = index_modulation * (
            1.0 + 0.003 * segment_positions
        )
        spectrum = segmented_fbg_reflectivity(
            wavelength_nm,
            local_bragg_nm,
            effective_index,
            grating_length_m,
            local_modulation,
        )
    else:
        raise ValueError(f"Unknown physics profile: {physics_profile}")
    spectrum = smooth_same(spectrum, gaussian_kernel(1.0))
    centered_axis = wavelength_nm - wavelength_nm.mean()
    baseline = nuisance["baseline_offset"] + nuisance["baseline_slope"] * centered_axis
    spectrum = nuisance["power_scale"] * spectrum + baseline
    spectrum += rng.normal(0.0, nuisance["noise_std"], size=spectrum.shape)
    return np.clip(spectrum, 1e-8, 0.999999).astype(np.float32)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate_dataset(
    output_dir: Path,
    condition_count: int,
    seed: int,
    physics_profile: str = "uniform",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    design = latin_hypercube(condition_count, 2, rng)
    temperatures = TEMPERATURE_MIN_C + design[:, 0] * (TEMPERATURE_MAX_C - TEMPERATURE_MIN_C)
    strains = STRAIN_MIN_UE + design[:, 1] * (STRAIN_MAX_UE - STRAIN_MIN_UE)

    shuffled_ids = rng.permutation(condition_count)
    train_end = int(round(0.70 * condition_count))
    validation_end = int(round(0.85 * condition_count))
    split_by_id = np.empty(condition_count, dtype="U10")
    split_by_id[shuffled_ids[:train_end]] = "train"
    split_by_id[shuffled_ids[train_end:validation_end]] = "validation"
    split_by_id[shuffled_ids[validation_end:]] = "test"

    wavelength_nm = np.arange(1547.5, 1549.801, 0.005, dtype=np.float64)
    solver_names = ["uniform_cmt", "simplus_like"]
    spectra = np.empty(
        (condition_count, len(solver_names), wavelength_nm.size), dtype=np.float32
    )
    feature_rows: list[list[object]] = []
    condition_rows: list[list[object]] = []

    for condition_index in range(condition_count):
        condition_id = condition_index + 1
        same_device = physics_profile in {
            "uniform_same_device",
            "strain_gradient_same_device",
        }
        nuisance = {
            "wavelength_zero_nm": float(
                rng.normal(0.0, 0.0006 if same_device else 0.0020)
            ),
            "length_scale": float(
                rng.normal(1.0, 0.00015 if same_device else 0.0020)
            ),
            "index_modulation_scale": float(
                rng.normal(1.0, 0.0008 if same_device else 0.0060)
            ),
            "power_scale": float(rng.normal(1.0, 0.004 if same_device else 0.008)),
            "baseline_offset": float(
                rng.normal(0.0, 0.0004 if same_device else 0.0008)
            ),
            "baseline_slope": float(
                rng.normal(0.0, 0.0004 if same_device else 0.0008)
            ),
            "noise_std": float(
                rng.uniform(0.0005, 0.0015)
                if same_device
                else rng.uniform(0.0008, 0.0025)
            ),
            "gradient_sign": 1.0 if same_device else float(rng.choice([-1.0, 1.0])),
            "strain_gradient_scale": float(
                rng.uniform(0.98, 1.02) if same_device else rng.uniform(0.85, 1.15)
            ),
            "temperature_gradient_C": float(
                rng.normal(0.0, 0.01 if same_device else 0.03)
            ),
        }
        condition_rows.append(
            [
                condition_id,
                f"{temperatures[condition_index]:.8f}",
                f"{strains[condition_index]:.8f}",
                split_by_id[condition_index],
                f"{nuisance['wavelength_zero_nm']:.8f}",
                f"{nuisance['length_scale']:.8f}",
                f"{nuisance['index_modulation_scale']:.8f}",
                f"{nuisance['gradient_sign'] * nuisance['strain_gradient_scale'] * (5.0 + 0.10 * strains[condition_index]):.8f}",
                f"{nuisance['temperature_gradient_C']:.8f}",
            ]
        )
        for solver_index, solver_name in enumerate(solver_names):
            sample_rng = np.random.default_rng(seed + condition_id * 17 + solver_index * 1_000_003)
            spectrum = make_spectrum(
                wavelength_nm,
                float(temperatures[condition_index]),
                float(strains[condition_index]),
                solver_name,
                nuisance,
                sample_rng,
                physics_profile,
            )
            spectra[condition_index, solver_index] = spectrum
            extracted = six_features(wavelength_nm, spectrum.astype(np.float64))
            feature_rows.append(
                [
                    f"E{condition_id:05d}_{solver_name}",
                    condition_id,
                    solver_name,
                    split_by_id[condition_index],
                    f"{temperatures[condition_index]:.8f}",
                    f"{strains[condition_index]:.8f}",
                    *[f"{value:.10f}" for value in extracted],
                ]
            )

    write_csv(
        output_dir / "expanded_conditions.csv",
        [
            "condition_id",
            "temperature_C",
            "strain_ue",
            "split",
            "shared_wavelength_zero_nm",
            "shared_length_scale",
            "shared_index_modulation_scale",
            "end_to_end_strain_gradient_ue",
            "end_to_end_temperature_gradient_C",
        ],
        condition_rows,
    )
    write_csv(
        output_dir / "expanded_six_features.csv",
        [
            "sample_id",
            "condition_id",
            "solver_source",
            "split",
            "temperature_C",
            "strain_ue",
            "main_peak_wavelength_nm",
            "main_peak_power_db",
            "left_sidelobe_wavelength_nm",
            "left_sidelobe_power_db",
            "right_sidelobe_wavelength_nm",
            "right_sidelobe_power_db",
        ],
        feature_rows,
    )
    np.savez_compressed(
        output_dir / "expanded_spectra.npz",
        wavelength_nm=wavelength_nm.astype(np.float32),
        reflectivity=spectra,
        temperature_C=temperatures.astype(np.float32),
        strain_ue=strains.astype(np.float32),
        split=split_by_id,
        solver_source=np.asarray(solver_names),
        condition_id=np.arange(1, condition_count + 1, dtype=np.int32),
    )
    metadata: dict[str, object] = {
        "dataset_name": "expanded_uniform_fbg_two_solver_v1",
        "seed": seed,
        "independent_conditions": condition_count,
        "solver_count": len(solver_names),
        "spectra_count": int(condition_count * len(solver_names)),
        "wavelength_points": int(wavelength_nm.size),
        "wavelength_range_nm": [float(wavelength_nm[0]), float(wavelength_nm[-1])],
        "wavelength_step_nm": float(wavelength_nm[1] - wavelength_nm[0]),
        "temperature_range_C": [TEMPERATURE_MIN_C, TEMPERATURE_MAX_C],
        "strain_range_ue": [STRAIN_MIN_UE, STRAIN_MAX_UE],
        "sampling": "Latin hypercube; continuous temperature and strain",
        "physics_profile": physics_profile,
        "split_rule": "70/15/15 by condition_id; paired solvers always share a split",
        "solvers": solver_names,
        "important_limitation": (
            "Synthetic data cannot replace experimental strain validation. "
            "The 40 real files currently validate temperature only."
        ),
    }
    (output_dir / "expanded_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--physics-profile",
        choices=[
            "uniform",
            "uniform_same_device",
            "strain_gradient",
            "strain_gradient_same_device",
        ],
        default="uniform",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_expanded_v1",
    )
    args = parser.parse_args()
    if args.conditions < 100:
        parser.error("--conditions must be at least 100")
    metadata = generate_dataset(
        args.output_dir, args.conditions, args.seed, args.physics_profile
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
