from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import FEATURE_COLUMNS, extract_six_features, spectral_columns_and_axis
from .modeling import TARGET_COLUMNS
from .modeling import make_split_indices
from .config import ModelConfig


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]


@dataclass
class HybridConfig:
    strain_levels_ue: list[float] = field(
        default_factory=lambda: [float(value) for value in range(0, 1000, 100)]
    )
    replicates_per_channel: int = 2
    output_wavelength_min_nm: float = 1536.5
    output_wavelength_max_nm: float = 1555.0
    wavelength_step_nm: float = 0.02
    template_half_width_nm: float = 3.2
    photoelastic_coefficient: float = 0.22
    strain_gradient_fraction_mean: float = 0.035
    strain_gradient_fraction_std: float = 0.010
    strain_jitter_std_ue: float = 2.0
    temperature_jitter_std_c: float = 0.05
    wavelength_jitter_std_nm: float = 0.003
    residual_scale_min: float = 0.65
    residual_scale_max: float = 1.15
    power_offset_std_db: float = 0.12
    white_noise_std_db: float = 0.04
    random_seed: int = 20261012

    @classmethod
    def from_json(cls, path: str | Path) -> "HybridConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw.get("hybrid", raw))


@dataclass
class ChannelTemplate:
    channel: str
    relative_wavelength_nm: np.ndarray
    template_db: np.ndarray
    residuals_db: np.ndarray
    temperatures_c: np.ndarray
    temperature_sensitivity_nm_per_c: float
    temperature_intercept_nm: float
    reference_center_nm: float


def _quadratic_peak(
    wavelength_nm: np.ndarray, power_db: np.ndarray
) -> tuple[float, float]:
    index = int(np.argmax(power_db))
    if index == 0 or index == len(power_db) - 1:
        return float(wavelength_nm[index]), float(power_db[index])
    left, center, right = power_db[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(wavelength_nm[index]), float(center)
    offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    step = float(wavelength_nm[index + 1] - wavelength_nm[index])
    peak_nm = wavelength_nm[index] + offset * step
    peak_db = center - 0.25 * (left - right) * offset
    return float(peak_nm), float(peak_db)


def load_real_files(input_dir: str | Path) -> list[tuple[Path, float, pd.DataFrame]]:
    input_dir = Path(input_dir)
    records = []
    for path in input_dir.glob("*.list"):
        try:
            temperature = float(path.stem)
        except ValueError as exc:
            raise ValueError(f"Non-numeric temperature label: {path.name}") from exc
        frame = pd.read_csv(path, sep="\t")
        missing = {"Wave", *CHANNELS}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        records.append((path, temperature, frame))
    if not records:
        raise FileNotFoundError(f"No .list files found in {input_dir}")
    records.sort(key=lambda item: item[1])
    reference_wave = records[0][2]["Wave"].to_numpy(dtype=float)
    for path, _, frame in records[1:]:
        if not np.allclose(frame["Wave"].to_numpy(dtype=float), reference_wave):
            raise ValueError(f"Wavelength grid differs in {path.name}")
    return records


def build_channel_templates(
    records: list[tuple[Path, float, pd.DataFrame]], config: HybridConfig
) -> dict[str, ChannelTemplate]:
    relative_axis = np.arange(
        -config.template_half_width_nm,
        config.template_half_width_nm + config.wavelength_step_nm / 2.0,
        config.wavelength_step_nm,
    )
    templates: dict[str, ChannelTemplate] = {}
    for channel in CHANNELS:
        aligned_rows = []
        centers = []
        temperatures = []
        for _, temperature, frame in records:
            wavelength = frame["Wave"].to_numpy(dtype=float)
            power_db = frame[channel].to_numpy(dtype=float)
            center_nm, _ = _quadratic_peak(wavelength, power_db)
            baseline = float(np.median(power_db))
            aligned = np.interp(
                center_nm + relative_axis,
                wavelength,
                power_db,
                left=baseline,
                right=baseline,
            )
            aligned_rows.append(aligned)
            centers.append(center_nm)
            temperatures.append(temperature)
        aligned_array = np.asarray(aligned_rows)
        smooth_rows = np.asarray(
            [savgol_filter(row, 11, 2, mode="interp") for row in aligned_array]
        )
        template_db = np.median(smooth_rows, axis=0)
        residuals_db = aligned_array - template_db[None, :]
        calibration = LinearRegression().fit(
            np.asarray(temperatures)[:, None], np.asarray(centers)
        )
        templates[channel] = ChannelTemplate(
            channel=channel,
            relative_wavelength_nm=relative_axis,
            template_db=template_db,
            residuals_db=residuals_db,
            temperatures_c=np.asarray(temperatures, dtype=float),
            temperature_sensitivity_nm_per_c=float(calibration.coef_[0]),
            temperature_intercept_nm=float(calibration.intercept_),
            reference_center_nm=float(np.median(centers)),
        )
    return templates


def strain_sensitivity_nm_per_ue(
    bragg_wavelength_nm: float, photoelastic_coefficient: float
) -> float:
    return bragg_wavelength_nm * (1.0 - photoelastic_coefficient) * 1e-6


def _broaden_template(
    template_db: np.ndarray, sigma_nm: float, step_nm: float
) -> np.ndarray:
    linear_power = 10.0 ** (template_db / 10.0)
    if sigma_nm > 0.0:
        linear_power = gaussian_filter1d(
            linear_power, sigma=sigma_nm / step_nm, mode="nearest"
        )
    return 10.0 * np.log10(np.clip(linear_power, 1e-8, None))


def _output_axis(config: HybridConfig) -> np.ndarray:
    return np.arange(
        config.output_wavelength_min_nm,
        config.output_wavelength_max_nm + config.wavelength_step_nm / 2.0,
        config.wavelength_step_nm,
    )


def generate_hybrid_spectra(
    records: list[tuple[Path, float, pd.DataFrame]],
    templates: dict[str, ChannelTemplate],
    config: HybridConfig,
) -> pd.DataFrame:
    wavelengths = _output_axis(config)
    temperatures = np.asarray([record[1] for record in records], dtype=float)
    sample_count = (
        len(temperatures)
        * len(config.strain_levels_ue)
        * len(CHANNELS)
        * config.replicates_per_channel
    )
    spectra = np.empty((sample_count, len(wavelengths)), dtype=np.float32)
    metadata: list[dict[str, object]] = []
    rng = np.random.default_rng(config.random_seed)
    row = 0
    condition_id = 0
    for temperature_nominal in temperatures:
        for strain_nominal in config.strain_levels_ue:
            condition_id += 1
            for replicate in range(1, config.replicates_per_channel + 1):
                # The four channels are treated as simultaneous measurements,
                # so they share one physical temperature and strain state.
                temperature_actual = temperature_nominal + rng.normal(
                    0.0, config.temperature_jitter_std_c
                )
                strain_actual = max(
                    0.0,
                    strain_nominal + rng.normal(0.0, config.strain_jitter_std_ue),
                )
                gradient_fraction = max(
                    0.0,
                    rng.normal(
                        config.strain_gradient_fraction_mean,
                        config.strain_gradient_fraction_std,
                    ),
                )
                for channel in CHANNELS:
                    template = templates[channel]
                    strain_coefficient = strain_sensitivity_nm_per_ue(
                        template.reference_center_nm, config.photoelastic_coefficient
                    )
                    temperature_center = (
                        template.temperature_intercept_nm
                        + template.temperature_sensitivity_nm_per_c * temperature_actual
                    )
                    strain_shift = strain_coefficient * strain_actual
                    measured_center = (
                        temperature_center
                        + strain_shift
                        + rng.normal(0.0, config.wavelength_jitter_std_nm)
                    )
                    gradient_sigma_nm = (
                        strain_coefficient * strain_actual * gradient_fraction
                    )
                    shaped_db = _broaden_template(
                        template.template_db,
                        gradient_sigma_nm,
                        config.wavelength_step_nm,
                    )
                    residual_index = int(rng.integers(0, len(template.residuals_db)))
                    residual_scale = rng.uniform(
                        config.residual_scale_min, config.residual_scale_max
                    )
                    shaped_db = (
                        shaped_db
                        + residual_scale * template.residuals_db[residual_index]
                        + rng.normal(0.0, config.power_offset_std_db)
                    )
                    baseline = float(np.median(shaped_db))
                    output_db = np.interp(
                        wavelengths - measured_center,
                        template.relative_wavelength_nm,
                        shaped_db,
                        left=baseline,
                        right=baseline,
                    )
                    output_db += rng.normal(
                        0.0, config.white_noise_std_db, size=len(output_db)
                    )
                    spectra[row] = 10.0 ** (output_db / 10.0)
                    metadata.append(
                        {
                            "dataset": "semi_empirical_digital_twin",
                            "spectrum_id": f"H{row + 1:05d}",
                            "condition_id": condition_id,
                            "channel": channel,
                            "replicate": replicate,
                            "strain_nominal_ue": float(strain_nominal),
                            "temperature_nominal_C": float(temperature_nominal),
                            "strain_actual_ue": float(strain_actual),
                            "temperature_actual_C": float(temperature_actual),
                            "temperature_center_nm": float(temperature_center),
                            "strain_shift_nm": float(strain_shift),
                            "strain_gradient_fraction": float(gradient_fraction),
                            "strain_gradient_sigma_nm": float(gradient_sigma_nm),
                            "bragg_center_model_nm": float(measured_center),
                            "residual_source_index": residual_index,
                        }
                    )
                    row += 1
    spectral_columns = [f"R_{value:.3f}nm" for value in wavelengths]
    return pd.concat(
        [pd.DataFrame(metadata), pd.DataFrame(spectra, columns=spectral_columns)],
        axis=1,
    )


def real_anchor_spectra(
    records: list[tuple[Path, float, pd.DataFrame]], config: HybridConfig
) -> pd.DataFrame:
    wavelengths = _output_axis(config)
    rows = []
    spectra = []
    spectrum_id = 0
    for condition_id, (path, temperature, frame) in enumerate(records, start=1):
        source_wave = frame["Wave"].to_numpy(dtype=float)
        for channel in CHANNELS:
            spectrum_id += 1
            source_db = frame[channel].to_numpy(dtype=float)
            baseline = float(np.median(source_db))
            output_db = np.interp(
                wavelengths, source_wave, source_db, left=baseline, right=baseline
            )
            spectra.append(10.0 ** (output_db / 10.0))
            rows.append(
                {
                    "dataset": "real_temperature_anchor",
                    "spectrum_id": f"R{spectrum_id:04d}",
                    "condition_id": condition_id,
                    "channel": channel,
                    "replicate": 1,
                    "source_file": path.name,
                    "strain_nominal_ue": 0.0,
                    "temperature_nominal_C": temperature,
                    "strain_actual_ue": 0.0,
                    "temperature_actual_C": temperature,
                }
            )
    spectral_columns = [f"R_{value:.3f}nm" for value in wavelengths]
    return pd.concat(
        [pd.DataFrame(rows), pd.DataFrame(spectra, columns=spectral_columns)], axis=1
    )


def combine_channel_features(
    channel_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    base_features = [*FEATURE_COLUMNS, "halfmax_width_nm"]
    required = {"condition_id", "replicate", "channel", *base_features, *TARGET_COLUMNS}
    missing = required.difference(channel_features.columns)
    if missing:
        raise ValueError(f"Missing channel feature columns: {sorted(missing)}")
    key_columns = ["condition_id", "replicate"]
    counts = channel_features.groupby(key_columns)["channel"].nunique()
    if not (counts == len(CHANNELS)).all():
        raise ValueError("Each fused sample must contain all four channels.")
    pivot = channel_features.pivot(
        index=key_columns, columns="channel", values=base_features
    )
    pivot.columns = [f"{channel}_{feature}" for feature, channel in pivot.columns]
    pivot = pivot.reset_index()
    feature_columns = [
        f"{channel}_{feature}" for channel in CHANNELS for feature in base_features
    ]
    metadata_candidates = [
        "dataset",
        "strain_nominal_ue",
        "temperature_nominal_C",
        "strain_actual_ue",
        "temperature_actual_C",
    ]
    metadata_columns = [
        column for column in metadata_candidates if column in channel_features.columns
    ]
    metadata = channel_features.groupby(key_columns, as_index=False)[metadata_columns].first()
    fused = metadata.merge(pivot, on=key_columns, how="inner")
    for channel in CHANNELS:
        center = fused[f"{channel}_center_peak_wavelength_nm"]
        fused[f"{channel}_left_offset_nm"] = (
            center - fused[f"{channel}_left_sidelobe_wavelength_nm"]
        )
        fused[f"{channel}_right_offset_nm"] = (
            fused[f"{channel}_right_sidelobe_wavelength_nm"] - center
        )
        fused[f"{channel}_sidelobe_power_balance_db"] = (
            fused[f"{channel}_right_sidelobe_power_db"]
            - fused[f"{channel}_left_sidelobe_power_db"]
        )
        feature_columns.extend(
            [
                f"{channel}_left_offset_nm",
                f"{channel}_right_offset_nm",
                f"{channel}_sidelobe_power_balance_db",
            ]
        )
    fused.insert(0, "spectrum_id", [f"M{index + 1:05d}" for index in range(len(fused))])
    return fused, feature_columns


def evaluate_classical_on_real_anchor(
    features: pd.DataFrame,
    model_root: str | Path,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_root = Path(model_root)
    feature_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    x = features[feature_columns].to_numpy(dtype=float)
    actual = features[TARGET_COLUMNS].to_numpy(dtype=float)
    prediction_frames = []
    metric_rows = []
    for split_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        for model_path in sorted(split_dir.glob("*.joblib")):
            model = joblib.load(model_path)
            predicted = np.asarray(model.predict(x))
            detail_columns = [
                column
                for column in [
                    "spectrum_id",
                    "condition_id",
                    "channel",
                    "temperature_actual_C",
                ]
                if column in features.columns
            ]
            details = features[detail_columns].copy()
            details.insert(0, "split_method", split_dir.name)
            details.insert(1, "model", model_path.stem)
            details["strain_reference_ue"] = actual[:, 0]
            details["strain_predicted_ue"] = predicted[:, 0]
            details["temperature_predicted_C"] = predicted[:, 1]
            prediction_frames.append(details)
            channels = ["all", *CHANNELS] if "channel" in features.columns else ["all"]
            for channel in channels:
                mask = np.ones(len(features), dtype=bool)
                if channel != "all":
                    mask = features["channel"].to_numpy() == channel
                for target_index, target in enumerate(["strain_ue", "temperature_c"]):
                    true_values = actual[mask, target_index]
                    predicted_values = predicted[mask, target_index]
                    metric_rows.append(
                        {
                            "split_method": split_dir.name,
                            "model": model_path.stem,
                            "channel": channel,
                            "target": target,
                            "r2": (
                                float(r2_score(true_values, predicted_values))
                                if np.unique(true_values).size > 1
                                else float("nan")
                            ),
                            "mae": float(mean_absolute_error(true_values, predicted_values)),
                            "rmse": float(
                                np.sqrt(mean_squared_error(true_values, predicted_values))
                            ),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def analytical_multichannel_predictions(
    features: pd.DataFrame,
    templates: dict[str, ChannelTemplate],
    config: HybridConfig,
) -> pd.DataFrame:
    temperature_coefficients = np.asarray(
        [templates[channel].temperature_sensitivity_nm_per_c for channel in CHANNELS]
    )
    strain_coefficients = np.asarray(
        [
            strain_sensitivity_nm_per_ue(
                templates[channel].reference_center_nm,
                config.photoelastic_coefficient,
            )
            for channel in CHANNELS
        ]
    )
    intercepts = np.asarray(
        [templates[channel].temperature_intercept_nm for channel in CHANNELS]
    )
    design = np.column_stack([strain_coefficients, temperature_coefficients])
    centers = features[
        [f"{channel}_center_peak_wavelength_nm" for channel in CHANNELS]
    ].to_numpy(dtype=float)
    solution = np.linalg.lstsq(design, (centers - intercepts).T, rcond=None)[0].T
    result = features[
        [
            column
            for column in [
                "spectrum_id",
                "condition_id",
                "replicate",
                "strain_actual_ue",
                "temperature_actual_C",
            ]
            if column in features.columns
        ]
    ].copy()
    result["strain_predicted_ue"] = solution[:, 0]
    result["temperature_predicted_C"] = solution[:, 1]
    return result


def prediction_metrics(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for target, actual_column, predicted_column in [
        ("strain_ue", "strain_actual_ue", "strain_predicted_ue"),
        ("temperature_c", "temperature_actual_C", "temperature_predicted_C"),
    ]:
        actual = predictions[actual_column].to_numpy(dtype=float)
        predicted = predictions[predicted_column].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "target": target,
                "r2": (
                    float(r2_score(actual, predicted))
                    if np.unique(actual).size > 1
                    else float("nan")
                ),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            }
        )
    return pd.DataFrame(rows)


def multichannel_shape_columns() -> list[str]:
    names = []
    for channel in CHANNELS:
        names.extend(
            [
                f"{channel}_center_peak_power_db",
                f"{channel}_left_sidelobe_power_db",
                f"{channel}_right_sidelobe_power_db",
                f"{channel}_halfmax_width_nm",
                f"{channel}_left_offset_nm",
                f"{channel}_right_offset_nm",
                f"{channel}_sidelobe_power_balance_db",
            ]
        )
    return names


def _temperature_after_strain_correction(
    features: pd.DataFrame,
    strain_prediction: np.ndarray,
    templates: dict[str, ChannelTemplate],
    config: HybridConfig,
) -> np.ndarray:
    channel_temperatures = []
    for channel in CHANNELS:
        template = templates[channel]
        center = features[f"{channel}_center_peak_wavelength_nm"].to_numpy(dtype=float)
        strain_coefficient = strain_sensitivity_nm_per_ue(
            template.reference_center_nm, config.photoelastic_coefficient
        )
        estimate = (
            center
            - template.temperature_intercept_nm
            - strain_coefficient * strain_prediction
        ) / template.temperature_sensitivity_nm_per_c
        channel_temperatures.append(estimate)
    return np.median(np.column_stack(channel_temperatures), axis=1)


def train_physics_guided_models(
    features: pd.DataFrame,
    templates: dict[str, ChannelTemplate],
    hybrid_config: HybridConfig,
    model_config: ModelConfig,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_columns = multichannel_shape_columns()
    x = features[shape_columns].to_numpy(dtype=float)
    strain = features["strain_actual_ue"].to_numpy(dtype=float)
    splitters = make_split_indices(features, model_config.random_seed)
    metric_frames = []
    prediction_frames = []
    for split_name, partitions in splitters.items():
        model = RandomForestRegressor(
            n_estimators=model_config.n_estimators,
            random_state=model_config.random_seed,
            n_jobs=-1,
        )
        model.fit(x[partitions["train"]], strain[partitions["train"]])
        joblib.dump(model, output_dir / f"physics_guided_{split_name}.joblib")
        for partition_name, indices in partitions.items():
            strain_prediction = np.clip(
                model.predict(x[indices]),
                min(hybrid_config.strain_levels_ue),
                max(hybrid_config.strain_levels_ue),
            )
            subset = features.iloc[indices]
            temperature_prediction = _temperature_after_strain_correction(
                subset, strain_prediction, templates, hybrid_config
            )
            prediction = subset[
                [
                    "spectrum_id",
                    "condition_id",
                    "replicate",
                    "strain_actual_ue",
                    "temperature_actual_C",
                ]
            ].copy()
            prediction.insert(0, "split_method", split_name)
            prediction.insert(1, "partition", partition_name)
            prediction["strain_predicted_ue"] = strain_prediction
            prediction["temperature_predicted_C"] = temperature_prediction
            prediction_frames.append(prediction)
            local_metrics = prediction_metrics(prediction, "physics_guided_two_stage")
            local_metrics.insert(0, "split_method", split_name)
            local_metrics.insert(1, "partition", partition_name)
            metric_frames.append(local_metrics)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(
        prediction_frames, ignore_index=True
    )


def evaluate_physics_guided_on_anchor(
    features: pd.DataFrame,
    templates: dict[str, ChannelTemplate],
    config: HybridConfig,
    model_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = Path(model_dir)
    x = features[multichannel_shape_columns()].to_numpy(dtype=float)
    metric_frames = []
    prediction_frames = []
    for model_path in sorted(model_dir.glob("physics_guided_*.joblib")):
        split_name = model_path.stem.replace("physics_guided_", "")
        model = joblib.load(model_path)
        strain_prediction = np.clip(
            model.predict(x),
            min(config.strain_levels_ue),
            max(config.strain_levels_ue),
        )
        temperature_prediction = _temperature_after_strain_correction(
            features, strain_prediction, templates, config
        )
        prediction = features[
            ["spectrum_id", "condition_id", "strain_actual_ue", "temperature_actual_C"]
        ].copy()
        prediction.insert(0, "split_method", split_name)
        prediction["strain_predicted_ue"] = strain_prediction
        prediction["temperature_predicted_C"] = temperature_prediction
        prediction_frames.append(prediction)
        local_metrics = prediction_metrics(prediction, "physics_guided_two_stage")
        local_metrics.insert(0, "split_method", split_name)
        metric_frames.append(local_metrics)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(
        prediction_frames, ignore_index=True
    )


def sensitivity_geometry(
    templates: dict[str, ChannelTemplate], config: HybridConfig
) -> dict[str, float]:
    temperature_vector = np.asarray(
        [templates[channel].temperature_sensitivity_nm_per_c for channel in CHANNELS]
    )
    strain_vector = np.asarray(
        [
            strain_sensitivity_nm_per_ue(
                templates[channel].reference_center_nm,
                config.photoelastic_coefficient,
            )
            for channel in CHANNELS
        ]
    )
    cosine = float(
        np.dot(temperature_vector, strain_vector)
        / (np.linalg.norm(temperature_vector) * np.linalg.norm(strain_vector))
    )
    return {
        "temperature_strain_sensitivity_cosine": cosine,
        "separation_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
    }


def plot_hybrid_spectra(spectra: pd.DataFrame, output_path: str | Path) -> None:
    columns, wavelengths = spectral_columns_and_axis(spectra)
    temperature = float(spectra["temperature_nominal_C"].median())
    available = np.sort(spectra["temperature_nominal_C"].unique())
    temperature = float(available[np.argmin(np.abs(available - temperature))])
    strain_levels = np.sort(spectra["strain_nominal_ue"].unique())
    selected_strains = [strain_levels[0], strain_levels[len(strain_levels) // 2], strain_levels[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, channel in zip(axes.flat, CHANNELS):
        for strain in selected_strains:
            row = spectra[
                (spectra["channel"] == channel)
                & (spectra["temperature_nominal_C"] == temperature)
                & (spectra["strain_nominal_ue"] == strain)
            ].iloc[0]
            axis.plot(
                wavelengths,
                10.0 * np.log10(np.clip(row[columns].to_numpy(dtype=float), 1e-8, None)),
                label=f"{strain:.0f} µε",
            )
        axis.set_title(channel)
        axis.set_ylabel("Power (dB)")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes[-1, :]:
        axis.set_xlabel("Wavelength (nm)")
    fig.suptitle(f"Semi-empirical spectra at {temperature:g} °C")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_cnn_on_real_anchor(
    spectra: pd.DataFrame, checkpoint_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    from .cnn import SpectrumCNN

    checkpoint_dir = Path(checkpoint_dir)
    spectral_columns, wavelengths = spectral_columns_and_axis(spectra)
    raw_x = spectra[spectral_columns].to_numpy(dtype=np.float32)
    raw_x = np.clip(10.0 * np.log10(np.clip(raw_x, 1e-8, None)), -60.0, 0.0)
    actual = spectra[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows = []
    prediction_frames = []
    for checkpoint_path in sorted(checkpoint_dir.glob("spectrum_cnn_*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not np.allclose(wavelengths, checkpoint["wavelengths_nm"]):
            raise ValueError(f"Wavelength grid differs for {checkpoint_path.name}")
        config = checkpoint["cnn_config"]
        model = SpectrumCNN(dropout=float(config["dropout"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        standardized_x = (raw_x - checkpoint["x_mean"]) / checkpoint["x_std"]
        with torch.no_grad():
            prediction_standard = model(
                torch.from_numpy(standardized_x[:, None, :]).float().to(device)
            ).cpu().numpy()
        predicted = (
            prediction_standard * checkpoint["y_std"] + checkpoint["y_mean"]
        )
        split_method = str(checkpoint["split_method"])
        details = spectra[
            ["spectrum_id", "condition_id", "channel", "temperature_actual_C"]
        ].copy()
        details.insert(0, "split_method", split_method)
        details.insert(1, "model", "spectrum_cnn")
        details["strain_reference_ue"] = actual[:, 0]
        details["strain_predicted_ue"] = predicted[:, 0]
        details["temperature_predicted_C"] = predicted[:, 1]
        prediction_frames.append(details)
        for channel in ["all", *CHANNELS]:
            mask = np.ones(len(spectra), dtype=bool)
            if channel != "all":
                mask = spectra["channel"].to_numpy() == channel
            for target_index, target in enumerate(["strain_ue", "temperature_c"]):
                true_values = actual[mask, target_index]
                predicted_values = predicted[mask, target_index]
                metric_rows.append(
                    {
                        "split_method": split_method,
                        "model": "spectrum_cnn",
                        "channel": channel,
                        "target": target,
                        "r2": (
                            float(r2_score(true_values, predicted_values))
                            if np.unique(true_values).size > 1
                            else float("nan")
                        ),
                        "mae": float(mean_absolute_error(true_values, predicted_values)),
                        "rmse": float(
                            np.sqrt(mean_squared_error(true_values, predicted_values))
                        ),
                    }
                )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def template_summary(
    templates: dict[str, ChannelTemplate], config: HybridConfig
) -> list[dict[str, float | str]]:
    return [
        {
            "channel": channel,
            "temperature_sensitivity_pm_per_C": (
                template.temperature_sensitivity_nm_per_c * 1000.0
            ),
            "temperature_intercept_nm": template.temperature_intercept_nm,
            "reference_center_nm": template.reference_center_nm,
            "strain_sensitivity_pm_per_ue": (
                strain_sensitivity_nm_per_ue(
                    template.reference_center_nm, config.photoelastic_coefficient
                )
                * 1000.0
            ),
        }
        for channel, template in templates.items()
    ]


def config_as_dict(config: HybridConfig) -> dict[str, object]:
    return asdict(config)
