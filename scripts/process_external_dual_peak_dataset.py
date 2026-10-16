"""Process and visualize the public dual-peak FBG temperature dataset.

Dataset:
    Serif Ali Sadik (2025), Optical Spectrum Dataset for Dual-Peak
    Temperature Measurements, Mendeley Data, version 2.
    DOI: 10.17632/fymrd8js55.2 (CC BY 4.0)

The script preserves the raw files, creates standardized machine-readable
tables, evaluates a leave-one-out linear temperature calibration, and exports
publication-ready PNG/PDF/SVG figures.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
RAW = (
    ROOT
    / "data"
    / "external_mendeley_dual_peak"
    / "raw"
    / "extracted"
    / "Optical Spectrum Dataset for Dual-Peak Temperature"
)
PROCESSED = ROOT / "data" / "external_mendeley_dual_peak" / "processed"
OUTPUT = ROOT / "outputs" / "external_mendeley_dual_peak"
FIGURES = ROOT / "figures" / "publication"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#6BB36B",
    "red": "#B64342",
    "teal": "#42949E",
    "violet": "#8064A2",
    "neutral": "#CFCECE",
    "neutral_dark": "#767676",
    "black": "#272727",
    "grid": "#D9D9D9",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        color=PALETTE["black"],
    )


def light_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=PALETTE["grid"], linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.25)
    paths: list[Path] = []
    for extension in ("png", "pdf", "svg"):
        path = FIGURES / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        paths.append(path)
    plt.close(fig)
    return paths


def read_metadata() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with (RAW / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            rows.append(
                {
                    "file_id": int(row["File"]),
                    "measurement": int(row["Measurement"]),
                    "temperature_sensing_c": float(row["Temp1"]),
                    "temperature_reference_c": float(row["Temp2"]),
                    "peak_sensing_nm": float(row["Peak1"]),
                    "peak_reference_nm": float(row["Peak2"]),
                }
            )
    return sorted(rows, key=lambda row: float(row["temperature_sensing_c"]))


def read_osa_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wavelength: list[float] = []
    power: list[float] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                candidate_wavelength = float(parts[0])
                candidate_power = float(parts[1])
            except ValueError:
                continue
            if 1500.0 <= candidate_wavelength <= 1600.0:
                wavelength.append(candidate_wavelength)
                power.append(candidate_power)
    if len(wavelength) < 10:
        raise ValueError(f"No OSA spectrum found in {path}")
    return np.asarray(wavelength, dtype=float), np.asarray(power, dtype=float)


def normalize_and_smooth(power: np.ndarray) -> np.ndarray:
    smooth = savgol_filter(power, window_length=21, polyorder=3, mode="interp")
    baseline = float(np.percentile(smooth, 1.0))
    scale = max(float(smooth.max() - baseline), 1e-15)
    return np.clip((smooth - baseline) / scale, 0.0, 1.05)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def load_dataset() -> dict[str, np.ndarray | list[dict[str, float | int]]]:
    metadata = read_metadata()
    spectra: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    wavelength_reference: np.ndarray | None = None
    for row in metadata:
        wavelength, power = read_osa_spectrum(RAW / f"DATA{int(row['file_id'])}.TXT")
        if wavelength_reference is None:
            wavelength_reference = wavelength
        elif wavelength.shape != wavelength_reference.shape or not np.allclose(wavelength, wavelength_reference):
            raise ValueError("The 17 OSA files do not share one wavelength grid")
        spectra.append(power)
        normalized.append(normalize_and_smooth(power))
    assert wavelength_reference is not None
    return {
        "metadata": metadata,
        "wavelength_nm": wavelength_reference,
        "power_w": np.vstack(spectra),
        "normalized_power": np.vstack(normalized),
        "temperature_c": np.asarray([row["temperature_sensing_c"] for row in metadata], dtype=float),
        "peak_sensing_nm": np.asarray([row["peak_sensing_nm"] for row in metadata], dtype=float),
        "peak_reference_nm": np.asarray([row["peak_reference_nm"] for row in metadata], dtype=float),
        "file_id": np.asarray([row["file_id"] for row in metadata], dtype=int),
    }


def leave_one_out_temperature_prediction(peak_nm: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(temperature_c)
    for index in range(len(temperature_c)):
        keep = np.arange(len(temperature_c)) != index
        slope, intercept = np.polyfit(peak_nm[keep], temperature_c[keep], 1)
        predictions[index] = slope * peak_nm[index] + intercept
    return predictions


def save_processed_data(data: dict[str, object]) -> dict[str, float]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    wavelength = np.asarray(data["wavelength_nm"], dtype=float)
    power = np.asarray(data["power_w"], dtype=float)
    normalized = np.asarray(data["normalized_power"], dtype=float)
    temperature = np.asarray(data["temperature_c"], dtype=float)
    peak_sensing = np.asarray(data["peak_sensing_nm"], dtype=float)
    peak_reference = np.asarray(data["peak_reference_nm"], dtype=float)
    file_id = np.asarray(data["file_id"], dtype=int)

    np.savez_compressed(
        PROCESSED / "external_dual_peak_spectra.npz",
        wavelength_nm=wavelength,
        power_w=power,
        normalized_power=normalized,
        temperature_c=temperature,
        peak_sensing_nm=peak_sensing,
        peak_reference_nm=peak_reference,
        file_id=file_id,
    )

    predictions = leave_one_out_temperature_prediction(peak_sensing, temperature)
    residuals = predictions - temperature
    shift_pm = (peak_sensing - peak_sensing[0]) * 1000.0
    reference_shift_pm = (peak_reference - np.median(peak_reference)) * 1000.0
    separation_pm = np.abs(peak_sensing - peak_reference) * 1000.0

    metadata_path = PROCESSED / "external_dual_peak_metadata.csv"
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_id",
                "temperature_sensing_c",
                "temperature_reference_c",
                "peak_sensing_nm",
                "peak_reference_nm",
                "sensing_peak_shift_pm",
                "reference_peak_shift_from_median_pm",
                "peak_separation_pm",
                "loocv_predicted_temperature_c",
                "loocv_error_c",
            ]
        )
        for values in zip(
            file_id,
            temperature,
            np.full_like(temperature, 20.0),
            peak_sensing,
            peak_reference,
            shift_pm,
            reference_shift_pm,
            separation_pm,
            predictions,
            residuals,
        ):
            writer.writerow(values)

    with gzip.open(PROCESSED / "external_dual_peak_spectra_long.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "temperature_c", "wavelength_nm", "power_w", "normalized_power"])
        for row_index, identifier in enumerate(file_id):
            for column_index, wavelength_nm in enumerate(wavelength):
                writer.writerow(
                    [
                        int(identifier),
                        float(temperature[row_index]),
                        float(wavelength_nm),
                        float(power[row_index, column_index]),
                        float(normalized[row_index, column_index]),
                    ]
                )

    slope_nm_per_c, intercept_nm = np.polyfit(temperature, peak_sensing, 1)
    fitted_peak = slope_nm_per_c * temperature + intercept_nm
    metrics = {
        "n_spectra": int(len(temperature)),
        "n_wavelength_points": int(len(wavelength)),
        "temperature_min_c": float(temperature.min()),
        "temperature_max_c": float(temperature.max()),
        "wavelength_min_nm": float(wavelength.min()),
        "wavelength_max_nm": float(wavelength.max()),
        "sensing_peak_sensitivity_pm_per_c": float(slope_nm_per_c * 1000.0),
        "sensing_peak_fit_r2": float(r2_score(peak_sensing, fitted_peak)),
        "reference_peak_std_pm": float(np.std(peak_reference, ddof=1) * 1000.0),
        "minimum_peak_separation_pm": float(separation_pm.min()),
        "loocv_temperature_mae_c": float(np.mean(np.abs(residuals))),
        "loocv_temperature_rmse_c": float(np.sqrt(np.mean(residuals**2))),
        "loocv_temperature_r2": float(r2_score(temperature, predictions)),
        "source_doi": "10.17632/fymrd8js55.2",
        "license": "CC BY 4.0",
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def figure_external_characterization(data: dict[str, object], metrics: dict[str, float]) -> list[Path]:
    wavelength = np.asarray(data["wavelength_nm"], dtype=float)
    normalized = np.asarray(data["normalized_power"], dtype=float)
    temperature = np.asarray(data["temperature_c"], dtype=float)
    peak_sensing = np.asarray(data["peak_sensing_nm"], dtype=float)
    peak_reference = np.asarray(data["peak_reference_nm"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    selected_temperatures = [2.0, 14.0, 20.0, 29.0, 38.0, 50.0]
    selected_colors = [
        PALETTE["teal"],
        PALETTE["green"],
        PALETTE["blue_secondary"],
        PALETTE["violet"],
        PALETTE["red"],
        PALETTE["blue_main"],
    ]
    for target, color in zip(selected_temperatures, selected_colors):
        index = int(np.argmin(np.abs(temperature - target)))
        axes[0, 0].plot(
            wavelength,
            normalized[index],
            color=color,
            linewidth=1.7,
            label=f"{temperature[index]:.0f} °C",
        )
    axes[0, 0].set_xlim(1549.20, 1550.02)
    axes[0, 0].set_ylim(-0.03, 1.06)
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Normalized power")
    axes[0, 0].set_title("Representative measured spectra")
    axes[0, 0].legend(ncol=2, loc="upper left")
    axes[0, 0].ticklabel_format(axis="x", style="plain", useOffset=False)
    light_grid(axes[0, 0])

    mesh = axes[0, 1].pcolormesh(
        wavelength,
        temperature,
        normalized,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    axes[0, 1].plot(peak_sensing, temperature, color="white", linewidth=1.6, label="Sensing peak")
    axes[0, 1].plot(peak_reference, temperature, color=PALETTE["red"], linewidth=1.6, label="Reference peak")
    axes[0, 1].set_xlim(1549.20, 1550.02)
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("Sensing-FBG temperature (°C)")
    axes[0, 1].set_title("Spectral evolution with temperature")
    axes[0, 1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0, 1].legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    colorbar = fig.colorbar(mesh, ax=axes[0, 1], pad=0.02)
    colorbar.set_label("Normalized power")

    order = np.argsort(temperature)
    slope_nm_per_c, intercept_nm = np.polyfit(temperature, peak_sensing, 1)
    fit = slope_nm_per_c * temperature + intercept_nm
    axes[1, 0].scatter(
        temperature,
        peak_sensing,
        s=34,
        color=PALETTE["blue_main"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Sensing FBG",
    )
    axes[1, 0].plot(temperature[order], fit[order], color=PALETTE["black"], linewidth=1.8)
    axes[1, 0].scatter(
        temperature,
        peak_reference,
        s=29,
        color=PALETTE["red"],
        marker="s",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="20 °C reference FBG",
    )
    axes[1, 0].set_xlabel("Sensing-FBG temperature (°C)")
    axes[1, 0].set_ylabel("Reported peak wavelength (nm)")
    axes[1, 0].set_title("Independent temperature calibration")
    axes[1, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1, 0].text(
        0.04,
        0.95,
        f"Sensitivity = {metrics['sensing_peak_sensitivity_pm_per_c']:.2f} pm/°C\n"
        f"$R^2$ = {metrics['sensing_peak_fit_r2']:.3f}",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    axes[1, 0].legend(loc="lower right")
    light_grid(axes[1, 0])

    separation_pm = np.abs(peak_sensing - peak_reference) * 1000.0
    axes[1, 1].axhspan(0.0, 96.0, color=PALETTE["neutral"], alpha=0.45, linewidth=0)
    axes[1, 1].plot(
        temperature,
        separation_pm,
        color=PALETTE["blue_main"],
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )
    axes[1, 1].axhline(96.0, color=PALETTE["neutral_dark"], linewidth=1.1, linestyle="--")
    axes[1, 1].text(
        0.97,
        0.18,
        "Peak separation below\nOSA effective resolution",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=PALETTE["neutral_dark"],
    )
    axes[1, 1].set_xlabel("Sensing-FBG temperature (°C)")
    axes[1, 1].set_ylabel("Absolute peak separation (pm)")
    axes[1, 1].set_title("Most difficult overlap region")
    axes[1, 1].set_ylim(bottom=0.0)
    light_grid(axes[1, 1])

    for index, ax in enumerate(axes.flat):
        panel_label(ax, f"({chr(97 + index)})")
    fig.suptitle("External dual-peak FBG dataset: measured spectral behaviour", fontsize=14, fontweight="bold", y=1.02)
    return save_figure(fig, "fig06_external_dual_peak_dataset")


def read_own_channel_sensitivities() -> tuple[list[str], np.ndarray, np.ndarray]:
    source = ROOT / "outputs" / "real_data" / "real_peak_measurements.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels: list[str] = []
    sensitivities: list[float] = []
    fit_scores: list[float] = []
    for channel in ("CH1", "CH2", "CH3", "CH4"):
        selected = [row for row in rows if row["channel"] == channel]
        temperature = np.asarray([float(row["temperature_c"]) for row in selected])
        peak = np.asarray([float(row["peak_wavelength_nm"]) for row in selected])
        slope, intercept = np.polyfit(temperature, peak, 1)
        fitted = slope * temperature + intercept
        labels.append(channel)
        sensitivities.append(float(slope * 1000.0))
        fit_scores.append(float(r2_score(peak, fitted)))
    return labels, np.asarray(sensitivities), np.asarray(fit_scores)


def figure_external_benchmark(data: dict[str, object], metrics: dict[str, float]) -> list[Path]:
    temperature = np.asarray(data["temperature_c"], dtype=float)
    peak_sensing = np.asarray(data["peak_sensing_nm"], dtype=float)
    predictions = leave_one_out_temperature_prediction(peak_sensing, temperature)
    own_labels, own_sensitivity, own_r2 = read_own_channel_sensitivities()

    labels = own_labels + ["External\nsensing FBG"]
    sensitivities = np.concatenate(
        [own_sensitivity, [metrics["sensing_peak_sensitivity_pm_per_c"]]]
    )
    fit_scores = np.concatenate([own_r2, [metrics["sensing_peak_fit_r2"]]])
    colors = [
        PALETTE["blue_secondary"],
        PALETTE["teal"],
        PALETTE["green"],
        PALETTE["violet"],
        PALETTE["red"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    x = np.arange(len(labels))
    bars = axes[0].bar(
        x,
        sensitivities,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=1.0,
        width=0.72,
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Temperature sensitivity (pm/°C)")
    axes[0].set_title("Independent sensor sensitivity")
    axes[0].set_ylim(0.0, max(sensitivities) * 1.23)
    light_grid(axes[0])
    for bar, sensitivity, score in zip(bars, sensitivities, fit_scores):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            sensitivity + 0.18,
            f"{sensitivity:.2f}\n$R^2$={score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )

    lower = float(min(temperature.min(), predictions.min()) - 2.0)
    upper = float(max(temperature.max(), predictions.max()) + 2.0)
    axes[1].plot([lower, upper], [lower, upper], color=PALETTE["neutral_dark"], linestyle="--", linewidth=1.4)
    scatter = axes[1].scatter(
        temperature,
        predictions,
        c=temperature,
        cmap="cividis",
        s=48,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Measured temperature (°C)")
    axes[1].set_ylabel("LOOCV predicted temperature (°C)")
    axes[1].set_title("External-data calibration baseline")
    axes[1].text(
        0.05,
        0.94,
        f"MAE = {metrics['loocv_temperature_mae_c']:.2f} °C\n"
        f"RMSE = {metrics['loocv_temperature_rmse_c']:.2f} °C\n"
        f"$R^2$ = {metrics['loocv_temperature_r2']:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    light_grid(axes[1], axis="both")
    colorbar = fig.colorbar(scatter, ax=axes[1], pad=0.02)
    colorbar.set_label("Measured temperature (°C)")

    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.suptitle("Third-party data provide an independent temperature benchmark", fontsize=14, fontweight="bold", y=1.03)
    return save_figure(fig, "fig07_external_temperature_benchmark")


def write_output_readme(metrics: dict[str, float]) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "README.md"
    path.write_text(
        "# External dual-peak FBG dataset analysis\n\n"
        "Source: Serif Ali Sadik, *Optical Spectrum Dataset for Dual-Peak Temperature Measurements*, "
        "Mendeley Data v2, DOI: 10.17632/fymrd8js55.2. License: CC BY 4.0.\n\n"
        "This dataset is treated as an independent external benchmark, not as a large deep-learning "
        "training set. It contains 17 OSA spectra (1001 wavelength points each), with a sensing FBG at "
        "2–50 °C and a reference FBG held at 20 °C.\n\n"
        f"- Sensing-FBG sensitivity: {metrics['sensing_peak_sensitivity_pm_per_c']:.3f} pm/°C\n"
        f"- Linear fit R²: {metrics['sensing_peak_fit_r2']:.4f}\n"
        f"- Leave-one-out temperature MAE: {metrics['loocv_temperature_mae_c']:.3f} °C\n"
        f"- Leave-one-out temperature RMSE: {metrics['loocv_temperature_rmse_c']:.3f} °C\n"
        f"- Reference-peak standard deviation: {metrics['reference_peak_std_pm']:.3f} pm\n\n"
        "Processed files are under `data/external_mendeley_dual_peak/processed`; publication figures "
        "are `fig06_external_dual_peak_dataset` and `fig07_external_temperature_benchmark`.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    apply_publication_style()
    data = load_dataset()
    metrics = save_processed_data(data)
    paths = []
    paths.extend(figure_external_characterization(data, metrics))
    paths.extend(figure_external_benchmark(data, metrics))
    paths.append(write_output_readme(metrics))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
