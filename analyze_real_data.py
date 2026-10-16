from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and calibrate the real FBG .list spectra."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def temperature_from_filename(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"File name is not a numeric temperature label: {path.name}") from exc


def refined_peak(wavelength: np.ndarray, signal_db: np.ndarray) -> tuple[float, float]:
    """Locate the maximum and refine it with a three-point parabola."""
    index = int(np.argmax(signal_db))
    if index == 0 or index == len(signal_db) - 1:
        return float(wavelength[index]), float(signal_db[index])
    left, center, right = signal_db[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(wavelength[index]), float(center)
    offset = 0.5 * (left - right) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    spacing = float(wavelength[index + 1] - wavelength[index])
    peak_nm = float(wavelength[index] + offset * spacing)
    peak_db = float(center - 0.25 * (left - right) * offset)
    return peak_nm, peak_db


def load_measurements(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_rows: list[dict[str, object]] = []
    spectrum_frames: list[pd.DataFrame] = []
    files = sorted(input_dir.glob("*.list"), key=temperature_from_filename)
    if not files:
        raise FileNotFoundError(f"No .list files found in {input_dir}")
    reference_wave: np.ndarray | None = None
    for path in files:
        temperature = temperature_from_filename(path)
        frame = pd.read_csv(path, sep="\t")
        required = {"Wave", *CHANNELS}
        if not required.issubset(frame.columns):
            missing = sorted(required.difference(frame.columns))
            raise ValueError(f"{path.name} is missing columns: {missing}")
        wavelength = frame["Wave"].to_numpy(dtype=float)
        if reference_wave is None:
            reference_wave = wavelength
        elif not np.allclose(wavelength, reference_wave):
            raise ValueError(f"Wavelength grid differs in {path.name}")
        tidy = frame.melt(
            id_vars="Wave", value_vars=CHANNELS, var_name="channel", value_name="power_db"
        )
        tidy.insert(0, "temperature_c", temperature)
        tidy.insert(0, "source_file", path.name)
        spectrum_frames.append(tidy)
        for channel in CHANNELS:
            signal = frame[channel].to_numpy(dtype=float)
            peak_nm, peak_db = refined_peak(wavelength, signal)
            peak_rows.append(
                {
                    "source_file": path.name,
                    "temperature_c": temperature,
                    "channel": channel,
                    "peak_wavelength_nm": peak_nm,
                    "peak_power_db": peak_db,
                }
            )
    return pd.concat(spectrum_frames, ignore_index=True), pd.DataFrame(peak_rows)


def calibrate(peaks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for channel in CHANNELS:
        subset = peaks[peaks["channel"] == channel]
        temperature = subset[["temperature_c"]].to_numpy()
        wavelength = subset["peak_wavelength_nm"].to_numpy()
        model = LinearRegression().fit(temperature, wavelength)
        predicted_wavelength = model.predict(temperature)
        predicted_temperature = (wavelength - model.intercept_) / model.coef_[0]
        loocv_temperature = np.empty(len(subset), dtype=float)
        for held_out in range(len(subset)):
            keep = np.arange(len(subset)) != held_out
            fold_model = LinearRegression().fit(temperature[keep], wavelength[keep])
            loocv_temperature[held_out] = (
                wavelength[held_out] - fold_model.intercept_
            ) / fold_model.coef_[0]
        rows.append(
            {
                "channel": channel,
                "measurement_count": len(subset),
                "sensitivity_nm_per_C": float(model.coef_[0]),
                "sensitivity_pm_per_C": float(model.coef_[0] * 1000.0),
                "intercept_nm": float(model.intercept_),
                "wavelength_r2": float(r2_score(wavelength, predicted_wavelength)),
                "wavelength_mae_pm": float(
                    mean_absolute_error(wavelength, predicted_wavelength) * 1000.0
                ),
                "fit_temperature_mae_C": float(
                    mean_absolute_error(temperature.ravel(), predicted_temperature)
                ),
                "fit_temperature_rmse_C": float(
                    np.sqrt(mean_squared_error(temperature.ravel(), predicted_temperature))
                ),
                "loocv_temperature_r2": float(
                    r2_score(temperature.ravel(), loocv_temperature)
                ),
                "loocv_temperature_mae_C": float(
                    mean_absolute_error(temperature.ravel(), loocv_temperature)
                ),
                "loocv_temperature_rmse_C": float(
                    np.sqrt(mean_squared_error(temperature.ravel(), loocv_temperature))
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_spectra(spectra: pd.DataFrame, output_path: Path) -> None:
    temperatures = sorted(spectra["temperature_c"].unique())
    selected = [temperatures[0], temperatures[len(temperatures) // 2], temperatures[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, channel in zip(axes.flat, CHANNELS):
        subset = spectra[(spectra["channel"] == channel) & spectra["temperature_c"].isin(selected)]
        for temperature, group in subset.groupby("temperature_c"):
            axis.plot(group["Wave"], group["power_db"], label=f"{temperature:g} °C")
        axis.set_title(channel)
        axis.set_ylabel("Power (dB)")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes[-1, :]:
        axis.set_xlabel("Wavelength (nm)")
    fig.suptitle("Representative real FBG spectra")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration(peaks: pd.DataFrame, calibration: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, channel in zip(axes.flat, CHANNELS):
        subset = peaks[peaks["channel"] == channel].sort_values("temperature_c")
        stats = calibration[calibration["channel"] == channel].iloc[0]
        fitted = (
            stats["intercept_nm"]
            + stats["sensitivity_nm_per_C"] * subset["temperature_c"]
        )
        axis.scatter(subset["temperature_c"], subset["peak_wavelength_nm"], s=24)
        axis.plot(subset["temperature_c"], fitted, color="tab:red")
        axis.set_title(
            f"{channel}: {stats['sensitivity_pm_per_C']:.2f} pm/°C, "
            f"R²={stats['wavelength_r2']:.4f}"
        )
        axis.set_ylabel("Peak wavelength (nm)")
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Temperature (°C)")
    fig.suptitle("Real FBG temperature calibration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectra, peaks = load_measurements(args.input_dir)
    calibration = calibrate(peaks)
    spectra.to_csv(
        args.output_dir / "real_spectra_long.csv", index=False, encoding="utf-8-sig"
    )
    peaks.to_csv(args.output_dir / "real_peak_measurements.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(
        args.output_dir / "temperature_calibration.csv", index=False, encoding="utf-8-sig"
    )
    plot_spectra(spectra, args.output_dir / "real_spectra.png")
    plot_calibration(peaks, calibration, args.output_dir / "temperature_calibration.png")
    summary = {
        "label_assumption": "Numeric file names are interpreted as temperature in degrees Celsius.",
        "file_count": int(peaks["source_file"].nunique()),
        "channel_count": len(CHANNELS),
        "wavelength_point_count": int(spectra["Wave"].nunique()),
        "wavelength_min_nm": float(spectra["Wave"].min()),
        "wavelength_max_nm": float(spectra["Wave"].max()),
        "temperature_min_C": float(peaks["temperature_c"].min()),
        "temperature_max_C": float(peaks["temperature_c"].max()),
        "calibration": calibration.to_dict(orient="records"),
    }
    (args.output_dir / "real_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
