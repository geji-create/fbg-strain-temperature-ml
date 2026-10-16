"""Create publication-ready figures for the FBG reproduction project.

Outputs are written as PNG, PDF and SVG without overwriting exploratory plots.
The visual system follows the scientific-figure-making skill: sans-serif type,
minimal spines, semantic blue/green/red/gray colors, direct value labels and
300 dpi raster export.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "publication"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#6BB36B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#767676",
    "teal": "#42949E",
    "violet": "#8064A2",
    "black": "#272727",
    "grid": "#D9D9D9",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 11
    axes_linewidth: float = 1.5


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "axes.titleweight": "bold",
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": style.font_size - 1,
            "ytick.labelsize": style.font_size - 1,
            "legend.fontsize": style.font_size - 1,
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
        va="top",
        ha="left",
        color=PALETTE["black"],
    )


def light_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=PALETTE["grid"], linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)


def finalize_figure(fig: plt.Figure, stem: str, *, dpi: int = 300) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.25)
    paths: list[Path] = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
        paths.append(path)
    plt.close(fig)
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_real_temperature_calibration() -> list[Path]:
    rows = read_csv(ROOT / "outputs" / "real_data" / "real_peak_measurements.csv")
    channels = ["CH1", "CH2", "CH3", "CH4"]
    colors = [
        PALETTE["blue_main"],
        PALETTE["teal"],
        PALETTE["green_3"],
        PALETTE["violet"],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    for index, (ax, channel, color) in enumerate(zip(axes.flat, channels, colors)):
        selected = [row for row in rows if row["channel"] == channel]
        temperature = np.asarray([float(row["temperature_c"]) for row in selected])
        wavelength = np.asarray([float(row["peak_wavelength_nm"]) for row in selected])
        reference = wavelength[np.argmin(temperature)]
        delta_pm = (wavelength - reference) * 1000.0
        slope, intercept = np.polyfit(temperature, delta_pm, 1)
        fitted = slope * temperature + intercept
        denominator = np.sum((delta_pm - delta_pm.mean()) ** 2)
        r2 = 1.0 - np.sum((delta_pm - fitted) ** 2) / denominator
        order = np.argsort(temperature)

        ax.scatter(
            temperature,
            delta_pm,
            s=31,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax.plot(temperature[order], fitted[order], color=PALETTE["black"], linewidth=1.8)
        ax.set_title(channel, pad=7)
        ax.text(
            0.04,
            0.93,
            f"Sensitivity = {slope:.2f} pm/°C\n$R^2$ = {r2:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            color=PALETTE["black"],
        )
        light_grid(ax)
        panel_label(ax, f"({chr(97 + index)})")
    axes[1, 0].set_xlabel("Temperature (°C)")
    axes[1, 1].set_xlabel("Temperature (°C)")
    axes[0, 0].set_ylabel("Peak shift (pm)")
    axes[1, 0].set_ylabel("Peak shift (pm)")
    fig.suptitle("Real FBG temperature calibration across four channels", fontsize=14, fontweight="bold", y=1.01)
    return finalize_figure(fig, "fig01_real_temperature_calibration")


def figure_split_generalization() -> list[Path]:
    rows = read_csv(ROOT / "outputs" / "model_comparison" / "model_comparison.csv")
    splits = [
        ("paper_random_70_20_10", "Random split"),
        ("unseen_condition_70_20_10", "Unseen-condition split"),
    ]
    targets = [("strain_ue", "Strain"), ("temperature_c", "Temperature")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.15), sharey=True)
    x = np.arange(len(targets))
    width = 0.34
    for panel_index, (ax, (split_key, title)) in enumerate(zip(axes, splits)):
        classical = []
        cnn = []
        classical_names = []
        for target_key, _ in targets:
            matches = [row for row in rows if row["split_method"] == split_key and row["target"] == target_key]
            classical_row = next(row for row in matches if row["approach"].startswith("Best classical"))
            cnn_row = next(row for row in matches if row["approach"] == "1D CNN")
            classical.append(float(classical_row["r2"]))
            cnn.append(float(cnn_row["r2"]))
            classical_names.append(classical_row["approach"].split(": ")[-1].replace("_", " ").title())

        bars_a = ax.bar(
            x - width / 2,
            classical,
            width,
            label="Best classical",
            color=PALETTE["neutral"],
            edgecolor=PALETTE["black"],
            linewidth=1.0,
            hatch="//",
        )
        bars_b = ax.bar(
            x + width / 2,
            cnn,
            width,
            label="1D CNN",
            color=PALETTE["blue_main"],
            edgecolor=PALETTE["black"],
            linewidth=1.0,
        )
        ax.axhline(0.0, color=PALETTE["black"], linewidth=1.0)
        ax.set_xticks(x, [label for _, label in targets])
        ax.set_ylim(-0.58, 1.08)
        ax.set_title(title, pad=9)
        light_grid(ax)
        panel_label(ax, f"({chr(97 + panel_index)})")
        for bars in (bars_a, bars_b):
            for bar in bars:
                value = bar.get_height()
                offset = 0.035
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                )
        winner_text = " / ".join(classical_names)
        ax.text(
            0.5,
            -0.17,
            f"Classical winners: {winner_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=PALETTE["neutral_dark"],
        )
    axes[0].set_ylabel("Test $R^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Random splitting overestimates generalization", fontsize=14, fontweight="bold", y=1.12)
    return finalize_figure(fig, "fig02_split_generalization")


def figure_pytorch_progression() -> list[Path]:
    rows = read_csv(ROOT / "outputs" / "pytorch_improvement_summary.csv")
    label_map = {
        "V1 uniform FBG": "V1  Uniform FBG",
        "V2 strain gradient": "V2  Strain gradient",
        "V3 same device + positional CNN": "V3  Positional CNN",
        "Cross: uniform_cmt -> simplus_like": "CMT → SimPlus-like",
        "Cross: simplus_like -> uniform_cmt": "SimPlus-like → CMT",
    }
    labels = [label_map[row["experiment"]] for row in rows]
    temperature = np.asarray([float(row["temperature_mae_C"]) for row in rows])
    strain = np.asarray([float(row["strain_mae_ue"]) for row in rows])
    colors = [
        PALETTE["neutral_dark"],
        PALETTE["green_3"],
        PALETTE["blue_main"],
        PALETTE["teal"],
        PALETTE["violet"],
    ]
    hatches = ["//", "..", "", "\\", "xx"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    y = np.arange(len(labels))
    for panel_index, (ax, values, xlabel, limit) in enumerate(
        [
            (axes[0], temperature, "Temperature MAE (°C)", 8.6),
            (axes[1], strain, "Strain MAE (µε)", 72.0),
        ]
    ):
        bars = ax.barh(
            y,
            values,
            color=colors,
            edgecolor=PALETTE["black"],
            linewidth=1.0,
            height=0.62,
        )
        for bar, hatch, value in zip(bars, hatches, values):
            bar.set_hatch(hatch)
            ax.text(value + limit * 0.018, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9.5)
        ax.set_xlim(0, limit)
        ax.set_xlabel(xlabel)
        ax.set_title("Temperature" if panel_index == 0 else "Strain", pad=8)
        light_grid(ax, axis="x")
        panel_label(ax, f"({chr(97 + panel_index)})")
        ax.text(0.98, 0.04, "lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=PALETTE["neutral_dark"])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("PyTorch model progression and cross-solver validation", fontsize=14, fontweight="bold", y=1.02)
    return finalize_figure(fig, "fig03_pytorch_progression")


def figure_ablation() -> list[Path]:
    rows = read_csv(ROOT / "outputs" / "ablation_study" / "ablation_summary.csv")
    label_map = {
        "gradient_six_feature_mlp": "Six features\nMLP",
        "gradient_gap_cnn": "Full spectrum\nGAP CNN",
        "gradient_centered_cnn": "Peak-centred\nspectrum",
        "gradient_positional_cnn": "Full spectrum\n+ position",
        "uniform_positional_cnn": "Uniform strain\n+ position",
    }
    labels = [label_map[row["variant"]] for row in rows]
    colors = [
        PALETTE["neutral_dark"],
        PALETTE["green_3"],
        PALETTE["red_2"],
        PALETTE["blue_main"],
        PALETTE["red_strong"],
    ]
    hatches = ["//", "..", "xx", "", "\\"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    settings = [
        ("temperature_mae_C", "temperature_r2", "Temperature MAE (°C)", 9.7),
        ("strain_mae_ue", "strain_r2", "Strain MAE (µε)", 78.0),
    ]
    x = np.arange(len(labels))
    for panel_index, (ax, (mae_key, r2_key, ylabel, ymax)) in enumerate(zip(axes, settings)):
        means = np.asarray([float(row[f"{mae_key}_mean"]) for row in rows])
        stds = np.asarray([float(row[f"{mae_key}_std"]) for row in rows])
        r2s = np.asarray([float(row[f"{r2_key}_mean"]) for row in rows])
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3.5,
            color=colors,
            edgecolor=PALETTE["black"],
            linewidth=1.0,
            width=0.68,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        for bar, hatch, mean, r2 in zip(bars, hatches, means, r2s):
            bar.set_hatch(hatch)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + ymax * 0.035,
                f"{mean:.2f}\n$R^2$={r2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                linespacing=1.05,
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ymax)
        ax.set_title("Temperature discrimination" if panel_index == 0 else "Strain discrimination", pad=9)
        light_grid(ax)
        panel_label(ax, f"({chr(97 + panel_index)})")
        ax.text(0.98, 0.96, "lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=PALETTE["neutral_dark"])
    fig.suptitle("Ablation study reveals complementary position and shape cues", fontsize=14, fontweight="bold", y=1.03)
    return finalize_figure(fig, "fig04_ablation_position_shape")


def select_condition(temperature: np.ndarray, strain: np.ndarray, target_t: float, target_e: float) -> int:
    t_scale = max(float(np.ptp(temperature)), 1.0)
    e_scale = max(float(np.ptp(strain)), 1.0)
    distance = ((temperature - target_t) / t_scale) ** 2 + ((strain - target_e) / e_scale) ** 2
    return int(np.argmin(distance))


def peak_aligned(wavelength: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = (signal - signal.min()) / max(float(signal.max() - signal.min()), 1e-12)
    # A single argmax is unstable for flat-topped spectra.  Use the midpoint
    # of the high-reflectivity plateau as a reproducible spectral centre.
    plateau = wavelength[normalized >= 0.90]
    centre = float((plateau[0] + plateau[-1]) / 2.0) if plateau.size else float(wavelength[np.argmax(normalized)])
    relative = wavelength - centre
    return relative, normalized


def figure_information_mechanism() -> list[Path]:
    payload = np.load(ROOT / "data" / "synthetic_expanded_v3_same_device" / "expanded_spectra.npz")
    wavelength = payload["wavelength_nm"].astype(float)
    spectra = payload["reflectivity"][:, 0, :].astype(float)
    temperature = payload["temperature_C"].astype(float)
    strain = payload["strain_ue"].astype(float)

    temp_ids = [
        select_condition(temperature, strain, 10.0, 450.0),
        select_condition(temperature, strain, 38.0, 450.0),
    ]
    strain_ids = [
        select_condition(temperature, strain, 24.0, 100.0),
        select_condition(temperature, strain, 24.0, 800.0),
    ]
    temp_colors = [PALETTE["teal"], PALETTE["red_strong"]]
    strain_colors = [PALETTE["green_3"], PALETTE["blue_main"]]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.7))
    for idx, color in zip(temp_ids, temp_colors):
        raw = (spectra[idx] - spectra[idx].min()) / max(float(np.ptp(spectra[idx])), 1e-12)
        axes[0, 0].plot(wavelength, raw, color=color, linewidth=2.0, label=f"{temperature[idx]:.1f} °C, {strain[idx]:.0f} µε")
        rel, aligned = peak_aligned(wavelength, spectra[idx])
        axes[0, 1].plot(rel, aligned, color=color, linewidth=2.0, label=f"{temperature[idx]:.1f} °C")
    for idx, color in zip(strain_ids, strain_colors):
        raw = (spectra[idx] - spectra[idx].min()) / max(float(np.ptp(spectra[idx])), 1e-12)
        axes[1, 0].plot(wavelength, raw, color=color, linewidth=2.0, label=f"{temperature[idx]:.1f} °C, {strain[idx]:.0f} µε")
        rel, aligned = peak_aligned(wavelength, spectra[idx])
        axes[1, 1].plot(rel, aligned, color=color, linewidth=2.0, label=f"{strain[idx]:.0f} µε")

    titles = [
        "Temperature sweep — absolute wavelength",
        "Temperature sweep — centre aligned",
        "Strain sweep — absolute wavelength",
        "Strain sweep — centre aligned",
    ]
    for index, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, pad=8)
        ax.set_ylabel("Normalized reflectivity")
        ax.set_ylim(-0.04, 1.08)
        light_grid(ax)
        ax.legend(loc="best", fontsize=8.5)
        panel_label(ax, f"({chr(97 + index)})")
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_xlabel("Wavelength relative to centre (nm)")
    axes[1, 1].set_xlabel("Wavelength relative to centre (nm)")
    axes[0, 1].set_xlim(-0.34, 0.34)
    axes[1, 1].set_xlim(-0.34, 0.34)
    callout = dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PALETTE["grid"], alpha=0.92)
    axes[0, 1].text(0.97, 0.08, "Temperature curves nearly overlap\nafter centre alignment", transform=axes[0, 1].transAxes, fontsize=8.5, color=PALETTE["neutral_dark"], ha="right", va="bottom", bbox=callout)
    axes[1, 1].text(0.97, 0.08, "Strain-dependent shape\ndifference remains", transform=axes[1, 1].transAxes, fontsize=8.5, color=PALETTE["neutral_dark"], ha="right", va="bottom", bbox=callout)
    fig.suptitle("Why position identifies temperature and shape identifies strain", fontsize=14, fontweight="bold", y=1.02)
    return finalize_figure(fig, "fig05_position_shape_mechanism")


def write_readme() -> Path:
    path = OUT / "README.md"
    path.write_text(
        "# Publication figures\n\n"
        "All figures are exported as PNG, PDF and SVG. PNG files are 300 dpi; PDF/SVG retain vector text and geometry.\n\n"
        "- fig01_real_temperature_calibration: real four-channel temperature calibration.\n"
        "- fig02_split_generalization: random split versus unseen-condition evaluation.\n"
        "- fig03_pytorch_progression: V1–V3 improvements and cross-solver validation.\n"
        "- fig04_ablation_position_shape: three-seed ablation results with uncertainty.\n"
        "- fig05_position_shape_mechanism: new data-driven mechanism figure using V3 spectra.\n\n"
        "Additional figures generated by process_external_dual_peak_dataset.py:\n"
        "- fig06_external_dual_peak_dataset: third-party measured spectral behaviour.\n"
        "- fig07_external_temperature_benchmark: sensitivity comparison and LOOCV calibration.\n\n"
        "Color semantics: blue = key method, green = improvement, red = contrast, gray = baseline.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    apply_publication_style()
    outputs: list[Path] = []
    outputs.extend(figure_real_temperature_calibration())
    outputs.extend(figure_split_generalization())
    outputs.extend(figure_pytorch_progression())
    outputs.extend(figure_ablation())
    outputs.extend(figure_information_mechanism())
    outputs.append(write_readme())
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
