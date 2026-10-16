from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import spectral_columns_and_axis


def plot_representative_spectra(frame: pd.DataFrame, output_path: str | Path) -> None:
    columns, wavelengths = spectral_columns_and_axis(frame)
    condition_ids = sorted(frame["condition_id"].unique())
    selected = [condition_ids[0], condition_ids[len(condition_ids) // 2], condition_ids[-1]]
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for condition_id in selected:
        row = frame.loc[frame["condition_id"] == condition_id].iloc[0]
        label = (
            f"strain={row['strain_nominal_ue']:.0f} µε, "
            f"T={row['temperature_nominal_C']:.1f} °C"
        )
        axis.plot(wavelengths, row[columns].to_numpy(dtype=float), label=label)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Reflectivity")
    axis.set_title("Representative simulated uniform-FBG spectra")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_scores(metrics: pd.DataFrame, output_path: str | Path) -> None:
    selected = metrics[
        (metrics["partition"] == "test") & (metrics["target"] != "macro_average")
    ].copy()
    split_names = list(selected["split_method"].unique())
    fig, axes = plt.subplots(1, len(split_names), figsize=(13, 5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, split_name in zip(axes, split_names):
        subset = selected[selected["split_method"] == split_name]
        pivot = subset.pivot(index="model", columns="target", values="r2")
        pivot.plot(kind="bar", ax=axis, color=["#3b82f6", "#f97316"])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(split_name.replace("_", " "))
        axis.set_xlabel("")
        axis.set_ylabel("Test R²")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Paper-style random split vs unseen-condition evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_predictions(predictions: pd.DataFrame, output_path: str | Path) -> None:
    subset = predictions[
        (predictions["split_method"] == "paper_random_70_20_10")
        & (predictions["model"] == "gradient_boosting")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    pairs = [
        ("strain_actual_ue", "strain_predicted_ue", "Strain (µε)"),
        ("temperature_actual_C", "temperature_predicted_C", "Temperature (°C)"),
    ]
    for axis, (actual, predicted, label) in zip(axes, pairs):
        axis.scatter(subset[actual], subset[predicted], s=16, alpha=0.65)
        low = min(subset[actual].min(), subset[predicted].min())
        high = max(subset[actual].max(), subset[predicted].max())
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_xlabel(f"Actual {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.grid(alpha=0.25)
    fig.suptitle("Gradient boosting predictions: paper-style split")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

