from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the best classical model with the end-to-end 1D CNN."
    )
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classical = pd.read_csv(args.classical)
    classical = classical[
        (classical["partition"] == "test")
        & (classical["target"] != "macro_average")
    ].copy()
    classical["r2"] = pd.to_numeric(classical["r2"])
    best_indices = classical.groupby(["split_method", "target"])["r2"].idxmax()
    classical = classical.loc[best_indices].copy()
    classical["approach"] = "Best classical: " + classical["model"]

    cnn = pd.read_csv(args.cnn)
    cnn = cnn[(cnn["partition"] == "test") & (cnn["target"] != "macro_average")].copy()
    cnn["approach"] = "1D CNN"

    columns = ["split_method", "target", "approach", "r2", "mae", "rmse"]
    combined = pd.concat([classical[columns], cnn[columns]], ignore_index=True)
    combined = combined.sort_values(["split_method", "target", "approach"])
    combined.to_csv(
        args.output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    split_names = list(combined["split_method"].drop_duplicates())
    target_names = ["strain_ue", "temperature_c"]
    fig, axes = plt.subplots(1, len(split_names), figsize=(12, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, split_name in zip(axes, split_names):
        subset = combined[combined["split_method"] == split_name]
        positions = np.arange(len(target_names))
        width = 0.36
        classical_values: list[float] = []
        cnn_values: list[float] = []
        classical_labels: list[str] = []
        for target in target_names:
            target_rows = subset[subset["target"] == target]
            classical_row = target_rows[target_rows["approach"] != "1D CNN"].iloc[0]
            cnn_row = target_rows[target_rows["approach"] == "1D CNN"].iloc[0]
            classical_values.append(float(classical_row["r2"]))
            cnn_values.append(float(cnn_row["r2"]))
            classical_labels.append(str(classical_row["approach"]))
        bars_a = axis.bar(
            positions - width / 2,
            classical_values,
            width,
            label="Best classical model",
        )
        bars_b = axis.bar(positions + width / 2, cnn_values, width, label="1D CNN")
        axis.axhline(0, color="black", linewidth=0.8)
        short_names = {
            "random_forest": "RF",
            "adaboost": "AdaBoost",
            "gradient_boosting": "GBR",
        }
        tick_labels = []
        for target, label in zip(target_names, classical_labels):
            model_name = label.replace("Best classical: ", "")
            model_name = short_names.get(model_name, model_name)
            target_name = "Strain" if target == "strain_ue" else "Temperature"
            tick_labels.append(f"{target_name}\n(classical: {model_name})")
        axis.set_xticks(positions, tick_labels)
        axis.set_title(split_name.replace("_", " "))
        axis.set_ylabel("Test R²")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars_a, fmt="%.3f", padding=3, fontsize=9)
        axis.bar_label(bars_b, fmt="%.3f", padding=3, fontsize=9)
    axes[0].legend(loc="lower left")
    fig.suptitle("Classical ML vs end-to-end spectrum CNN")
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
