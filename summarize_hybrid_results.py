from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def select_best(
    frame: pd.DataFrame,
    method: str,
    scope: str,
    split: str | None = None,
    partition: str | None = None,
    choose: str = "r2",
) -> pd.DataFrame:
    data = frame.copy()
    if split is not None and "split_method" in data:
        data = data[data["split_method"] == split]
    if partition is not None and "partition" in data:
        data = data[data["partition"] == partition]
    if "channel" in data:
        data = data[data["channel"] == "all"]
    data = data[data["target"] != "macro_average"]
    ascending = choose == "mae"
    selected = (
        data.sort_values(choose, ascending=ascending)
        .groupby("target", as_index=False)
        .first()
    )
    selected["method"] = method
    selected["scope"] = scope
    return selected[["scope", "method", "target", "r2", "mae", "rmse"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--cnn-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = "unseen_condition_70_20_10"
    simulated = []
    anchor = []
    for subdir, label in [
        ("classical", "Single-channel six features"),
        ("multichannel", "Four-channel feature fusion"),
    ]:
        simulated.append(
            select_best(
                pd.read_csv(args.hybrid_dir / subdir / "model_metrics.csv"),
                label,
                "simulated_unseen_condition",
                split=split,
                partition="test",
            )
        )
        anchor.append(
            select_best(
                pd.read_csv(args.hybrid_dir / subdir / "real_anchor_metrics.csv"),
                label,
                "real_temperature_anchor",
                split=split,
                choose="mae",
            )
        )
    simulated.append(
        select_best(
            pd.read_csv(args.hybrid_dir / "multichannel" / "physics_guided_metrics.csv"),
            "Physics-guided two-stage",
            "simulated_unseen_condition",
            split=split,
            partition="test",
        )
    )
    anchor.append(
        select_best(
            pd.read_csv(
                args.hybrid_dir / "multichannel" / "physics_guided_real_anchor_metrics.csv"
            ),
            "Physics-guided two-stage",
            "real_temperature_anchor",
            split=split,
            choose="mae",
        )
    )
    simulated.append(
        select_best(
            pd.read_csv(args.cnn_dir / "cnn_metrics.csv"),
            "Four-channel CNN (quick)",
            "simulated_unseen_condition",
            split=split,
            partition="test",
        )
    )
    anchor.append(
        select_best(
            pd.read_csv(args.cnn_dir / "real_anchor_metrics.csv"),
            "Four-channel CNN (quick)",
            "real_temperature_anchor",
            split=split,
            choose="mae",
        )
    )
    combined = pd.concat([*simulated, *anchor], ignore_index=True)
    combined.to_csv(
        args.output_dir / "method_comparison.csv", index=False, encoding="utf-8-sig"
    )

    methods = [
        "Single-channel six features",
        "Four-channel feature fusion",
        "Physics-guided two-stage",
        "Four-channel CNN (quick)",
    ]
    short = ["Single 6F", "Four-channel", "Physics-guided", "4CH CNN"]
    simulated_frame = combined[combined["scope"] == "simulated_unseen_condition"]
    anchor_frame = combined[combined["scope"] == "real_temperature_anchor"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    positions = np.arange(len(methods))
    width = 0.36
    for offset, target, label in [
        (-width / 2, "strain_ue", "Strain"),
        (width / 2, "temperature_c", "Temperature"),
    ]:
        values = [
            simulated_frame[
                (simulated_frame["method"] == method)
                & (simulated_frame["target"] == target)
            ]["r2"].iloc[0]
            for method in methods
        ]
        bars = axes[0].bar(positions + offset, values, width, label=label)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[0].set_title("Semi-empirical unseen-condition R²")
    axes[0].set_xticks(positions, short, rotation=20)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()
    temp_mae = [
        anchor_frame[
            (anchor_frame["method"] == method)
            & (anchor_frame["target"] == "temperature_c")
        ]["mae"].iloc[0]
        for method in methods
    ]
    bars = axes[1].bar(positions, temp_mae, color="#f97316")
    axes[1].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[1].set_title("Real temperature anchor MAE (°C)")
    axes[1].set_xticks(positions, short, rotation=20)
    strain_mae = [
        anchor_frame[
            (anchor_frame["method"] == method)
            & (anchor_frame["target"] == "strain_ue")
        ]["mae"].iloc[0]
        for method in methods
    ]
    bars = axes[2].bar(positions, strain_mae, color="#3b82f6")
    axes[2].bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    axes[2].set_title("Real zero-strain false positive (µε)")
    axes[2].set_xticks(positions, short, rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("No-new-experiment FBG method comparison")
    fig.tight_layout()
    fig.savefig(args.output_dir / "method_comparison.png", dpi=180)
    plt.close(fig)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
