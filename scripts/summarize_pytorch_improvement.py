"""Create a compact CSV and figure comparing the PyTorch model revisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


def cnn_test_row(label: str, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = next(item for item in payload["models"] if item["model"] == "spectrum_cnn")
    test = model["metrics"]["test"]
    return {
        "experiment": label,
        "temperature_mae_C": test["temperature"]["mae_C"],
        "temperature_r2": test["temperature"]["r2"],
        "strain_mae_ue": test["strain"]["mae_ue"],
        "strain_r2": test["strain"]["r2"],
    }


def main() -> None:
    rows = [
        cnn_test_row(
            "V1 uniform FBG",
            OUTPUT / "pytorch_expanded_v1" / "metrics.json",
        ),
        cnn_test_row(
            "V2 strain gradient",
            OUTPUT / "pytorch_expanded_v2_nonuniform" / "metrics.json",
        ),
        cnn_test_row(
            "V3 same device + positional CNN",
            OUTPUT / "pytorch_expanded_v3_same_device" / "metrics.json",
        ),
    ]
    cross = json.loads(
        (
            OUTPUT
            / "pytorch_expanded_v3_cross_solver"
            / "cross_solver_metrics.json"
        ).read_text(encoding="utf-8")
    )
    for item in cross["directions"]:
        test = item["metrics"]["test"]
        rows.append(
            {
                "experiment": f"Cross: {item['train_solver']} -> {item['test_solver']}",
                "temperature_mae_C": test["temperature"]["mae_C"],
                "temperature_r2": test["temperature"]["r2"],
                "strain_mae_ue": test["strain"]["mae_ue"],
                "strain_r2": test["strain"]["r2"],
            }
        )

    csv_path = OUTPUT / "pytorch_improvement_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["experiment"]) for row in rows]
    temperature_mae = [float(row["temperature_mae_C"]) for row in rows]
    strain_mae = [float(row["strain_mae_ue"]) for row in rows]
    colors = ["#9CA3AF", "#60A5FA", "#10B981", "#F59E0B", "#F97316"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(labels, temperature_mae, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test MAE (°C)")
    axes[0].set_title("Temperature prediction")
    axes[1].barh(labels, strain_mae, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test MAE (με)")
    axes[1].set_title("Strain prediction")
    for axis, values in zip(axes, [temperature_mae, strain_mae]):
        axis.grid(axis="x", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(value, index, f" {value:.2f}", va="center", fontsize=9)
    figure.suptitle("FBG PyTorch model improvement and cross-solver validation")
    figure.tight_layout()
    figure.savefig(OUTPUT / "pytorch_improvement_summary.png", dpi=180, bbox_inches="tight")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
