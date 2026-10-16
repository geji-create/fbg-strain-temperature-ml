"""Run controlled ablations for the FBG temperature/strain model.

The study isolates three questions on leakage-safe, condition-grouped splits:

1. Are six hand-crafted features sufficient, or is the full spectrum useful?
2. Does preserving absolute wavelength position matter?
3. Does a non-uniform strain field add identifiable peak-shape information?

The uniform and non-uniform datasets use the same conditions, random seed,
device nuisance ranges, solvers, and split assignment.  Their only intended
physical difference is the longitudinal strain profile.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRADIENT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic_expanded_v3_same_device"
UNIFORM_DATA_DIR = PROJECT_ROOT / "data" / "synthetic_ablation_uniform_same_device"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation_study"


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING = load_script_module(
    "expanded_training", Path(__file__).with_name("train_pytorch_expanded.py")
)
GENERATOR = load_script_module(
    "expanded_generator_ablation", Path(__file__).with_name("generate_expanded_synthetic.py")
)


class GlobalAverageSpectrumCNN(nn.Module):
    """Translation-tolerant CNN that deliberately discards absolute position."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(values))


def peak_center_spectra(spectra: np.ndarray, half_width: int = 100) -> np.ndarray:
    """Extract and resize a peak-centred window, removing absolute wavelength."""
    if spectra.ndim != 3 or spectra.shape[1] != 1:
        raise ValueError("Expected spectra with shape (samples, 1, wavelength)")
    sample_count, _, wavelength_count = spectra.shape
    source_axis = np.linspace(-1.0, 1.0, 2 * half_width + 1, dtype=np.float32)
    target_axis = np.linspace(-1.0, 1.0, wavelength_count, dtype=np.float32)
    centered = np.empty_like(spectra)
    for sample_index in range(sample_count):
        values = spectra[sample_index, 0]
        peak_index = int(np.argmax(values))
        indices = np.arange(peak_index - half_width, peak_index + half_width + 1)
        clipped = np.clip(indices, 0, wavelength_count - 1)
        window = values[clipped].copy()
        invalid = (indices < 0) | (indices >= wavelength_count)
        if np.any(invalid):
            window[invalid] = float(np.min(values))
        centered[sample_index, 0] = np.interp(target_axis, source_axis, window)
    return centered


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    question: str
    model_factory: Callable[[], nn.Module]
    data_kind: str
    physics: str


def require_paired_datasets(
    conditions: int, seed: int
) -> tuple[tuple, tuple, tuple]:
    required_gradient = GRADIENT_DATA_DIR / "expanded_spectra.npz"
    if not required_gradient.exists():
        GENERATOR.generate_dataset(
            GRADIENT_DATA_DIR, conditions, seed, "strain_gradient_same_device"
        )

    uniform_metadata = UNIFORM_DATA_DIR / "expanded_dataset_metadata.json"
    regenerate_uniform = True
    if uniform_metadata.exists():
        metadata = json.loads(uniform_metadata.read_text(encoding="utf-8"))
        regenerate_uniform = not (
            metadata.get("independent_conditions") == conditions
            and metadata.get("seed") == seed
            and metadata.get("physics_profile") == "uniform_same_device"
        )
    if regenerate_uniform:
        GENERATOR.generate_dataset(
            UNIFORM_DATA_DIR, conditions, seed, "uniform_same_device"
        )

    gradient_features = TRAINING.load_feature_data(GRADIENT_DATA_DIR)
    gradient_spectra = TRAINING.load_spectrum_data(GRADIENT_DATA_DIR)
    uniform_spectra = TRAINING.load_spectrum_data(UNIFORM_DATA_DIR)

    gradient_x, gradient_y, gradient_split, gradient_meta = gradient_spectra
    uniform_x, uniform_y, uniform_split, uniform_meta = uniform_spectra
    if gradient_x.shape != uniform_x.shape:
        raise ValueError("Paired physics datasets have different spectrum shapes")
    if not np.allclose(gradient_y, uniform_y, atol=1e-6):
        raise ValueError("Paired physics datasets do not share identical targets")
    if not np.array_equal(gradient_split, uniform_split):
        raise ValueError("Paired physics datasets do not share identical splits")
    if len(gradient_meta) != len(uniform_meta):
        raise ValueError("Paired physics datasets have different metadata lengths")
    return gradient_features, gradient_spectra, uniform_spectra


def flatten_metrics(
    variant: Variant, seed: int, result: dict[str, object]
) -> dict[str, object]:
    test = result["metrics"]["test"]
    return {
        "variant": variant.key,
        "label": variant.label,
        "question": variant.question,
        "physics": variant.physics,
        "seed": seed,
        "parameters": result["parameters"],
        "best_epoch": result["best_epoch"],
        "epochs_run": result["epochs_run"],
        "temperature_mae_C": test["temperature"]["mae_C"],
        "temperature_rmse_C": test["temperature"]["rmse_C"],
        "temperature_r2": test["temperature"]["r2"],
        "strain_mae_ue": test["strain"]["mae_ue"],
        "strain_rmse_ue": test["strain"]["rmse_ue"],
        "strain_r2": test["strain"]["r2"],
    }


def aggregate_rows(rows: list[dict[str, object]], variants: list[Variant]) -> list[dict[str, object]]:
    metric_names = [
        "temperature_mae_C",
        "temperature_rmse_C",
        "temperature_r2",
        "strain_mae_ue",
        "strain_rmse_ue",
        "strain_r2",
    ]
    aggregated: list[dict[str, object]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant.key]
        item: dict[str, object] = {
            "variant": variant.key,
            "label": variant.label,
            "question": variant.question,
            "physics": variant.physics,
            "runs": len(selected),
            "parameters_mean": float(np.mean([float(row["parameters"]) for row in selected])),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in selected])
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregated.append(item)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    panels = [
        ("temperature_mae_C", "Temperature MAE (°C)", False),
        ("strain_mae_ue", "Strain MAE (µε)", False),
        ("temperature_r2", "Temperature R²", True),
        ("strain_r2", "Strain R²", True),
    ]
    colors = ["#6B7280", "#F59E0B", "#8B5CF6", "#0F766E", "#2563EB"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (metric, title, is_r2) in zip(axes.flat, panels):
        means = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        errors = np.asarray([float(row[f"{metric}_std"]) for row in rows])
        bars = axis.bar(x, means, yerr=errors, capsize=4, color=colors, alpha=0.92)
        axis.set_title(title, fontsize=13)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if is_r2:
            axis.set_ylim(min(-0.1, float((means - errors).min()) - 0.05), 1.05)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if is_r2 else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "FBG temperature–strain ablation study (condition-grouped test set)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(rows: list[dict[str, object]], output_path: Path, seeds: list[int]) -> None:
    by_key = {str(row["variant"]): row for row in rows}
    positional = by_key["gradient_positional_cnn"]
    uniform = by_key["uniform_positional_cnn"]
    features = by_key["gradient_six_feature_mlp"]
    gap = by_key["gradient_gap_cnn"]
    centered = by_key["gradient_centered_cnn"]

    def value(row: dict[str, object], metric: str) -> float:
        return float(row[f"{metric}_mean"])

    lines = [
        "# FBG temperature–strain ablation study",
        "",
        f"Model seeds: {', '.join(map(str, seeds))}.",
        "All results use the same 70/15/15 condition-grouped split. Paired solver spectra from one condition never cross splits.",
        "",
        "## Test-set results",
        "",
        "| Variant | Temperature MAE (°C) | Strain MAE (µε) | Temperature R² | Strain R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {tmae:.3f} ± {tmae_std:.3f} | {smae:.3f} ± {smae_std:.3f} | {tr2:.4f} ± {tr2_std:.4f} | {sr2:.4f} ± {sr2_std:.4f} |".format(
                label=row["label"],
                tmae=value(row, "temperature_mae_C"),
                tmae_std=float(row["temperature_mae_C_std"]),
                smae=value(row, "strain_mae_ue"),
                smae_std=float(row["strain_mae_ue_std"]),
                tr2=value(row, "temperature_r2"),
                tr2_std=float(row["temperature_r2_std"]),
                sr2=value(row, "strain_r2"),
                sr2_std=float(row["strain_r2_std"]),
            )
        )
    temp_physics_gain = 100.0 * (
        value(uniform, "temperature_mae_C") - value(positional, "temperature_mae_C")
    ) / value(uniform, "temperature_mae_C")
    strain_physics_gain = 100.0 * (
        value(uniform, "strain_mae_ue") - value(positional, "strain_mae_ue")
    ) / value(uniform, "strain_mae_ue")
    lines.extend(
        [
            "",
            "## Controlled conclusions",
            "",
            f"- Full spectrum + position reduces temperature MAE by {100.0 * (value(features, 'temperature_mae_C') - value(positional, 'temperature_mae_C')) / value(features, 'temperature_mae_C'):.1f}% and strain MAE by {100.0 * (value(features, 'strain_mae_ue') - value(positional, 'strain_mae_ue')) / value(features, 'strain_mae_ue'):.1f}% relative to the six-feature MLP.",
            f"- Replacing the position-aware head with global average pooling changes temperature MAE from {value(positional, 'temperature_mae_C'):.3f} to {value(gap, 'temperature_mae_C'):.3f} °C, quantifying the value of absolute wavelength position.",
            f"- Explicit peak centering changes temperature MAE from {value(positional, 'temperature_mae_C'):.3f} to {value(centered, 'temperature_mae_C'):.3f} °C; this is a second, independent check of position information.",
            f"- With device nuisance and operating conditions held fixed, non-uniform strain reduces temperature MAE by {temp_physics_gain:.1f}% and strain MAE by {strain_physics_gain:.1f}% relative to uniform strain.",
            "- These are synthetic-data ablations. They establish mechanism and internal validity, but do not replace experimental strain validation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=int, default=5000)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20261013, 20261014, 20261015]
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--cnn-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.conditions < 100:
        parser.error("--conditions must be at least 100")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gradient_features, gradient_spectra, uniform_spectra = require_paired_datasets(
        args.conditions, args.seeds[0]
    )
    feature_x, feature_y, feature_splits, feature_meta = gradient_features
    gradient_x, gradient_y, gradient_splits, gradient_meta = gradient_spectra
    uniform_x, uniform_y, uniform_splits, uniform_meta = uniform_spectra
    centered_x = peak_center_spectra(gradient_x)

    variants = [
        Variant(
            "gradient_six_feature_mlp",
            "Six features (MLP)",
            "input representation",
            TRAINING.FeatureMLP,
            "features",
            "non-uniform",
        ),
        Variant(
            "gradient_gap_cnn",
            "Full spectrum (GAP CNN)",
            "absolute position",
            GlobalAverageSpectrumCNN,
            "gradient",
            "non-uniform",
        ),
        Variant(
            "gradient_centered_cnn",
            "Peak-centred spectrum",
            "absolute position",
            TRAINING.SpectrumCNN,
            "centered",
            "non-uniform",
        ),
        Variant(
            "gradient_positional_cnn",
            "Full spectrum + position",
            "complete model",
            TRAINING.SpectrumCNN,
            "gradient",
            "non-uniform",
        ),
        Variant(
            "uniform_positional_cnn",
            "Uniform strain + position",
            "strain-profile physics",
            TRAINING.SpectrumCNN,
            "uniform",
            "uniform",
        ),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for variant in variants:
            TRAINING.set_seed(seed)
            if variant.data_kind == "features":
                raw_x, raw_y, splits, metadata = (
                    feature_x,
                    feature_y,
                    feature_splits,
                    feature_meta,
                )
                epochs = args.mlp_epochs
            elif variant.data_kind == "gradient":
                raw_x, raw_y, splits, metadata = (
                    gradient_x,
                    gradient_y,
                    gradient_splits,
                    gradient_meta,
                )
                epochs = args.cnn_epochs
            elif variant.data_kind == "centered":
                raw_x, raw_y, splits, metadata = (
                    centered_x,
                    gradient_y,
                    gradient_splits,
                    gradient_meta,
                )
                epochs = args.cnn_epochs
            else:
                raw_x, raw_y, splits, metadata = (
                    uniform_x,
                    uniform_y,
                    uniform_splits,
                    uniform_meta,
                )
                epochs = args.cnn_epochs
            run_dir = args.output_dir / "runs" / variant.key / str(seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running {variant.key}, seed={seed}, device={device}", flush=True)
            result = TRAINING.train_one_model(
                variant.key,
                variant.model_factory,
                raw_x,
                raw_y,
                splits,
                metadata,
                run_dir,
                device,
                args.batch_size,
                epochs,
                args.patience,
            )
            run_rows.append(flatten_metrics(variant, seed, result))
            print(json.dumps(run_rows[-1], ensure_ascii=False), flush=True)

    aggregate = aggregate_rows(run_rows, variants)
    write_csv(args.output_dir / "ablation_runs.csv", run_rows)
    write_csv(args.output_dir / "ablation_summary.csv", aggregate)
    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "conditions": args.conditions,
        "samples_per_dataset": int(gradient_x.shape[0]),
        "seeds": args.seeds,
        "split_rule": "70/15/15 grouped by condition; paired solvers share split",
        "paired_physics_control": (
            "uniform_same_device and strain_gradient_same_device share conditions, "
            "nuisance ranges, solvers and split; strain profile is the intended difference"
        ),
        "results": aggregate,
    }
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_aggregate(aggregate, args.output_dir / "ablation_comparison.png")
    write_report(aggregate, args.output_dir / "README.md", args.seeds)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
