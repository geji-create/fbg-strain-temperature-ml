"""Generate the expanded FBG dataset when needed and train PyTorch baselines.

Two models are trained with leakage-safe, condition-grouped splits:
1. A compact MLP using the six peak/side-lobe features.
2. A compact 1-D CNN using the complete reflectivity spectrum.

All target scaling is fitted on the training split only.  Metrics and
predictions are reported in physical units (degree Celsius and microstrain).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic_expanded_v3_same_device"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pytorch_expanded_v3_same_device"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dataset(
    data_dir: Path, conditions: int, seed: int, physics_profile: str
) -> None:
    required = [
        data_dir / "expanded_conditions.csv",
        data_dir / "expanded_six_features.csv",
        data_dir / "expanded_spectra.npz",
        data_dir / "expanded_dataset_metadata.json",
    ]
    if all(path.exists() for path in required):
        return
    generator_path = Path(__file__).with_name("generate_expanded_synthetic.py")
    spec = importlib.util.spec_from_file_location("expanded_generator", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_dataset(data_dir, conditions, seed, physics_profile)


def load_feature_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    path = data_dir / "expanded_six_features.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    feature_names = [
        "main_peak_wavelength_nm",
        "main_peak_power_db",
        "left_sidelobe_wavelength_nm",
        "left_sidelobe_power_db",
        "right_sidelobe_wavelength_nm",
        "right_sidelobe_power_db",
    ]
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in records], dtype=np.float32
    )
    targets = np.asarray(
        [[float(row["temperature_C"]), float(row["strain_ue"])] for row in records],
        dtype=np.float32,
    )
    splits = np.asarray([row["split"] for row in records])
    return features, targets, splits, records


def load_spectrum_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    payload = np.load(data_dir / "expanded_spectra.npz")
    spectra = payload["reflectivity"].astype(np.float32)
    condition_count, solver_count, wavelength_count = spectra.shape
    features = spectra.reshape(condition_count * solver_count, 1, wavelength_count)
    condition_targets = np.column_stack(
        [payload["temperature_C"], payload["strain_ue"]]
    ).astype(np.float32)
    targets = np.repeat(condition_targets, solver_count, axis=0)
    splits = np.repeat(payload["split"].astype(str), solver_count)
    solver_names = payload["solver_source"].astype(str)
    metadata: list[dict[str, str]] = []
    for condition_index in range(condition_count):
        for solver_index in range(solver_count):
            metadata.append(
                {
                    "sample_id": f"E{condition_index + 1:05d}_{solver_names[solver_index]}",
                    "condition_id": str(condition_index + 1),
                    "solver_source": solver_names[solver_index],
                    "split": str(payload["split"][condition_index]),
                }
            )
    return features, targets, splits, metadata


class FeatureMLP(nn.Module):
    def __init__(self, input_features: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SpectrumCNN(nn.Module):
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
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            # 461 wavelength samples become 115 after two 2x pooling stages.
            # Keeping the spatial axis preserves the absolute Bragg position.
            nn.Linear(48 * 115, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(values))


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axes: tuple[int, ...] | int = 0) -> "Standardizer":
        mean = values.mean(axis=axes, keepdims=True)
        scale = values.std(axis=axes, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale.reshape(1, -1) + self.mean.reshape(1, -1)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    residual = predicted - actual
    mae = np.mean(np.abs(residual), axis=0)
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    denominator = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(residual**2, axis=0) / np.maximum(denominator, 1e-12)
    return {
        "temperature": {
            "mae_C": float(mae[0]),
            "rmse_C": float(rmse[0]),
            "r2": float(r2[0]),
        },
        "strain": {
            "mae_ue": float(mae[1]),
            "rmse_ue": float(rmse[1]),
            "r2": float(r2[1]),
        },
    }


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[mask]), torch.from_numpy(targets[mask])
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def predict(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    for (batch,) in loader:
        outputs.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_one_model(
    name: str,
    model_factory: Callable[[], nn.Module],
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    splits: np.ndarray,
    metadata: list[dict[str, str]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> dict[str, object]:
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if not (train_mask.any() and validation_mask.any() and test_mask.any()):
        raise ValueError("Dataset must contain train, validation and test splits")

    if raw_features.ndim == 2:
        feature_scaler = Standardizer.fit(raw_features[train_mask], axes=0)
    else:
        feature_scaler = Standardizer.fit(raw_features[train_mask], axes=(0, 2))
    target_scaler = Standardizer.fit(raw_targets[train_mask], axes=0)
    features = feature_scaler.transform(raw_features)
    targets = target_scaler.transform(raw_targets)

    train_loader = make_loader(features, targets, train_mask, batch_size, True)
    validation_loader = make_loader(features, targets, validation_mask, batch_size, False)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, patience // 5), min_lr=1e-5
    )

    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / f"{name}_best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        training_total = 0.0
        training_count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            training_total += float(loss.item()) * batch_features.shape[0]
            training_count += batch_features.shape[0]

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.inference_mode():
            for batch_features, batch_targets in validation_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                loss = loss_function(model(batch_features), batch_targets)
                validation_total += float(loss.item()) * batch_features.shape[0]
                validation_count += batch_features.shape[0]
        training_loss = training_total / max(training_count, 1)
        validation_loss = validation_total / max(validation_count, 1)
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": name,
                    "feature_mean": feature_scaler.mean,
                    "feature_scale": feature_scaler.scale,
                    "target_mean": target_scaler.mean,
                    "target_scale": target_scaler.scale,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    normalized_predictions = predict(model, features, device, batch_size)
    physical_predictions = target_scaler.inverse(normalized_predictions)

    split_metrics = {
        split_name: metrics(raw_targets[splits == split_name], physical_predictions[splits == split_name])
        for split_name in ["train", "validation", "test"]
    }
    with (output_dir / f"{name}_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    prediction_fields = [
        "sample_id",
        "condition_id",
        "solver_source",
        "split",
        "actual_temperature_C",
        "predicted_temperature_C",
        "temperature_error_C",
        "actual_strain_ue",
        "predicted_strain_ue",
        "strain_error_ue",
    ]
    with (output_dir / f"{name}_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for index, row in enumerate(metadata):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "condition_id": row["condition_id"],
                    "solver_source": row["solver_source"],
                    "split": splits[index],
                    "actual_temperature_C": float(raw_targets[index, 0]),
                    "predicted_temperature_C": float(physical_predictions[index, 0]),
                    "temperature_error_C": float(physical_predictions[index, 0] - raw_targets[index, 0]),
                    "actual_strain_ue": float(raw_targets[index, 1]),
                    "predicted_strain_ue": float(physical_predictions[index, 1]),
                    "strain_error_ue": float(physical_predictions[index, 1] - raw_targets[index, 1]),
                }
            )

    return {
        "model": name,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss_scaled": best_validation,
        "metrics": split_metrics,
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--cnn-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument(
        "--physics-profile",
        choices=[
            "uniform",
            "uniform_same_device",
            "strain_gradient",
            "strain_gradient_same_device",
        ],
        default="strain_gradient_same_device",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dataset(args.data_dir, args.conditions, args.seed, args.physics_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_x, feature_y, feature_splits, feature_metadata = load_feature_data(args.data_dir)
    spectrum_x, spectrum_y, spectrum_splits, spectrum_metadata = load_spectrum_data(args.data_dir)
    if not np.isfinite(feature_x).all() or not np.isfinite(spectrum_x).all():
        raise ValueError("Non-finite input value found")

    results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "conditions": args.conditions,
        "physics_profile": args.physics_profile,
        "feature_samples": int(feature_x.shape[0]),
        "spectrum_samples": int(spectrum_x.shape[0]),
        "split_counts": {
            name: int(np.sum(feature_splits == name))
            for name in ["train", "validation", "test"]
        },
        "models": [],
    }
    results["models"].append(
        train_one_model(
            "feature_mlp",
            FeatureMLP,
            feature_x,
            feature_y,
            feature_splits,
            feature_metadata,
            args.output_dir,
            device,
            args.batch_size,
            args.mlp_epochs,
            args.patience,
        )
    )
    results["models"].append(
        train_one_model(
            "spectrum_cnn",
            SpectrumCNN,
            spectrum_x,
            spectrum_y,
            spectrum_splits,
            spectrum_metadata,
            args.output_dir,
            device,
            args.batch_size,
            args.cnn_epochs,
            args.patience,
        )
    )
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
