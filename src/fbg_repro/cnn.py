from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import spectral_columns_and_axis
from .modeling import TARGET_COLUMNS, TARGET_NAMES, make_split_indices


@dataclass
class CNNConfig:
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    dropout: float = 0.15
    random_seed: int = 42


class SpectrumCNN(nn.Module):
    def __init__(
        self, output_count: int = 2, dropout: float = 0.15, input_channels: int = 1
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(inputs))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[indices, None, :]).float(),
        torch.from_numpy(y[indices]).float(),
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
            total_count += len(inputs)
    return total_loss / max(total_count, 1)


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for inputs, _ in loader:
            batches.append(model(inputs.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _metrics(
    split_method: str,
    partition: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, float | str]]:
    rows = []
    r2_values = []
    for index, target in enumerate(TARGET_NAMES):
        true_values = actual[:, index]
        predicted_values = predicted[:, index]
        r2 = r2_score(true_values, predicted_values)
        r2_values.append(r2)
        rows.append(
            {
                "split_method": split_method,
                "model": "spectrum_cnn",
                "partition": partition,
                "target": target,
                "r2": r2,
                "mae": mean_absolute_error(true_values, predicted_values),
                "rmse": mean_squared_error(true_values, predicted_values) ** 0.5,
            }
        )
    rows.append(
        {
            "split_method": split_method,
            "model": "spectrum_cnn",
            "partition": partition,
            "target": "macro_average",
            "r2": float(np.mean(r2_values)),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    )
    return rows


def _plot_history(history: pd.DataFrame, output_path: Path) -> None:
    split_names = list(history["split_method"].unique())
    fig, axes = plt.subplots(1, len(split_names), figsize=(12, 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, split_name in zip(axes, split_names):
        subset = history[history["split_method"] == split_name]
        axis.plot(subset["epoch"], subset["train_loss"], label="train")
        axis.plot(subset["epoch"], subset["validation_loss"], label="validation")
        axis.set_title(split_name.replace("_", " "))
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Standardized MSE")
    fig.suptitle("1D CNN training history")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    split_names = list(predictions["split_method"].unique())
    fig, axes = plt.subplots(len(split_names), 2, figsize=(10.5, 4.5 * len(split_names)))
    axes = np.atleast_2d(axes)
    pairs = [
        ("strain_actual_ue", "strain_predicted_ue", "Strain (µε)"),
        ("temperature_actual_C", "temperature_predicted_C", "Temperature (°C)"),
    ]
    for row_index, split_name in enumerate(split_names):
        subset = predictions[predictions["split_method"] == split_name]
        for column_index, (actual, predicted, label) in enumerate(pairs):
            axis = axes[row_index, column_index]
            axis.scatter(subset[actual], subset[predicted], s=13, alpha=0.55)
            low = min(subset[actual].min(), subset[predicted].min())
            high = max(subset[actual].max(), subset[predicted].max())
            axis.plot([low, high], [low, high], "k--", linewidth=1)
            axis.set_xlabel(f"Actual {label}")
            axis.set_ylabel(f"Predicted {label}")
            axis.set_title(split_name.replace("_", " "))
            axis.grid(alpha=0.25)
    fig.suptitle("End-to-end 1D CNN test predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_cnn_models(
    spectra: pd.DataFrame,
    output_dir: str | Path,
    config: CNNConfig,
) -> dict[str, object]:
    _seed_everything(config.random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_columns, wavelengths = spectral_columns_and_axis(spectra)
    reflectivity = spectra[spectral_columns].to_numpy(dtype=np.float32)
    raw_x = np.clip(10.0 * np.log10(np.clip(reflectivity, 1e-8, None)), -60.0, 0.0)
    raw_y = spectra[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    splitters = make_split_indices(spectra, config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: list[dict[str, float | str]] = []
    prediction_frames = []
    history_frames = []
    for split_name, partitions in splitters.items():
        train_indices = partitions["train"]
        x_mean = raw_x[train_indices].mean(axis=0, keepdims=True)
        x_std = raw_x[train_indices].std(axis=0, keepdims=True)
        x_std = np.maximum(x_std, 1e-4)
        x = (raw_x - x_mean) / x_std
        y_mean = raw_y[train_indices].mean(axis=0, keepdims=True)
        y_std = raw_y[train_indices].std(axis=0, keepdims=True)
        y_std = np.maximum(y_std, 1e-6)
        y = (raw_y - y_mean) / y_std
        loaders = {
            name: _loader(
                x,
                y,
                indices,
                config.batch_size,
                name == "train",
                device,
            )
            for name, indices in partitions.items()
        }
        model = SpectrumCNN(dropout=config.dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        rows = []
        for epoch in range(1, config.epochs + 1):
            train_loss = _epoch(model, loaders["train"], criterion, device, optimizer)
            validation_loss = _epoch(
                model, loaders["validation"], criterion, device, None
            )
            scheduler.step(validation_loss)
            rows.append(
                {
                    "split_method": split_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
        if best_state is None:
            raise RuntimeError("CNN training did not produce a valid checkpoint.")
        model.load_state_dict(best_state)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_class": "SpectrumCNN",
            "cnn_config": asdict(config),
            "wavelengths_nm": wavelengths,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "target_columns": TARGET_COLUMNS,
            "split_method": split_name,
        }
        torch.save(checkpoint, output_dir / f"spectrum_cnn_{split_name}.pt")
        history_frames.append(pd.DataFrame(rows))
        for partition_name, indices in partitions.items():
            # Keep predictions aligned with the source rows. The training
            # loader itself is shuffled, so evaluation uses a fresh loader.
            evaluation_loader = _loader(
                x,
                y,
                indices,
                batch_size=config.batch_size,
                shuffle=False,
                device=device,
            )
            predicted_standard = _predict(model, evaluation_loader, device)
            predicted = predicted_standard * y_std + y_mean
            actual = raw_y[indices]
            metric_rows.extend(
                _metrics(split_name, partition_name, actual, predicted)
            )
            if partition_name == "test":
                details = spectra.iloc[indices][
                    [
                        name
                        for name in ["spectrum_id", "condition_id"]
                        if name in spectra.columns
                    ]
                ].reset_index(drop=True)
                details.insert(0, "split_method", split_name)
                details["strain_actual_ue"] = actual[:, 0]
                details["strain_predicted_ue"] = predicted[:, 0]
                details["temperature_actual_C"] = actual[:, 1]
                details["temperature_predicted_C"] = predicted[:, 1]
                prediction_frames.append(details)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    metrics.to_csv(output_dir / "cnn_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        output_dir / "cnn_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    history.to_csv(output_dir / "cnn_training_history.csv", index=False, encoding="utf-8-sig")
    _plot_history(history, output_dir / "cnn_training_history.png")
    _plot_predictions(predictions, output_dir / "cnn_predictions.png")
    test_rows = metrics[
        (metrics["partition"] == "test") & (metrics["target"] != "macro_average")
    ]
    summary = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "spectrum_count": int(len(spectra)),
        "wavelength_points": int(len(spectral_columns)),
        "metrics": test_rows[
            ["split_method", "target", "r2", "mae", "rmse"]
        ].to_dict(orient="records"),
    }
    with (output_dir / "cnn_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary

