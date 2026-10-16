from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cnn import (
    CNNConfig,
    SpectrumCNN,
    _metrics,
    _plot_history,
    _plot_predictions,
    _seed_everything,
)
from .features import spectral_columns_and_axis
from .hybrid import CHANNELS
from .modeling import TARGET_COLUMNS, make_split_indices


def fuse_channel_spectra(
    spectra: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    spectral_columns, wavelengths = spectral_columns_and_axis(spectra)
    rows = []
    arrays = []
    for (condition_id, replicate), group in spectra.groupby(
        ["condition_id", "replicate"], sort=True
    ):
        indexed = group.set_index("channel")
        if set(indexed.index) != set(CHANNELS):
            raise ValueError("Every fused CNN sample must have CH1-CH4 exactly once.")
        arrays.append(
            np.stack(
                [indexed.loc[channel, spectral_columns].to_numpy(dtype=np.float32) for channel in CHANNELS]
            )
        )
        first = group.iloc[0]
        rows.append(
            {
                "spectrum_id": f"MC{len(rows) + 1:05d}",
                "condition_id": condition_id,
                "replicate": replicate,
                "strain_actual_ue": float(first["strain_actual_ue"]),
                "temperature_actual_C": float(first["temperature_actual_C"]),
            }
        )
    return pd.DataFrame(rows), np.asarray(arrays), wavelengths


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        TensorDataset(torch.from_numpy(x[indices]).float(), torch.from_numpy(y[indices]).float()),
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
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * len(inputs)
            count += len(inputs)
    return total / max(count, 1)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for inputs, _ in loader:
            result.append(model(inputs.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(result)


def train_multichannel_cnn(
    spectra: pd.DataFrame, output_dir: str | Path, config: CNNConfig
) -> dict[str, object]:
    _seed_everything(config.random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, reflectivity, wavelengths = fuse_channel_spectra(spectra)
    raw_x = np.clip(10.0 * np.log10(np.clip(reflectivity, 1e-8, None)), -60.0, 0.0)
    raw_y = metadata[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    splitters = make_split_indices(metadata, config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows = []
    prediction_frames = []
    history_frames = []
    for split_name, partitions in splitters.items():
        train_indices = partitions["train"]
        x_mean = raw_x[train_indices].mean(axis=0, keepdims=True)
        x_std = np.maximum(raw_x[train_indices].std(axis=0, keepdims=True), 1e-4)
        y_mean = raw_y[train_indices].mean(axis=0, keepdims=True)
        y_std = np.maximum(raw_y[train_indices].std(axis=0, keepdims=True), 1e-6)
        x = ((raw_x - x_mean) / x_std).astype(np.float32)
        y = ((raw_y - y_mean) / y_std).astype(np.float32)
        loaders = {
            name: _loader(x, y, indices, config.batch_size, name == "train", device)
            for name, indices in partitions.items()
        }
        model = SpectrumCNN(dropout=config.dropout, input_channels=4).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )
        best_loss = float("inf")
        best_state = None
        stale = 0
        history = []
        for epoch in range(1, config.epochs + 1):
            train_loss = _epoch(model, loaders["train"], criterion, device, optimizer)
            validation_loss = _epoch(model, loaders["validation"], criterion, device, None)
            scheduler.step(validation_loss)
            history.append(
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
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                break
        if best_state is None:
            raise RuntimeError("Multichannel CNN did not produce a checkpoint.")
        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cnn_config": asdict(config),
                "input_channels": CHANNELS,
                "wavelengths_nm": wavelengths,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "split_method": split_name,
            },
            output_dir / f"multichannel_cnn_{split_name}.pt",
        )
        history_frames.append(pd.DataFrame(history))
        for partition_name, indices in partitions.items():
            loader = _loader(x, y, indices, config.batch_size, False, device)
            predicted = _predict(model, loader, device) * y_std + y_mean
            metric_rows.extend(
                _metrics(split_name, partition_name, raw_y[indices], predicted)
            )
            if partition_name == "test":
                details = metadata.iloc[indices][
                    ["spectrum_id", "condition_id", *TARGET_COLUMNS]
                ].reset_index(drop=True)
                details.insert(0, "split_method", split_name)
                details["strain_predicted_ue"] = predicted[:, 0]
                details["temperature_predicted_C"] = predicted[:, 1]
                prediction_frames.append(details)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    metrics.to_csv(output_dir / "cnn_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "cnn_test_predictions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(output_dir / "cnn_training_history.csv", index=False, encoding="utf-8-sig")
    _plot_history(history, output_dir / "cnn_training_history.png")
    _plot_predictions(predictions, output_dir / "cnn_predictions.png")
    test_rows = metrics[
        (metrics["partition"] == "test") & (metrics["target"] != "macro_average")
    ]
    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sample_count": len(metadata),
        "input_shape": [4, len(wavelengths)],
        "metrics": test_rows[["split_method", "target", "r2", "mae", "rmse"]].to_dict(
            orient="records"
        ),
    }
    (output_dir / "cnn_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_multichannel_cnn_on_anchor(
    spectra: pd.DataFrame, checkpoint_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata, reflectivity, wavelengths = fuse_channel_spectra(spectra)
    raw_x = np.clip(10.0 * np.log10(np.clip(reflectivity, 1e-8, None)), -60.0, 0.0)
    actual = metadata[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_frames = []
    prediction_frames = []
    for path in sorted(Path(checkpoint_dir).glob("multichannel_cnn_*.pt")):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not np.allclose(wavelengths, checkpoint["wavelengths_nm"]):
            raise ValueError("Real anchor wavelength grid differs from CNN checkpoint.")
        model = SpectrumCNN(
            dropout=float(checkpoint["cnn_config"]["dropout"]), input_channels=4
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        x = ((raw_x - checkpoint["x_mean"]) / checkpoint["x_std"]).astype(np.float32)
        with torch.no_grad():
            standardized = model(torch.from_numpy(x).float().to(device)).cpu().numpy()
        predicted = standardized * checkpoint["y_std"] + checkpoint["y_mean"]
        details = metadata.copy()
        details.insert(0, "split_method", checkpoint["split_method"])
        details["strain_predicted_ue"] = predicted[:, 0]
        details["temperature_predicted_C"] = predicted[:, 1]
        prediction_frames.append(details)
        local = _metrics(checkpoint["split_method"], "real_anchor", actual, predicted)
        metric_frames.append(pd.DataFrame(local))
    return pd.concat(metric_frames, ignore_index=True), pd.concat(
        prediction_frames, ignore_index=True
    )
