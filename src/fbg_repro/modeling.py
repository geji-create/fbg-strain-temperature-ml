from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from .config import ModelConfig
from .features import FEATURE_COLUMNS


TARGET_COLUMNS = ["strain_actual_ue", "temperature_actual_C"]
TARGET_NAMES = ["strain_ue", "temperature_c"]


def _random_split(frame: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(frame))
    labels = frame["condition_id"] if "condition_id" in frame else None
    try:
        train, remainder = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=labels
        )
        remainder_labels = labels.iloc[remainder] if labels is not None else None
        validation, test = train_test_split(
            remainder,
            test_size=1.0 / 3.0,
            random_state=seed,
            stratify=remainder_labels,
        )
    except ValueError:
        train, remainder = train_test_split(
            indices, test_size=0.30, random_state=seed
        )
        validation, test = train_test_split(
            remainder, test_size=1.0 / 3.0, random_state=seed
        )
    return {"train": train, "validation": validation, "test": test}


def _group_split(frame: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    if "condition_id" not in frame:
        raise ValueError("Grouped evaluation requires a condition_id column.")
    indices = np.arange(len(frame))
    groups = frame["condition_id"].to_numpy()
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_position, remainder_position = next(first.split(indices, groups=groups))
    train = indices[train_position]
    remainder = indices[remainder_position]
    remainder_groups = groups[remainder]
    second = GroupShuffleSplit(n_splits=1, train_size=2.0 / 3.0, random_state=seed)
    validation_position, test_position = next(
        second.split(remainder, groups=remainder_groups)
    )
    return {
        "train": train,
        "validation": remainder[validation_position],
        "test": remainder[test_position],
    }


def make_split_indices(frame: pd.DataFrame, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Return the paper-style random split and a leakage-resistant grouped split."""
    return {
        "paper_random_70_20_10": _random_split(frame, seed),
        "unseen_condition_70_20_10": _group_split(frame, seed),
    }


def _models(config: ModelConfig) -> dict[str, object]:
    tree = DecisionTreeRegressor(random_state=config.random_seed)
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_seed,
            n_jobs=-1,
        ),
        "bagging": BaggingRegressor(
            estimator=tree,
            n_estimators=config.n_estimators,
            random_state=config.random_seed,
            n_jobs=-1,
        ),
        "adaboost": MultiOutputRegressor(
            AdaBoostRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.boosting_learning_rate,
                random_state=config.random_seed,
            ),
            n_jobs=-1,
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.boosting_learning_rate,
                random_state=config.random_seed,
            ),
            n_jobs=-1,
        ),
    }


def _metric_rows(
    split_method: str,
    model_name: str,
    partition: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    r2_values = []
    for target_index, target_name in enumerate(TARGET_NAMES):
        y_true = actual[:, target_index]
        y_pred = predicted[:, target_index]
        r2 = r2_score(y_true, y_pred)
        r2_values.append(r2)
        rows.append(
            {
                "split_method": split_method,
                "model": model_name,
                "partition": partition,
                "target": target_name,
                "r2": r2,
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            }
        )
    rows.append(
        {
            "split_method": split_method,
            "model": model_name,
            "partition": partition,
            "target": "macro_average",
            "r2": float(np.mean(r2_values)),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    )
    return rows


def train_and_evaluate(
    features: pd.DataFrame,
    config: ModelConfig,
    output_dir: str | Path,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    missing = set(feature_columns + TARGET_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if features[feature_columns + TARGET_COLUMNS].isna().any().any():
        raise ValueError("Features or targets contain missing values.")
    output_dir = Path(output_dir)
    model_root = output_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    x = features[feature_columns].to_numpy(dtype=float)
    y = features[TARGET_COLUMNS].to_numpy(dtype=float)
    splitters = make_split_indices(features, config.random_seed)
    metric_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_name, partitions in splitters.items():
        split_model_dir = model_root / split_name
        split_model_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in _models(config).items():
            model.fit(x[partitions["train"]], y[partitions["train"]])
            joblib.dump(model, split_model_dir / f"{model_name}.joblib")
            for partition_name, indices in partitions.items():
                prediction = np.asarray(model.predict(x[indices]))
                metric_rows.extend(
                    _metric_rows(
                        split_name,
                        model_name,
                        partition_name,
                        y[indices],
                        prediction,
                    )
                )
                if partition_name == "test":
                    details = features.iloc[indices][
                        [
                            name
                            for name in ["spectrum_id", "condition_id"]
                            if name in features.columns
                        ]
                    ].reset_index(drop=True)
                    details.insert(0, "split_method", split_name)
                    details.insert(1, "model", model_name)
                    details["strain_actual_ue"] = y[indices, 0]
                    details["strain_predicted_ue"] = prediction[:, 0]
                    details["temperature_actual_C"] = y[indices, 1]
                    details["temperature_predicted_C"] = prediction[:, 1]
                    prediction_frames.append(details)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)

