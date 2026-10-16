"""Cross-solver validation for the improved FBG spectrum CNN."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic_expanded_v3_same_device"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pytorch_expanded_v3_cross_solver"


def load_training_module():
    path = Path(__file__).with_name("train_pytorch_expanded.py")
    spec = importlib.util.spec_from_file_location("expanded_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    training = load_training_module()
    training.set_seed(20261013)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, targets, original_splits, metadata = training.load_spectrum_data(DATA_DIR)
    solver_source = np.asarray([row["solver_source"] for row in metadata])
    results: dict[str, object] = {
        "device": str(device),
        "protocol": (
            "Train and validate on one solver; test on unseen conditions from "
            "the other solver. Conditions remain disjoint."
        ),
        "directions": [],
    }

    directions = [
        ("uniform_cmt", "simplus_like"),
        ("simplus_like", "uniform_cmt"),
    ]
    for train_solver, test_solver in directions:
        selected = (
            ((solver_source == train_solver) & (original_splits != "test"))
            | ((solver_source == test_solver) & (original_splits == "test"))
        )
        selected_features = features[selected]
        selected_targets = targets[selected]
        selected_metadata = [row for row, keep in zip(metadata, selected) if keep]
        selected_original_splits = original_splits[selected]
        selected_solver = solver_source[selected]
        cross_splits = np.full(selected_original_splits.shape, "unused", dtype="U10")
        cross_splits[
            (selected_solver == train_solver) & (selected_original_splits == "train")
        ] = "train"
        cross_splits[
            (selected_solver == train_solver)
            & (selected_original_splits == "validation")
        ] = "validation"
        cross_splits[
            (selected_solver == test_solver) & (selected_original_splits == "test")
        ] = "test"
        usable = cross_splits != "unused"
        direction_name = f"cnn_{train_solver}_to_{test_solver}"
        result = training.train_one_model(
            direction_name,
            training.SpectrumCNN,
            selected_features[usable],
            selected_targets[usable],
            cross_splits[usable],
            [row for row, keep in zip(selected_metadata, usable) if keep],
            OUTPUT_DIR,
            device,
            batch_size=256,
            max_epochs=160,
            patience=30,
        )
        result["train_solver"] = train_solver
        result["test_solver"] = test_solver
        results["directions"].append(result)

    (OUTPUT_DIR / "cross_solver_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
