from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .features import extract_six_features
from .modeling import train_and_evaluate
from .plotting import plot_model_scores, plot_predictions, plot_representative_spectra
from .simulation import generate_spectra


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the FBG strain-temperature ML workflow."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Use an existing spectra CSV instead of generating new spectra.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        help="Override replicates per condition from the configuration.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 10 replicates and 30 estimators for a fast end-to-end check.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    config = ProjectConfig.from_json(args.config)
    if args.replicates is not None:
        config.simulation.replicates_per_condition = args.replicates
    if args.quick:
        config.simulation.replicates_per_condition = 10
        config.models.n_estimators = 30
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = output_dir / "spectra.csv"
    features_path = output_dir / "six_features.csv"
    metrics_path = output_dir / "model_metrics.csv"
    predictions_path = output_dir / "test_predictions.csv"
    if args.input_csv:
        spectra = pd.read_csv(args.input_csv)
        source = str(args.input_csv.resolve())
    else:
        spectra = generate_spectra(config.simulation)
        spectra.to_csv(spectra_path, index=False, encoding="utf-8-sig")
        source = str(spectra_path)
    features = extract_six_features(spectra, config.features)
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    metrics, predictions = train_and_evaluate(features, config.models, output_dir)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    plot_representative_spectra(spectra, output_dir / "representative_spectra.png")
    plot_model_scores(metrics, output_dir / "model_scores.png")
    plot_predictions(predictions, output_dir / "predictions.png")
    test_metrics = metrics[
        (metrics["partition"] == "test") & (metrics["target"] != "macro_average")
    ]
    summary = {
        "spectra_source": source,
        "spectra_count": int(len(spectra)),
        "condition_count": int(features["condition_id"].nunique()),
        "feature_warning_count": int((features["feature_warning"].fillna("") != "").sum()),
        "best_test_rows": test_metrics.sort_values("r2", ascending=False)
        .groupby(["split_method", "target"], as_index=False)
        .first()[["split_method", "target", "model", "r2", "mae", "rmse"]]
        .to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = _arguments()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

