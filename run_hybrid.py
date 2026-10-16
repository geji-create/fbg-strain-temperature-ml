from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fbg_repro.cnn import CNNConfig, train_cnn_models
from fbg_repro.config import FeatureConfig, ModelConfig
from fbg_repro.features import extract_six_features
from fbg_repro.hybrid import (
    HybridConfig,
    analytical_multichannel_predictions,
    build_channel_templates,
    combine_channel_features,
    config_as_dict,
    evaluate_classical_on_real_anchor,
    evaluate_cnn_on_real_anchor,
    evaluate_physics_guided_on_anchor,
    generate_hybrid_spectra,
    load_real_files,
    plot_hybrid_spectra,
    prediction_metrics,
    real_anchor_spectra,
    sensitivity_geometry,
    template_summary,
    train_physics_guided_models,
)
from fbg_repro.modeling import train_and_evaluate
from fbg_repro.plotting import plot_model_scores, plot_predictions


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a semi-empirical FBG strain-temperature digital twin."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    config = HybridConfig.from_json(args.config)
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = ModelConfig(**raw_config.get("models", {}))
    cnn_config = CNNConfig(**raw_config.get("cnn", {}))
    if args.quick:
        config.replicates_per_channel = 1
        model_config.n_estimators = 30
        cnn_config.epochs = 12
        cnn_config.patience = 4

    output_dir = args.output_dir.resolve()
    classical_dir = output_dir / "classical"
    multichannel_dir = output_dir / "multichannel"
    cnn_dir = output_dir / "cnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    classical_dir.mkdir(parents=True, exist_ok=True)
    multichannel_dir.mkdir(parents=True, exist_ok=True)

    records = load_real_files(args.input_dir)
    templates = build_channel_templates(records, config)
    template_frame = pd.DataFrame(template_summary(templates, config))
    template_frame.to_csv(
        output_dir / "channel_template_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spectra = generate_hybrid_spectra(records, templates, config)
    anchors = real_anchor_spectra(records, config)
    spectra.to_csv(
        output_dir / "hybrid_spectra.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    anchors.to_csv(
        output_dir / "real_anchor_spectra.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    plot_hybrid_spectra(spectra, output_dir / "hybrid_representative_spectra.png")

    feature_config = FeatureConfig(
        savgol_window=9,
        savgol_order=2,
        sidelobe_near_nm=0.35,
        sidelobe_far_nm=2.0,
    )
    features = extract_six_features(spectra, feature_config)
    anchor_features = extract_six_features(anchors, feature_config)
    features.to_csv(
        output_dir / "hybrid_six_features.csv", index=False, encoding="utf-8-sig"
    )
    anchor_features.to_csv(
        output_dir / "real_anchor_six_features.csv", index=False, encoding="utf-8-sig"
    )
    multichannel_features, multichannel_columns = combine_channel_features(features)
    multichannel_anchor_features, anchor_multichannel_columns = combine_channel_features(
        anchor_features
    )
    if multichannel_columns != anchor_multichannel_columns:
        raise RuntimeError("Hybrid and real multichannel feature columns differ.")
    multichannel_features.to_csv(
        output_dir / "hybrid_multichannel_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multichannel_anchor_features.to_csv(
        output_dir / "real_anchor_multichannel_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics, predictions = train_and_evaluate(features, model_config, classical_dir)
    metrics.to_csv(
        classical_dir / "model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        classical_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    plot_model_scores(metrics, classical_dir / "model_scores.png")
    plot_predictions(predictions, classical_dir / "predictions.png")
    anchor_metrics, anchor_predictions = evaluate_classical_on_real_anchor(
        anchor_features, classical_dir / "models"
    )
    anchor_metrics.to_csv(
        classical_dir / "real_anchor_metrics.csv", index=False, encoding="utf-8-sig"
    )
    anchor_predictions.to_csv(
        classical_dir / "real_anchor_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    multi_metrics, multi_predictions = train_and_evaluate(
        multichannel_features,
        model_config,
        multichannel_dir,
        feature_columns=multichannel_columns,
    )
    multi_metrics.to_csv(
        multichannel_dir / "model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    multi_predictions.to_csv(
        multichannel_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    plot_model_scores(multi_metrics, multichannel_dir / "model_scores.png")
    plot_predictions(multi_predictions, multichannel_dir / "predictions.png")
    multi_anchor_metrics, multi_anchor_predictions = evaluate_classical_on_real_anchor(
        multichannel_anchor_features,
        multichannel_dir / "models",
        feature_columns=multichannel_columns,
    )
    multi_anchor_metrics.to_csv(
        multichannel_dir / "real_anchor_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multi_anchor_predictions.to_csv(
        multichannel_dir / "real_anchor_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analytical_predictions = analytical_multichannel_predictions(
        multichannel_features, templates, config
    )
    analytical_anchor_predictions = analytical_multichannel_predictions(
        multichannel_anchor_features, templates, config
    )
    analytical_metrics = prediction_metrics(
        analytical_predictions, "four_channel_physics_inverse"
    )
    analytical_anchor_metrics = prediction_metrics(
        analytical_anchor_predictions, "four_channel_physics_inverse"
    )
    analytical_predictions.to_csv(
        multichannel_dir / "analytical_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analytical_anchor_predictions.to_csv(
        multichannel_dir / "analytical_real_anchor_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analytical_metrics.to_csv(
        multichannel_dir / "analytical_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analytical_anchor_metrics.to_csv(
        multichannel_dir / "analytical_real_anchor_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    physics_metrics, physics_predictions = train_physics_guided_models(
        multichannel_features,
        templates,
        config,
        model_config,
        multichannel_dir,
    )
    physics_anchor_metrics, physics_anchor_predictions = (
        evaluate_physics_guided_on_anchor(
            multichannel_anchor_features,
            templates,
            config,
            multichannel_dir,
        )
    )
    physics_metrics.to_csv(
        multichannel_dir / "physics_guided_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    physics_predictions.to_csv(
        multichannel_dir / "physics_guided_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    physics_anchor_metrics.to_csv(
        multichannel_dir / "physics_guided_real_anchor_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    physics_anchor_predictions.to_csv(
        multichannel_dir / "physics_guided_real_anchor_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cnn_summary = None
    if not args.skip_cnn:
        cnn_summary = train_cnn_models(spectra, cnn_dir, cnn_config)
        cnn_anchor_metrics, cnn_anchor_predictions = evaluate_cnn_on_real_anchor(
            anchors, cnn_dir
        )
        cnn_anchor_metrics.to_csv(
            cnn_dir / "real_anchor_metrics.csv", index=False, encoding="utf-8-sig"
        )
        cnn_anchor_predictions.to_csv(
            cnn_dir / "real_anchor_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    test_metrics = metrics[
        (metrics["partition"] == "test") & (metrics["target"] != "macro_average")
    ]
    best_classical = (
        test_metrics.sort_values("r2", ascending=False)
        .groupby(["split_method", "target"], as_index=False)
        .first()[["split_method", "target", "model", "r2", "mae", "rmse"]]
    )
    anchor_all = anchor_metrics[anchor_metrics["channel"] == "all"]
    best_anchor = (
        anchor_all.sort_values("mae")
        .groupby(["split_method", "target"], as_index=False)
        .first()[["split_method", "target", "model", "r2", "mae", "rmse"]]
    )
    multi_test = multi_metrics[
        (multi_metrics["partition"] == "test")
        & (multi_metrics["target"] != "macro_average")
    ]
    best_multichannel = (
        multi_test.sort_values("r2", ascending=False)
        .groupby(["split_method", "target"], as_index=False)
        .first()[["split_method", "target", "model", "r2", "mae", "rmse"]]
    )
    multi_anchor_all = multi_anchor_metrics[multi_anchor_metrics["channel"] == "all"]
    best_multichannel_anchor = (
        multi_anchor_all.sort_values("mae")
        .groupby(["split_method", "target"], as_index=False)
        .first()[["split_method", "target", "model", "r2", "mae", "rmse"]]
    )
    summary = {
        "method": "semi-empirical digital twin",
        "scientific_status": (
            "Temperature behavior and noise come from real spectra; strain labels "
            "and strain-gradient broadening are physics-based simulations, not experiments."
        ),
        "hybrid_config": config_as_dict(config),
        "real_file_count": len(records),
        "spectrum_count": len(spectra),
        "condition_count": int(spectra["condition_id"].nunique()),
        "wavelength_point_count": int(
            sum(str(column).startswith("R_") for column in spectra.columns)
        ),
        "feature_warning_count": int(
            (features["feature_warning"].fillna("") != "").sum()
        ),
        "channel_calibration": template_frame.to_dict(orient="records"),
        "sensitivity_geometry": sensitivity_geometry(templates, config),
        "best_classical_test_metrics": best_classical.to_dict(orient="records"),
        "best_classical_real_anchor_metrics": best_anchor.to_dict(orient="records"),
        "best_multichannel_test_metrics": best_multichannel.to_dict(orient="records"),
        "best_multichannel_real_anchor_metrics": best_multichannel_anchor.to_dict(
            orient="records"
        ),
        "analytical_multichannel_metrics": analytical_metrics.to_dict(orient="records"),
        "analytical_multichannel_real_anchor_metrics": analytical_anchor_metrics.to_dict(
            orient="records"
        ),
        "physics_guided_metrics": physics_metrics[
            physics_metrics["partition"] == "test"
        ].to_dict(orient="records"),
        "physics_guided_real_anchor_metrics": physics_anchor_metrics.to_dict(
            orient="records"
        ),
        "cnn": cnn_summary,
    }
    (output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
