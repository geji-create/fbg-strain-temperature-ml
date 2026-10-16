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

from fbg_repro.cnn import CNNConfig
from fbg_repro.multichannel_cnn import (
    evaluate_multichannel_cnn_on_anchor,
    train_multichannel_cnn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a four-channel FBG spectrum CNN.")
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--real-anchor", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = CNNConfig(**raw_config.get("cnn", {}))
    if args.quick:
        config.epochs = 12
        config.patience = 4
    spectra = pd.read_csv(args.spectra)
    anchors = pd.read_csv(args.real_anchor)
    summary = train_multichannel_cnn(spectra, args.output_dir, config)
    anchor_metrics, anchor_predictions = evaluate_multichannel_cnn_on_anchor(
        anchors, args.output_dir
    )
    anchor_metrics.to_csv(
        args.output_dir / "real_anchor_metrics.csv", index=False, encoding="utf-8-sig"
    )
    anchor_predictions.to_csv(
        args.output_dir / "real_anchor_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary["real_anchor_metrics"] = anchor_metrics[
        anchor_metrics["target"] != "macro_average"
    ][["split_method", "target", "r2", "mae", "rmse"]].to_dict(orient="records")
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
