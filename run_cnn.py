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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an end-to-end 1D CNN on FBG spectra.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = CNNConfig(
        epochs=12 if args.quick else args.epochs,
        batch_size=args.batch_size,
        patience=4 if args.quick else args.patience,
    )
    spectra = pd.read_csv(args.input_csv)
    summary = train_cnn_models(spectra, args.output_dir, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

