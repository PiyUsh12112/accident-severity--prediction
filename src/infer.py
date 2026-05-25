from __future__ import annotations

import argparse
import pandas as pd

from src.modeling import load_model_bundle, predict_severity
from src.paths import MODEL_BUNDLE_PATH
from src.preprocessing import create_model_features
from src.schema import apply_canonical_types, standardize_column_names, validate_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Run severity inference on a CSV file.")
    parser.add_argument("input_csv", help="Path to the CSV file containing accident records")
    parser.add_argument("--output-csv", help="Optional path to write predictions")
    args = parser.parse_args()

    incoming = pd.read_csv(args.input_csv)
    incoming = standardize_column_names(incoming)
    validate_columns(incoming)
    incoming = apply_canonical_types(incoming)
    featured = create_model_features(incoming)

    bundle = load_model_bundle(str(MODEL_BUNDLE_PATH))
    predictions = predict_severity(bundle, featured)

    if args.output_csv:
        predictions.to_csv(args.output_csv, index=False)
    else:
        print(predictions[["accident_index", "predicted_severity", "prediction_confidence"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
