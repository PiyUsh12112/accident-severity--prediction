from __future__ import annotations

import pandas as pd

from src.modeling import save_training_outputs, train_model
from src.paths import EVALUATION_REPORT_PATH, FEATURE_DATA_PATH, MODEL_BUNDLE_PATH, ensure_directories


def load_feature_data(path=FEATURE_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    ensure_directories()
    feature_df = load_feature_data()
    artifacts = train_model(feature_df)
    save_training_outputs(str(MODEL_BUNDLE_PATH), str(EVALUATION_REPORT_PATH), artifacts)

    print(f"Saved model bundle to {MODEL_BUNDLE_PATH}")
    print(f"Saved evaluation report to {EVALUATION_REPORT_PATH}")
    print(f"Accuracy: {artifacts.metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
