from __future__ import annotations

import pandas as pd

from src.features import load_clean_data, save_feature_data
from src.ingest import load_raw_data, save_clean_data
from src.modeling import predict_severity_probabilities, save_training_outputs, train_model
from src.paths import (
    CLEAN_DATA_PATH,
    DATA_DICTIONARY_PATH,
    EVALUATION_REPORT_PATH,
    FEATURE_DATA_PATH,
    MODEL_BUNDLE_PATH,
    ZONE_RISK_PATH,
    ensure_directories,
)
from src.preprocessing import clean_raw_dataframe, create_model_features
from src.schema import CANONICAL_COLUMNS, build_data_dictionary
from src.zones import attach_predicted_zone_risk, build_zone_risk_scores


def main() -> None:
    ensure_directories()

    raw_df = load_raw_data()
    clean_df, issues = clean_raw_dataframe(raw_df)
    save_clean_data(clean_df[CANONICAL_COLUMNS], CLEAN_DATA_PATH)

    feature_df = create_model_features(clean_df)
    save_feature_data(feature_df, FEATURE_DATA_PATH)

    artifacts = train_model(feature_df)
    save_training_outputs(str(MODEL_BUNDLE_PATH), str(EVALUATION_REPORT_PATH), artifacts)

    zone_scores = build_zone_risk_scores(clean_df)
    scored_features = feature_df.copy()
    scored_features["lat_grid"] = (scored_features["latitude"] / 0.01).round().astype("Int64")
    scored_features["lon_grid"] = (scored_features["longitude"] / 0.01).round().astype("Int64")
    probability_frame = predict_severity_probabilities(artifacts.bundle, scored_features)
    scored_features = pd.concat([scored_features.reset_index(drop=True), probability_frame.reset_index(drop=True)], axis=1)
    zone_scores = attach_predicted_zone_risk(zone_scores, scored_features)
    zone_scores.to_csv(ZONE_RISK_PATH, index=False)

    data_dictionary = build_data_dictionary()
    data_dictionary.to_csv(DATA_DICTIONARY_PATH, index=False)

    print(f"Saved cleaned data to {CLEAN_DATA_PATH}")
    print(f"Saved feature data to {FEATURE_DATA_PATH}")
    print(f"Saved zone scores to {ZONE_RISK_PATH}")
    print(f"Saved model bundle to {MODEL_BUNDLE_PATH}")
    print(f"Saved metrics to {EVALUATION_REPORT_PATH}")
    print("Validation notes:")
    for issue in issues:
        print(f"- {issue}")


if __name__ == "__main__":
    main()
