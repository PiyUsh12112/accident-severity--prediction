from __future__ import annotations

import pandas as pd

from src.modeling import load_model_bundle, predict_severity_probabilities
from src.paths import CLEAN_DATA_PATH, MODEL_BUNDLE_PATH, ZONE_RISK_PATH, ensure_directories
from src.preprocessing import create_model_features
from src.zones import attach_predicted_zone_risk, build_zone_risk_scores


def main() -> None:
    ensure_directories()
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    clean_df["accident_date"] = pd.to_datetime(clean_df["accident_date"], errors="coerce")
    clean_df["time"] = pd.to_datetime(
        clean_df["time"].astype("string").str.strip().str[-8:],
        format="%H:%M:%S",
        errors="coerce",
    )
    zone_scores = build_zone_risk_scores(clean_df)

    feature_df = create_model_features(clean_df)
    feature_df["lat_grid"] = (feature_df["latitude"] / 0.01).round().astype("Int64")
    feature_df["lon_grid"] = (feature_df["longitude"] / 0.01).round().astype("Int64")
    bundle = load_model_bundle(str(MODEL_BUNDLE_PATH))
    probability_frame = predict_severity_probabilities(bundle, feature_df)
    enriched_features = pd.concat([feature_df.reset_index(drop=True), probability_frame.reset_index(drop=True)], axis=1)
    zone_scores = attach_predicted_zone_risk(zone_scores, enriched_features)

    zone_scores.to_csv(ZONE_RISK_PATH, index=False)
    print(f"Saved zone risk scores to {ZONE_RISK_PATH}")


if __name__ == "__main__":
    main()
