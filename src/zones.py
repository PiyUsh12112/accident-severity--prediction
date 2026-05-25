from __future__ import annotations

import pandas as pd


DEFAULT_GRID_SIZE = 0.01
DEFAULT_SEVERITY_WEIGHTS = {"Fatal": 5, "Serious": 3, "Slight": 1}


def build_zone_risk_scores(
    df: pd.DataFrame,
    grid_size: float = DEFAULT_GRID_SIZE,
    severity_weights: dict[str, int] | None = None,
    high_risk_quantile: float = 0.90,
) -> pd.DataFrame:
    weights = severity_weights or DEFAULT_SEVERITY_WEIGHTS
    zoned = df.copy()
    zoned["lat_grid"] = (zoned["latitude"] / grid_size).round().astype("Int64")
    zoned["lon_grid"] = (zoned["longitude"] / grid_size).round().astype("Int64")
    zoned["severity_weight"] = zoned["accident_severity"].map(weights).fillna(0)

    zone_risk = (
        zoned.groupby(["lat_grid", "lon_grid"], dropna=False)
        .agg(
            accident_count=("accident_index", "count"),
            risk_score=("severity_weight", "sum"),
            fatal_count=("accident_severity", lambda values: int((values == "Fatal").sum())),
            serious_count=("accident_severity", lambda values: int((values == "Serious").sum())),
            slight_count=("accident_severity", lambda values: int((values == "Slight").sum())),
            avg_lat=("latitude", "mean"),
            avg_lon=("longitude", "mean"),
        )
        .reset_index()
    )

    threshold = zone_risk["risk_score"].quantile(high_risk_quantile)
    zone_risk["danger_level"] = zone_risk["risk_score"].ge(threshold).map({True: "High", False: "Normal"})
    return zone_risk.sort_values(["risk_score", "accident_count"], ascending=[False, False])


def attach_predicted_zone_risk(zone_risk: pd.DataFrame, featured_with_predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_summary = (
        featured_with_predictions.groupby(["lat_grid", "lon_grid"], dropna=False)
        .agg(
            predicted_fatal_risk=("predicted_fatal_risk", "mean"),
            predicted_serious_risk=("predicted_serious_risk", "mean"),
            predicted_severe_risk=("predicted_severe_risk", "mean"),
        )
        .reset_index()
    )

    enriched = zone_risk.merge(prediction_summary, on=["lat_grid", "lon_grid"], how="left")
    for column in ["predicted_fatal_risk", "predicted_serious_risk", "predicted_severe_risk"]:
        enriched[column] = enriched[column].fillna(0.0)

    max_risk = enriched["risk_score"].max()
    historical_norm = enriched["risk_score"] / max_risk if max_risk else 0
    enriched["combined_risk_score"] = max_risk * (0.7 * historical_norm + 0.3 * enriched["predicted_severe_risk"])
    combined_threshold = enriched["combined_risk_score"].quantile(0.90)
    enriched["combined_danger_level"] = enriched["combined_risk_score"].ge(combined_threshold).map(
        {True: "High", False: "Normal"}
    )
    return enriched.sort_values(["combined_risk_score", "risk_score"], ascending=[False, False])
