from __future__ import annotations

import pandas as pd

from src.schema import SEVERITY_TO_LABEL, apply_canonical_types, standardize_column_names, validate_columns, validate_content


def clean_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.drop_duplicates().dropna(how="all")
    cleaned = standardize_column_names(cleaned)
    validate_columns(cleaned)
    cleaned = apply_canonical_types(cleaned)

    issues = [f"{issue.level}: {issue.message}" for issue in validate_content(cleaned)]
    return cleaned, issues


def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    featured = df.copy()

    featured["year"] = featured["accident_date"].dt.year
    featured["month"] = featured["accident_date"].dt.month
    featured["day"] = featured["accident_date"].dt.day
    featured["hour"] = featured["time"].dt.hour

    featured["casualties_per_vehicle"] = (
        featured["number_of_casualties"] / featured["number_of_vehicles"]
    )
    featured["casualties_per_vehicle"] = featured["casualties_per_vehicle"].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    featured["casualties_per_vehicle"] = featured["casualties_per_vehicle"].fillna(0)

    featured["severity_label"] = featured["accident_severity"].map(SEVERITY_TO_LABEL)
    featured = featured.dropna(subset=["severity_label", "latitude", "longitude"])
    featured["severity_label"] = featured["severity_label"].astype(int)
    return featured
