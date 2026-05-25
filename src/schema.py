from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


RAW_TO_CANONICAL = {
    "Accident_Index": "accident_index",
    "Accident Date": "accident_date",
    "Day_of_Week": "day_of_week",
    "Junction_Control": "junction_control",
    "Junction_Detail": "junction_detail",
    "Accident_Severity": "accident_severity",
    "Latitude": "latitude",
    "Light_Conditions": "light_conditions",
    "Local_Authority_(District)": "local_authority_district",
    "Carriageway_Hazards": "carriageway_hazards",
    "Longitude": "longitude",
    "Number_of_Casualties": "number_of_casualties",
    "Number_of_Vehicles": "number_of_vehicles",
    "Police_Force": "police_force",
    "Road_Surface_Conditions": "road_surface_conditions",
    "Road_Type": "road_type",
    "Speed_limit": "speed_limit",
    "Time": "time",
    "Urban_or_Rural_Area": "urban_or_rural_area",
    "Weather_Conditions": "weather_conditions",
    "Vehicle_Type": "vehicle_type",
}

CANONICAL_COLUMNS = list(RAW_TO_CANONICAL.values())
SEVERITY_SYNONYMS = {
    "fetal": "Fatal",
    "fatal": "Fatal",
    "serious": "Serious",
    "slight": "Slight",
}
SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_TO_LABEL = {name: index for index, name in enumerate(SEVERITY_ORDER)}
NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "number_of_casualties",
    "number_of_vehicles",
    "speed_limit",
]
CATEGORICAL_COLUMNS = [
    "day_of_week",
    "junction_control",
    "junction_detail",
    "light_conditions",
    "local_authority_district",
    "carriageway_hazards",
    "police_force",
    "road_surface_conditions",
    "road_type",
    "urban_or_rural_area",
    "weather_conditions",
    "vehicle_type",
]
REQUIRED_COLUMNS = {
    "accident_index",
    "accident_date",
    "accident_severity",
    "latitude",
    "longitude",
    "number_of_casualties",
    "number_of_vehicles",
    "speed_limit",
    "time",
}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=RAW_TO_CANONICAL).copy()
    renamed.columns = (
        renamed.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[()]+", "", regex=True)
    )
    return renamed


def validate_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_severity(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    normalized = str(value).strip()
    mapped = SEVERITY_SYNONYMS.get(normalized.lower())
    return mapped if mapped is not None else normalized.title()


def apply_canonical_types(df: pd.DataFrame) -> pd.DataFrame:
    canonical = df.copy()
    canonical["accident_date"] = pd.to_datetime(canonical["accident_date"], errors="coerce")
    canonical["time"] = pd.to_datetime(
        canonical["time"].astype("string").str.strip().str[-8:],
        format="%H:%M:%S",
        errors="coerce",
    )
    canonical["accident_severity"] = canonical["accident_severity"].map(normalize_severity)

    for column in NUMERIC_COLUMNS:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        if column in canonical.columns:
            canonical[column] = canonical[column].astype("string").str.strip()

    return canonical


def validate_content(df: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    unknown_severity = sorted(
        value for value in df["accident_severity"].dropna().unique() if value not in SEVERITY_TO_LABEL
    )
    if unknown_severity:
        issues.append(
            ValidationIssue(
                level="warning",
                message=f"Unexpected severity values preserved as-is: {unknown_severity}",
            )
        )

    bad_dates = int(df["accident_date"].isna().sum())
    if bad_dates:
        issues.append(ValidationIssue(level="warning", message=f"{bad_dates} rows have invalid accident_date values"))

    bad_times = int(df["time"].isna().sum())
    if bad_times:
        issues.append(ValidationIssue(level="warning", message=f"{bad_times} rows have invalid time values"))

    missing_geo = int(df[["latitude", "longitude"]].isna().any(axis=1).sum())
    if missing_geo:
        issues.append(ValidationIssue(level="warning", message=f"{missing_geo} rows are missing latitude/longitude"))

    return issues


def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("accident_index", "string", "Unique accident identifier"),
        ("accident_date", "datetime", "Date of the accident"),
        ("day_of_week", "string", "Weekday label"),
        ("junction_control", "string", "Junction control type"),
        ("junction_detail", "string", "Junction layout/detail"),
        ("accident_severity", "category", "Severity class: Fatal, Serious, Slight"),
        ("severity_label", "int", "Encoded severity label: Fatal=0, Serious=1, Slight=2"),
        ("latitude", "float", "Accident latitude"),
        ("longitude", "float", "Accident longitude"),
        ("time", "datetime", "Accident time parsed from HH:MM:SS"),
        ("year", "int", "Year extracted from accident_date"),
        ("month", "int", "Month extracted from accident_date"),
        ("day", "int", "Day extracted from accident_date"),
        ("hour", "int", "Hour extracted from time"),
        ("number_of_casualties", "float", "Number of casualties in the accident"),
        ("number_of_vehicles", "float", "Number of vehicles involved"),
        ("casualties_per_vehicle", "float", "number_of_casualties / number_of_vehicles"),
        ("speed_limit", "float", "Road speed limit"),
        ("road_type", "string", "Road type"),
        ("road_surface_conditions", "string", "Road surface state"),
        ("weather_conditions", "string", "Weather at the time of accident"),
        ("light_conditions", "string", "Lighting conditions"),
        ("urban_or_rural_area", "string", "Urban or rural classification"),
        ("vehicle_type", "string", "Primary vehicle type"),
        ("local_authority_district", "string", "District/local authority"),
    ]
    return pd.DataFrame(rows, columns=["column_name", "dtype", "description"])
