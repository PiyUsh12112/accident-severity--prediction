from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SEVERITY_ORDER


MODEL_FEATURES = NUMERIC_COLUMNS + [
    "year",
    "month",
    "day",
    "hour",
    "casualties_per_vehicle",
] + CATEGORICAL_COLUMNS


@dataclass
class TrainingArtifacts:
    bundle: dict
    metrics: dict


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [column for column in MODEL_FEATURES if column in df.columns] + ["severity_label"]
    frame = df[selected_columns].copy()
    return frame.dropna(subset=["severity_label"])


def build_model_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_columns = [
        column
        for column in feature_columns
        if column in NUMERIC_COLUMNS + ["year", "month", "day", "hour", "casualties_per_vehicle"]
    ]
    categorical_columns = [column for column in feature_columns if column in CATEGORICAL_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(SEVERITY_ORDER),
                    eval_metric="mlogloss",
                    tree_method="hist",
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def fit_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
    return pipeline


def train_model(df: pd.DataFrame) -> TrainingArtifacts:
    frame = build_training_frame(df)
    feature_columns = [column for column in frame.columns if column != "severity_label"]
    X = frame[feature_columns]
    y = frame["severity_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_model_pipeline(feature_columns)
    pipeline = fit_model(pipeline, X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "model_name": "XGBoost",
        "model_params": pipeline.named_steps["model"].get_params(),
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=SEVERITY_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist(),
        "feature_columns": feature_columns,
        "target_mapping": {name: index for index, name in enumerate(SEVERITY_ORDER)},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    bundle = {
        "model_name": "XGBoost",
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "target_mapping": metrics["target_mapping"],
    }
    return TrainingArtifacts(bundle=bundle, metrics=metrics)


def save_training_outputs(bundle_path: str, report_path: str, artifacts: TrainingArtifacts) -> None:
    joblib.dump(artifacts.bundle, bundle_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(artifacts.metrics, handle, indent=2)


def load_model_bundle(path: str) -> dict:
    return joblib.load(path)


def predict_severity(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    pipeline = bundle["pipeline"]
    usable = df.copy()
    for column in feature_columns:
        if column not in usable.columns:
            usable[column] = pd.NA

    predictions = pipeline.predict(usable[feature_columns])
    probabilities = pipeline.predict_proba(usable[feature_columns])
    inverse_mapping = {value: key for key, value in bundle["target_mapping"].items()}

    result = usable.copy()
    result["predicted_severity_label"] = predictions
    result["predicted_severity"] = [inverse_mapping[pred] for pred in predictions]
    result["prediction_confidence"] = probabilities.max(axis=1)
    return result


def predict_severity_probabilities(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    pipeline = bundle["pipeline"]
    usable = df.copy()
    for column in feature_columns:
        if column not in usable.columns:
            usable[column] = pd.NA

    probabilities = pipeline.predict_proba(usable[feature_columns])
    probability_frame = pd.DataFrame(
        probabilities,
        columns=["predicted_fatal_risk", "predicted_serious_risk", "predicted_slight_risk"],
        index=usable.index,
    )
    probability_frame["predicted_severe_risk"] = (
        probability_frame["predicted_fatal_risk"] + probability_frame["predicted_serious_risk"]
    )
    return probability_frame
