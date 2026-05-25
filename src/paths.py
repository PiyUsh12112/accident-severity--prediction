from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_DATA_PATH = RAW_DIR / "Raw Road Accident Data.xlsx"
CLEAN_DATA_PATH = PROCESSED_DIR / "cleaned_accident_data.csv"
FEATURE_DATA_PATH = PROCESSED_DIR / "model_features.csv"
ZONE_RISK_PATH = PROCESSED_DIR / "zone_risk_scores.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "accident_model_bundle.pkl"
EVALUATION_REPORT_PATH = REPORTS_DIR / "training_metrics.json"
DATA_DICTIONARY_PATH = DOCS_DIR / "data_dictionary.csv"


def ensure_directories() -> None:
    for path in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
