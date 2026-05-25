from __future__ import annotations

import pandas as pd

from src.paths import CLEAN_DATA_PATH, FEATURE_DATA_PATH, ensure_directories
from src.preprocessing import create_model_features


def load_clean_data(path=CLEAN_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["accident_date"] = pd.to_datetime(df["accident_date"], errors="coerce")
    df["time"] = pd.to_datetime(
        df["time"].astype("string").str.strip().str[-8:],
        format="%H:%M:%S",
        errors="coerce",
    )
    return df


def save_feature_data(df: pd.DataFrame, path=FEATURE_DATA_PATH) -> None:
    serializable = df.copy()
    serializable["accident_date"] = serializable["accident_date"].dt.strftime("%Y-%m-%d")
    serializable["time"] = serializable["time"].dt.strftime("%H:%M:%S")
    serializable.to_csv(path, index=False)


def main() -> None:
    ensure_directories()
    clean_df = load_clean_data()
    feature_df = create_model_features(clean_df)
    save_feature_data(feature_df)
    print(f"Saved feature dataset to {FEATURE_DATA_PATH}")


if __name__ == "__main__":
    main()
