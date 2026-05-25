from __future__ import annotations

import pandas as pd

from src.paths import CLEAN_DATA_PATH, RAW_DATA_PATH, ensure_directories
from src.preprocessing import clean_raw_dataframe
from src.schema import CANONICAL_COLUMNS


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def save_clean_data(df: pd.DataFrame, path=CLEAN_DATA_PATH) -> None:
    serializable = df.copy()
    serializable["accident_date"] = serializable["accident_date"].dt.strftime("%Y-%m-%d")
    serializable["time"] = serializable["time"].dt.strftime("%H:%M:%S")
    serializable.to_csv(path, index=False)


def main() -> None:
    ensure_directories()
    raw_df = load_raw_data()
    clean_df, issues = clean_raw_dataframe(raw_df)
    save_clean_data(clean_df[CANONICAL_COLUMNS])

    print(f"Saved cleaned dataset to {CLEAN_DATA_PATH}")
    for issue in issues:
        print(issue)


if __name__ == "__main__":
    main()
