import pandas as pd
from pathlib import Path

# Define file paths
INPUT_PATH = Path("data/processed/cleaned_accident_data.csv")
OUTPUT_PATH = Path("data/processed/processed_accident_data.csv")


def load_data():
    print("Loading cleaned data...")
    df = pd.read_csv(INPUT_PATH)
    return df


def create_features(df):
    print("Creating features...")

    # Example: if you have a date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["hour"] = df["date"].dt.hour

    # Example engineered feature
    if "number_of_casualties" in df.columns and "number_of_vehicles" in df.columns:
        df["casualties_per_vehicle"] = (
            df["number_of_casualties"] / df["number_of_vehicles"]
        )

    return df


def save_data(df):
    print("Saving processed data...")
    df.to_csv(OUTPUT_PATH, index=False)


def main():
    df = load_data()
    df = create_features(df)
    save_data(df)
    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()