import pandas as pd
from pathlib import Path

# Define file paths
RAW_PATH = Path("data/raw/Raw Road Accident Data.xlsx")
OUTPUT_PATH = Path("data/processed/cleaned_accident_data.csv")


def load_data():
    print("Loading raw data...")
    df = pd.read_excel(RAW_PATH)
    return df


def basic_cleaning(df):
    print("Cleaning data...")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop rows with all missing values
    df = df.dropna(how="all")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


def save_data(df):
    print("Saving cleaned data...")
    df.to_csv(OUTPUT_PATH, index=False)


def main():
    df = load_data()
    df = basic_cleaning(df)
    save_data(df)
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()