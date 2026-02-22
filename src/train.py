import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# File paths
DATA_PATH = Path("data/processed/processed_accident_data.csv")
MODEL_PATH = Path("models/accident_model.pkl")


def load_data():
    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH)
    return df


def train_model(df):
    print("Training model...")

    # Separate target
    y = df["accident_severity"]

    # Drop target column
    X = df.drop("accident_severity", axis=1)

    # Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred))

    return model


def save_model(model):
    print("Saving model...")
    joblib.dump(model, MODEL_PATH)


def main():
    df = load_data()
    model = train_model(df)
    save_model(model)
    print("Training completed successfully.")


if __name__ == "__main__":
    main()