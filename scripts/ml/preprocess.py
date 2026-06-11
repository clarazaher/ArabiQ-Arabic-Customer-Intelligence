"""
Preprocessing pipeline for the ArabiQ customer churn prediction model.

This script:
1. Loads the engineered churn dataset.
2. Separates features (X) from the target (y).
3. Splits the data into train and test sets.
4. Scales numeric columns.
5. One-hot encodes categorical columns.
6. Saves the fitted preprocessor for later use in model training and SHAP analysis.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINEERED_DATA_PATH = PROJECT_ROOT / "data/processed/churn_engineered.csv"
PREPROCESSOR_PATH = PROJECT_ROOT / "models/ml/preprocessor.pkl"


def build_preprocessing_pipeline(df: pd.DataFrame):
    """
    Build and apply a preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered churn dataset containing a binary target column called 'churn'.

    Returns
    -------
    tuple
        X_train_processed, X_test_processed, y_train, y_test, preprocessor
    """

    if "churn" not in df.columns:
        raise ValueError("Expected target column 'churn' was not found.")

    # X = input features, y = target answer.
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Identify categorical and numeric columns.
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print("Number of rows:", len(df))
    print("Number of features before preprocessing:", X.shape[1])
    print("\nCategorical columns:")
    print(categorical_cols)
    print("\nNumeric columns:")
    print(numeric_cols)

    # Numeric columns are scaled.
    # Categorical columns are one-hot encoded.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    # Stratify keeps the churn ratio similar in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain target distribution:")
    print(y_train.value_counts(normalize=True).round(3))

    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    # Very important:
    # fit_transform on train = learn scaling/encoding rules from train data.
    # transform on test = apply the same rules to test data without learning from it.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nProcessed train shape:", X_train_processed.shape)
    print("Processed test shape:", X_test_processed.shape)

    # Save the fitted preprocessor so the exact same transformations can be reused later.
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\nPreprocessor saved to:")
    print(PREPROCESSOR_PATH)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def main() -> None:
    """Run preprocessing as a standalone script."""

    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. "
            "Run scripts/ml/feature_engineering.py first."
        )

    print("Loading engineered dataset from:")
    print(ENGINEERED_DATA_PATH)

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    print("Engineered dataset shape:", df.shape)

    build_preprocessing_pipeline(df)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
