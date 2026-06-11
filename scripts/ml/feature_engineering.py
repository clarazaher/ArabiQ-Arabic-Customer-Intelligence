"""
Feature engineering for the ArabiQ customer churn prediction model.

This script:
1. Loads the raw IBM Telco Customer Churn dataset.
2. Removes columns that would cause data leakage.
3. Converts the target column into a binary value.
4. Creates new business-friendly predictive features.
5. Saves the processed dataset to data/processed/churn_engineered.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/telco_churn_ibm.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/churn_engineered.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations to the raw churn dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw churn dataset.

    Returns
    -------
    pd.DataFrame
        Engineered dataframe ready for preprocessing/model training.
    """

    df = df.copy()

    # -----------------------------
    # 1. Remove leakage and ID columns
    # -----------------------------
    # Leakage means columns that give the model information it would not have
    # at prediction time. This creates fake high performance.
    leakage_or_id_cols = [
        # Direct identifiers
        "CustomerID",

        # Leakage columns: not safe to use for prediction
        "Churn Score",
        "CLTV",
        "Churn Reason",
        "Churn Value",

        # Constant / low-business-value columns
        "Count",
        "Country",
        "State",

        # Geographic and demographic columns excluded for ethical deployment
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Gender",
        "Senior Citizen",
    ]

    existing_cols_to_drop = [col for col in leakage_or_id_cols if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    # -----------------------------
    # 2. Encode target variable
    # -----------------------------
    if "Churn Label" not in df.columns:
        raise ValueError("Expected target column 'Churn Label' was not found.")

    df["churn"] = (df["Churn Label"] == "Yes").astype(int)
    df = df.drop(columns=["Churn Label"])

    # -----------------------------
    # 3. Clean numeric columns
    # -----------------------------
    # Total Charges sometimes behaves like text because of spaces/missing values.
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(0)

    # -----------------------------
    # 4. Tenure segment
    # -----------------------------
    # Converts a number into a customer lifecycle group.
    if "Tenure Months" in df.columns:
        df["tenure_segment"] = pd.cut(
            df["Tenure Months"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["New", "Developing", "Established", "Loyal"],
        )

    # -----------------------------
    # 5. Charge per tenure month
    # -----------------------------
    # +1 prevents division by zero when tenure is 0.
    if {"Monthly Charges", "Tenure Months"}.issubset(df.columns):
        df["charge_per_month"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)

    # -----------------------------
    # 6. Number of active services
    # -----------------------------
    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]

    available_service_cols = [col for col in service_cols if col in df.columns]

    for col in available_service_cols:
        df[f"{col}_binary"] = (df[col] == "Yes").astype(int)

    if available_service_cols:
        binary_service_cols = [f"{col}_binary" for col in available_service_cols]
        df["num_services"] = df[binary_service_cols].sum(axis=1)

    # -----------------------------
    # 7. Paperless billing flag
    # -----------------------------
    if "Paperless Billing" in df.columns:
        df["paperless"] = (df["Paperless Billing"] == "Yes").astype(int)

    # -----------------------------
    # 8. Contract risk score
    # -----------------------------
    # Higher value = safer/longer contract.
    if "Contract" in df.columns:
        contract_map = {
            "Month-to-month": 0,
            "Month-to-Month": 0,
            "One year": 1,
            "One Year": 1,
            "Two year": 2,
            "Two Year": 2,
        }

        df["contract_risk"] = df["Contract"].map(contract_map)

        if df["contract_risk"].isna().any():
            unknown_contracts = df.loc[df["contract_risk"].isna(), "Contract"].unique()
            raise ValueError(f"Unknown contract values found: {unknown_contracts}")

    # -----------------------------
    # 9. High-value at-risk flag
    # -----------------------------
    if {"Monthly Charges", "Contract"}.issubset(df.columns):
        median_charge = df["Monthly Charges"].median()
        df["high_value_at_risk"] = (
            (df["Monthly Charges"] > median_charge)
            & (df["Contract"].isin(["Month-to-month", "Month-to-Month"]))
        ).astype(int)

    return df


def main() -> None:
    """Run the full feature engineering process."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. "
            "Make sure data/raw/telco_churn_ibm.csv exists."
        )

    print("Loading raw dataset from:")
    print(RAW_DATA_PATH)

    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw dataset shape: {df_raw.shape}")

    df_engineered = engineer_features(df_raw)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nFeature engineering completed successfully.")
    print(f"Processed dataset shape: {df_engineered.shape}")
    print(f"Saved processed dataset to: {PROCESSED_DATA_PATH}")

    print("\nTarget distribution:")
    print(df_engineered["churn"].value_counts())

    print("\nNew engineered columns:")
    engineered_cols = [
        "churn",
        "tenure_segment",
        "charge_per_month",
        "num_services",
        "paperless",
        "contract_risk",
        "high_value_at_risk",
    ]
    for col in engineered_cols:
        if col in df_engineered.columns:
            print(f"- {col}")


if __name__ == "__main__":
    main()
