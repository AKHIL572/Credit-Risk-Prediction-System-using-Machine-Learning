import numpy as np
import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary target variable from loan_status."""
    if "loan_status" in df.columns:
        df["target"] = df["loan_status"].map({
            "Fully Paid": 0,
            "Charged Off": 1,
            "Default": 1
        })
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived features matching the notebook preprocessing."""

    # Clean types first
    if df["int_rate"].dtype == object:
        df["int_rate"] = pd.to_numeric(
            df["int_rate"].str.replace("%", "", regex=False), errors="coerce"
        )
    if df["revol_util"].dtype == object:
        df["revol_util"] = pd.to_numeric(
            df["revol_util"].str.replace("%", "", regex=False), errors="coerce"
        )

    df["loan_to_income"] = df["loan_amnt"] / \
        df["annual_inc"].replace(0, np.nan)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["loan_term_numeric"] = pd.to_numeric(
        df["term"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    df["installment_to_income"] = df["installment"] / \
        (df["annual_inc"].replace(0, np.nan) / 12)
    df["dti_capped"] = df["dti"].clip(upper=100)

    return df
