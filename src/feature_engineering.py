import pandas as pd

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary target variable from loan_status.
    """
    if "loan_status" in df.columns:
        df["target"] = df["loan_status"].map({
            "Fully Paid": 0,
            "Charged Off": 1,
            "Default": 1
        })
    return df
