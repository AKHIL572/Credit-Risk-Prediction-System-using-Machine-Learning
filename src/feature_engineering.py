"""
Feature engineering for the credit risk project.

Matches the logic documented and justified in:
- notebooks/1_data_understanding.ipynb, Section 6 (dti sentinel value,
  annual_inc outlier)
- notebooks/3_data_preprocessing.ipynb, Sections 4-5a (fixes applied,
  ratio features capped)
- notebooks/4_modeling.ipynb, Section 3 (emp_length ordinal encoding)

IMPORTANT -- caps must be computed on TRAINING data only, then reused
(not recomputed) at inference time, or predict.py will silently leak
information from the training distribution into how a new batch is capped,
and caps computed on a small inference batch would be unstable. Call
`engineer_features(train_df)` once with `caps=None` during training, save
`df.attrs["caps"]` (see src/train.py), then load and pass that same dict as
`caps=` for every subsequent call (test set, or predict.py).
"""

import numpy as np
import pandas as pd

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the binary target from an already-resolved loan_status column."""
    df = df.copy()
    if "loan_status" in df.columns:
        df["target"] = df["loan_status"].map({
            "Fully Paid": 0,
            "Charged Off": 1,
            "Default": 1
        })
    return df


def engineer_features(df: pd.DataFrame, caps: dict = None) -> pd.DataFrame:
    """
    Creates every derived feature used by the models, applying the data
    quality fixes identified in notebook 1's raw-data audit.

    Parameters
    ----------
    df : pd.DataFrame
    caps : dict, optional
        Percentile cap values computed on training data (see module
        docstring). If None, caps are computed fresh from this dataframe --
        appropriate when this IS the training call, not appropriate for
        scoring new data at inference time.

    Returns
    -------
    pd.DataFrame with engineered features added. The caps actually used are
    attached at `result.attrs["caps"]` -- read this after the first
    (training) call and persist it for reuse.
    """
    df = df.copy()

    # --- Type corrections ---
    if df["int_rate"].dtype == object:
        df["int_rate"] = pd.to_numeric(
            df["int_rate"].astype(str).str.replace("%", "", regex=False), errors="coerce"
        )
    if df["revol_util"].dtype == object:
        df["revol_util"] = pd.to_numeric(
            df["revol_util"].astype(str).str.replace("%", "", regex=False), errors="coerce"
        )

    # --- dti sentinel fix (notebook 1, Section 6) ---
    # -1 is a LendingClub placeholder for "not calculable", not a real value.
    df.loc[df["dti"] == -1, "dti"] = np.nan
    df["dti_capped"] = df["dti"].clip(upper=100)

    # --- annual_inc winsorization (notebook 1, Section 6 / notebook 3, Section 4) ---
    caps = dict(caps) if caps else {}
    if "annual_inc_floor" not in caps:
        caps["annual_inc_floor"] = max(
            float(df["annual_inc"].quantile(0.01)), 1000.0)
    if "annual_inc_cap" not in caps:
        caps["annual_inc_cap"] = float(df["annual_inc"].quantile(0.99))
    # Safety guard: on pathologically skewed data the 99th percentile could in
    # principle fall below the floor, which would make clip() behave
    # unpredictably. Not expected on real income data, but cheap to guard.
    if caps["annual_inc_cap"] < caps["annual_inc_floor"]:
        caps["annual_inc_cap"] = caps["annual_inc_floor"]

    df["annual_inc_capped"] = df["annual_inc"].clip(
        lower=caps["annual_inc_floor"], upper=caps["annual_inc_cap"]
    )

    # --- Standard engineered features ---
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["loan_term_numeric"] = pd.to_numeric(
        df["term"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    df["loan_to_income"] = df["loan_amnt"] / \
        df["annual_inc_capped"].replace(0, np.nan)
    df["installment_to_income"] = df["installment"] / \
        (df["annual_inc_capped"].replace(0, np.nan) / 12)

    # --- Ratio outlier capping (notebook 3, Section 5a) ---
    # Direct cap on the ratios themselves -- robust regardless of exactly
    # which end of annual_inc is driving a given extreme ratio.
    if "loan_to_income_cap" not in caps:
        caps["loan_to_income_cap"] = float(df["loan_to_income"].quantile(0.99))
    if "installment_to_income_cap" not in caps:
        caps["installment_to_income_cap"] = float(
            df["installment_to_income"].quantile(0.99))

    df["loan_to_income"] = df["loan_to_income"].clip(
        upper=caps["loan_to_income_cap"])
    df["installment_to_income"] = df["installment_to_income"].clip(
        upper=caps["installment_to_income_cap"])

    # --- emp_length: numeric ordinal instead of one-hot (notebook 4, Section 3) ---
    df["emp_length_numeric"] = df["emp_length"].map(EMP_LENGTH_MAP)

    df.attrs["caps"] = caps
    return df
