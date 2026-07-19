"""
Data loading for the credit risk project.

Matches the logic documented and justified in:
- notebooks/1_data_understanding.ipynb, Section 8 (resolved status filtering,
  credit-policy-exception mapping)
- notebooks/3_data_preprocessing.ipynb, Section 1 (leakage-safe column loading)
"""

import pandas as pd
from typing import List

RESOLVED_STATUSES = ["Fully Paid", "Charged Off", "Default"]

# Legacy loans issued under a since-retired credit policy. Their underlying
# repayment outcome is known and valid -- decision documented in notebook 1,
# Section 8: treat as equivalent to the underlying resolved status.
POLICY_EXCEPTION_MAP = {
    "Does not meet the credit policy. Status:Fully Paid": "Fully Paid",
    "Does not meet the credit policy. Status:Charged Off": "Charged Off",
}


def load_lendingclub_data(
    file_path: str,
    required_cols: List[str],
    chunk_size: int = 100_000
) -> pd.DataFrame:
    """
    Load the LendingClub dataset efficiently using chunking, restricted to
    loans with a resolved outcome (Fully Paid / Charged Off / Default,
    including credit-policy-exception loans mapped to their underlying
    status).

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file.
    required_cols : List[str]
        Columns to load. Must include "loan_status". If a chronological
        (out-of-time) train/test split is needed downstream, "issue_d" must
        also be included here -- it is not a model feature and must be
        dropped from X before training (see src/feature_engineering.py and
        src/train.py).
    chunk_size : int
        Rows per chunk. Lower this if you hit a memory error on a
        constrained machine.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only resolved-outcome loans, with
        `loan_status` already mapped to its resolved-status form.
    """
    if "loan_status" not in required_cols:
        raise ValueError("required_cols must include 'loan_status'.")

    chunks = []
    for chunk in pd.read_csv(
        file_path,
        usecols=required_cols,
        chunksize=chunk_size,
        low_memory=False
    ):
        chunk["loan_status"] = chunk["loan_status"].replace(
            POLICY_EXCEPTION_MAP)
        chunk = chunk[chunk["loan_status"].isin(RESOLVED_STATUSES)]
        chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid data found after filtering loan_status.")

    return pd.concat(chunks, ignore_index=True)
