import pandas as pd
from typing import List


def load_lendingclub_data(
    file_path: str,
    required_cols: List[str],
    chunk_size: int = 100_000
) -> pd.DataFrame:
    """
    Load Lending Club dataset efficiently using chunking.

    Parameters
    ----------
    file_path : str
        Path to CSV file
    required_cols : List[str]
        Columns needed for modeling
    chunk_size : int
        Number of rows per chunk

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with selected columns and valid targets
    """

    chunks = []

    for chunk in pd.read_csv(
        file_path,
        usecols=required_cols,
        chunksize=chunk_size,
        low_memory=False
    ):
        # Keep only clear loan outcomes
        chunk = chunk[
            chunk["loan_status"].isin(
                ["Fully Paid", "Charged Off", "Default"]
            )
        ]

        chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid data found after filtering loan_status.")

    return pd.concat(chunks, ignore_index=True)
