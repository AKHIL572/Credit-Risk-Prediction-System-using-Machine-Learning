"""
Preprocessing pipeline construction for the credit risk project.
"""

from typing import Tuple, List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Columns that must never be treated as model features even if present in a
# dataframe passed here by mistake (identifiers, the raw target, or columns
# kept only for time-based splitting).
NON_FEATURE_COLS = {"id", "loan_status", "target", "issue_d", "issue_d_parsed"}


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical columns, excluding known non-feature
    columns as a safety guard.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    num_cols : List[str]
    cat_cols : List[str]
    """
    usable_cols = [c for c in X.columns if c not in NON_FEATURE_COLS]

    num_cols = X[usable_cols].select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    cat_cols = X[usable_cols].select_dtypes(
        include=["object", "category"]).columns.tolist()

    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numeric: median imputation -> standard scaling
    Categorical: most-frequent imputation -> one-hot encoding (sparse)

    Both are fit only on training data when used inside a full sklearn
    Pipeline with cross_val_score / .fit() -- this is what keeps imputation
    and encoding leak-safe (see notebooks/4_modeling.ipynb intro).

    Returns
    -------
    ColumnTransformer
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor
